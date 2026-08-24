"""Train hallucination detector.

Usage:
    uv run src/scripts/train_hallucination_detector.py <config_key>=<config_value> ...
"""

import gc
import logging
import os
from pathlib import Path

# Reduce CUDA fragmentation on small (8GB) GPUs. Must be set before torch
# initialises the CUDA allocator. setdefault so an explicit env override wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from lettucedetect import HallucinationDataset
from lettucedetect.models.evaluator import (
    evaluate_detector_char_level,
    evaluate_model,
    evaluate_model_example_level,
    print_metrics,
)
from lettucedetect.models.inference import HallucinationDetector
from lettucedetect.models.trainer import Trainer
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from factuality_eval.dataset_generation import (
    generate_hallucinations_from_qa_data,
    generate_lettucedetect_hallucination_samples,
    load_qa_data,
    sample_hallucination_intensities,
)
from factuality_eval.logging_utils import capture_stdio_to_file, header, log
from factuality_eval.train import format_dataset_to_ragtruth

torch.set_float32_matmul_precision("high")
load_dotenv()
logger = logging.getLogger("train_hallucination_detector")


def load_ragtruth_translated(dataset_id: str, language: str) -> tuple[list, list]:
    """Load translated RAGTruth from the Hugging Face Hub and split by native RAGTruth split.

    The dataset (e.g. ``alexandrainst/ragtruth-translated-hallucinations``) has one
    config per language, each with a single ``train`` split; the original RAGTruth
    train/test partition lives in the ``split`` column.

    Args:
        dataset_id: Hugging Face dataset id, one config per language.
        language: Target language code, used as the config name and sample tag.

    Returns:
        Tuple of (train_samples, test_samples), each a list of sample dicts
        in RAGTruth format compatible with generate_lettucedetect_hallucination_samples.
    """
    ds = load_dataset(dataset_id, name=language, split="train")

    train_samples, test_samples = [], []
    for row in ds:
        sample_dict = {
            "prompt": row["prompt"],
            "answer": row["answer"],
            "labels": row["labels"],
            "split": row["split"],
            "task_type": row["task_type"],
            "dataset": row["dataset"],
            "language": row["language"],
        }
        if row["split"] == "train":
            train_samples.append(sample_dict)
        elif row["split"] == "test":
            test_samples.append(sample_dict)
        else:
            logger.warning(f"Unknown split '{row['split']}' on sample, skipping.")

    log(
        f"Loaded translated RAGTruth ({language}) from {dataset_id}: "
        f"{len(train_samples)} train, {len(test_samples)} test",
        level=logging.INFO,
    )
    return train_samples, test_samples


@hydra.main(
    config_path="../../config", config_name="hallucination_detection", version_base=None
)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    capture_stdio_to_file(hydra_output_dir / f"{HydraConfig.get().job.name}.log")

    target_dataset_name = f"{config.base_dataset.id}-synthetic-hallucinations"

    if not config.multiwikiqa.enable and not config.ragtruth.enable:
        raise ValueError(
            "Both multiwikiqa and ragtruth are disabled; nothing to train on. "
            "Enable at least one of config.multiwikiqa.enable / config.ragtruth.enable."
        )

    # ------------------------------------------------------------------
    # 1. Load / generate synthetic MultiWikiQA hallucination dataset
    # ------------------------------------------------------------------
    synthetic_train: Dataset | None = None
    synthetic_test: Dataset | None = None
    if config.multiwikiqa.enable:
        header("Preparing synthetic dataset", color="light_blue", level=logging.INFO)
        multiwikiqa_language = "pt-pt" if config.language == "pt" else config.language
        try:
            dataset = load_dataset(
                f"{config.hub_organisation}/{target_dataset_name}",
                name=multiwikiqa_language,
            )
        except ValueError:
            log(
                f"Language '{config.language}' not found in hub dataset "
                f"'{config.hub_organisation}/{target_dataset_name}'. "
                "Generating dataset locally and pushing to hub...",
                level=logging.INFO,
            )
            contexts, questions, answers = load_qa_data(
                base_dataset_id=(
                    f"{config.base_dataset.organisation}/{config.base_dataset.id}"
                    f":{multiwikiqa_language}"
                ),
                split=config.base_dataset.split,
                context_key=config.base_dataset.context_key,
                question_key=config.base_dataset.question_key,
                answer_key=config.base_dataset.answer_key,
                squad_format=config.base_dataset.squad_format,
                testing=config.testing,
            )
            intensities = sample_hallucination_intensities(
                mean=config.beta_distribution.mean,
                std=config.beta_distribution.std,
                size=len(answers),
            )
            generated = generate_hallucinations_from_qa_data(
                contexts=contexts,
                questions=questions,
                answers=answers,
                intensities=intensities,
                model=config.models.hallu_gen_model,
                reasoning_effort=getattr(
                    config.models, "hallu_gen_reasoning_effort", None
                ),
                output_jsonl_path=Path(
                    "data", "final", f"{target_dataset_name}-{config.language}.jsonl"
                ),
                max_workers=config.max_workers,
            )
            generated.push_to_hub(
                repo_id=f"{config.hub_organisation}/{target_dataset_name}",
                config_name=config.language,
                private=config.private,
            )
            dataset = load_dataset(
                f"{config.hub_organisation}/{target_dataset_name}", name=config.language
            )
        train_test_split = dataset["train"].train_test_split(
            test_size=0.2, seed=42, shuffle=False
        )

        synthetic_train = format_dataset_to_ragtruth(
            train_test_split["train"], language=config.language, split="train"
        )
        synthetic_test = format_dataset_to_ragtruth(
            train_test_split["test"], language=config.language, split="test"
        )
        log(
            f"Synthetic dataset: {len(synthetic_train)} train, "
            f"{len(synthetic_test)} test",
            level=logging.INFO,
        )

    # ------------------------------------------------------------------
    # 2. Load translated RAGTruth
    # ------------------------------------------------------------------
    ragtruth_train_ds: Dataset | None = None
    ragtruth_test_ds: Dataset | None = None
    if config.ragtruth.enable:
        header("Loading translated RAGTruth", color="light_blue", level=logging.INFO)
        ragtruth_train, ragtruth_test = load_ragtruth_translated(
            config.ragtruth.id, language=config.language
        )
        ragtruth_train_ds = Dataset.from_list(ragtruth_train)
        ragtruth_test_ds = Dataset.from_list(ragtruth_test)

        if config.multiwikiqa.enable:
            assert synthetic_train is not None and synthetic_test is not None
            train_dataset = concatenate_datasets([synthetic_train, ragtruth_train_ds])
            test_dataset = concatenate_datasets([synthetic_test, ragtruth_test_ds])
            logger.info(
                f"Combined dataset: {len(train_dataset)} train "
                f"({len(synthetic_train)} synthetic + {len(ragtruth_train)} ragtruth), "
                f"{len(test_dataset)} test "
                f"({len(synthetic_test)} synthetic + {len(ragtruth_test)} ragtruth)"
            )
        else:
            train_dataset = ragtruth_train_ds
            test_dataset = ragtruth_test_ds
            logger.info("No wiki in config; training on RAGTruth data only.")
    else:
        logger.info("RAGTruth disabled in config; training on synthetic data only.")
        train_dataset = synthetic_train
        test_dataset = synthetic_test

    # Shuffle the combined train/test datasets so RAGTruth and synthetic
    # MultiWikiQA examples are interleaved. This is done *after* the
    # train/test split so paired clean/hallucinated synthetic rows stay in
    # the same split, but the per-split ordering becomes random.
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)

    # ------------------------------------------------------------------
    # 4. Tokenize and train
    # ------------------------------------------------------------------
    header("Setting up tokenizer & model", color="light_blue", level=logging.INFO)
    tokenizer = AutoTokenizer.from_pretrained(
        config.models.pretrained_model, trust_remote_code=True
    )
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, label_pad_token_id=-100
    )

    max_length = config.training.max_length

    def _fits_in_max_length(example: dict) -> bool:
        # Mirror HallucinationDataset's tokenization (prompt + answer as a pair,
        # with special tokens) but without truncation, so we can drop samples
        # whose context would otherwise be silently truncated.
        encoded = tokenizer(
            example["prompt"],
            example["answer"],
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoded["input_ids"]) <= max_length

    train_before = len(train_dataset)
    test_before = len(test_dataset)
    train_dataset = train_dataset.filter(_fits_in_max_length)
    test_dataset = test_dataset.filter(_fits_in_max_length)
    logger.info(
        f"Filtered samples exceeding max_length={max_length}: "
        f"train {train_before} -> {len(train_dataset)} "
        f"({train_before - len(train_dataset)} dropped), "
        f"test {test_before} -> {len(test_dataset)} "
        f"({test_before - len(test_dataset)} dropped)"
    )

    train_hallu_dataset = HallucinationDataset(
        generate_lettucedetect_hallucination_samples(train_dataset),
        tokenizer,
        max_length=max_length,
    )
    test_hallu_dataset = HallucinationDataset(
        generate_lettucedetect_hallucination_samples(test_dataset),
        tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_hallu_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    test_loader = DataLoader(
        test_hallu_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    # Naming: include "+ragtruth" suffix so combined-vs-synthetic-only models
    # don't overwrite each other in the output dir / hub repo.
    if config.get("ragtruth", None) and config.ragtruth.get("enable", False):
        suffix = (
            "-with-ragtruth"
            if config.multiwikiqa.get("enable", False)
            else "-only-ragtruth"
        )
    else:
        suffix = ""

    model_save_path = (
        f"{config.training.output_dir}/"
        f"{config.models.hallu_detect_model}-{target_dataset_name}{suffix}-{config.language}"
    )
    cuda_is_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_is_available else "cpu")
    detector_device = (
        torch.device("cuda:1")
        if cuda_is_available and torch.cuda.device_count() > 1
        else device
    )
    log(
        f"CUDA available: {cuda_is_available} — using device: {device}",
        level=logging.INFO,
    )

    def _run_full_evaluation(eval_model: AutoModelForTokenClassification) -> None:
        """Evaluate at token, example, and span (char) level — matches LettuceDetect paper."""
        eval_model.eval()  # type: ignore[attr-defined]

        logger.info("\n---- Token-level ----")
        print_metrics(evaluate_model(eval_model, test_loader, device))

        logger.info(
            "\n---- Example-level (any hallucinated token => hallucinated example) ----"
        )
        print_metrics(evaluate_model_example_level(eval_model, test_loader, device))

        logger.info("\n---- Span / char-level (overlap with gold spans) ----")
        # HallucinationDetector reloads from disk, so point it at the saved model.
        detector = HallucinationDetector(
            method="transformer",
            model_path=model_save_path,
            trust_remote_code=True,
            device=detector_device,
        )
        test_samples = generate_lettucedetect_hallucination_samples(test_dataset)
        char_metrics = evaluate_detector_char_level(detector, test_samples)
        logger.info(
            f"  Precision: {char_metrics['precision']:.4f}  "
            f"Recall: {char_metrics['recall']:.4f}  "
            f"F1: {char_metrics['f1']:.4f}"
        )

    if os.path.exists(model_save_path) and os.path.isdir(model_save_path):
        header("Evaluating existing checkpoint", color="light_blue", level=logging.INFO)
        log(f"Loading existing model from {model_save_path}", level=logging.INFO)
        model = AutoModelForTokenClassification.from_pretrained(
            model_save_path, trust_remote_code=True, use_safetensors=True
        )
        model.to(device)

        logger.info("\nEvaluating...")
        _run_full_evaluation(model)
        model_to_push = model

    else:
        model = AutoModelForTokenClassification.from_pretrained(
            config.models.pretrained_model, num_labels=2, trust_remote_code=True
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
            save_path=model_save_path,
            device=device,
        )

        header("Fine-tuning", color="light_blue", level=logging.INFO)
        log("Starting training...", level=logging.INFO)
        trainer.train()

        # Drop optimizer state + training-time model copy off the GPU before
        # loading the best checkpoint — avoids stacking multiple model copies
        # on one card (OOM on 8GB GPUs). Push now uses best_model, so `model`
        # is no longer needed.
        del trainer, model
        gc.collect()
        if cuda_is_available:
            torch.cuda.empty_cache()

        # Re-load the best checkpoint (Trainer saves the best-F1 model to save_path)
        # and run the full multi-level evaluation on it.
        best_model = AutoModelForTokenClassification.from_pretrained(
            model_save_path, trust_remote_code=True, use_safetensors=True
        ).to(device)
        logger.info("\nFinal evaluation on best checkpoint:")
        _run_full_evaluation(best_model)
        model_to_push = best_model

    if config.training.push_to_hub:
        header("Pushing to hub", color="light_blue", level=logging.INFO)
        hub_repo_id = (
            f"{config.hub_organisation}/"
            f"{config.models.hallu_detect_model}-{target_dataset_name}{suffix}-{config.language}"
        )
        model_to_push.push_to_hub(repo_id=hub_repo_id, private=config.private)
        tokenizer.push_to_hub(repo_id=hub_repo_id, private=config.private)


if __name__ == "__main__":
    main()
