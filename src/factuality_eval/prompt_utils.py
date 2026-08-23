"""Utilities for loading and formatting prompts. Adapted from LettuceDetect."""

from __future__ import annotations

import typing as t
from pathlib import Path
from string import Template

# Type for supported languages
Lang = t.Literal[
    "be",
    "bs",
    "bg",
    "ca",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fo",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "is",
    "it",
    "lb",
    "lv",
    "lt",
    "no",
    "pl",
    "pt",
    "ro",
    "sr",
    "sk",
    "sl",
    "sq",
    "es",
    "sv",
    "uk",
]

LANG_TO_PASSAGE = {
    "be": "уривок",  # Belarusian
    "bs": "odlomak",  # Bosnian
    "bg": "пасаж",  # Bulgarian
    "ca": "passatge",  # Catalan
    "hr": "odlomak",  # Croatian
    "cs": "pasáž",  # Czech
    "da": "afsnit",  # Danish
    "nl": "passage",  # Dutch
    "en": "passage",  # English
    "et": "lõik",  # Estonian
    "fo": "grein",  # Faroese
    "fi": "kappale",  # Finnish
    "fr": "passage",  # French
    "de": "Passage",  # German
    "el": "απόσπασμα",  # Greek
    "hu": "szövegrészlet",  # Hungarian
    "is": "efnisgrein",  # Icelandic
    "it": "brano",  # Italian
    "lb": "passage",  # Luxembourgish
    "lv": "posms",  # Latvian
    "lt": "ištrauka",  # Lithuanian
    "no": "avsnitt",  # Norwegian
    "pl": "fragment",  # Polish
    "pt": "passagem",  # Portuguese
    "ro": "pasaj",  # Romanian
    "sr": "одломак",  # Serbian
    "sk": "pasáž",  # Slovak
    "sl": "odlomek",  # Slovenian
    "es": "pasaje",  # Spanish
    "sv": "stycke",  # Swedish
    "uk": "уривок",  # Ukrainian
}

LANG_TO_FULL_NAME = {
    "be": "Belarusian",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fo": "Faroese",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "is": "Icelandic",
    "it": "Italian",
    "lb": "Luxembourgish",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "es": "Spanish",
    "sv": "Swedish",
    "uk": "Ukrainian",
}


PROMPT_DIR = Path(__file__).parent.parent / "prompts"


class PromptUtils:
    """Utility class for loading and formatting prompts."""

    @staticmethod
    def load_prompt(filename: str) -> Template:
        """Load a prompt template from the prompts directory.

        Args:
            filename:
                Name of the prompt file.

        Returns:
            Template object for the prompt.

        Raises:
            FileNotFoundError:
                If the prompt file doesn't exist.
        """
        path = PROMPT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return Template(path.read_text(encoding="utf-8"))

    @staticmethod
    def format_context(context: list[str], question: str | None, lang: Lang) -> str:
        """Format context and question into a prompt.

        Args:
            context:
                List of passages.
            question:
                The question, or None for summarization tasks.
            lang:
                The language code.

        Returns:
            Formatted prompt.
        """
        passage_word = LANG_TO_PASSAGE[lang]
        ctx_block = "\n".join(
            f"{passage_word} {i + 1}: {p}" for i, p in enumerate(context)
        )

        tmpl = PromptUtils.load_prompt(f"qa_prompt_{lang.lower()}.txt")
        return tmpl.substitute(question=question, text=ctx_block)

    @staticmethod
    def get_full_language_name(lang: Lang) -> str:
        """Get the full language name for a language code.

        Args:
            lang: Language code.

        Returns:
            Full language name.
        """
        return LANG_TO_FULL_NAME.get(lang, "Unknown")
