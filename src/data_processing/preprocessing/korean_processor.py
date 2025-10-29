from typing import List, Union, Dict
from .text_cleaner import (
    normalize_unicode,
    remove_extra_whitespace,
    normalize_punctuation,
    remove_special_chars_korean,
    is_empty_or_invalid
)


DEFAULT_SLANG_MAPPINGS = {
    'ㅎㅇ': '안녕',
    'ㄱㅅ': '고마워',
    'ㅋㅋㅋ': 'ㅋㅋ',
    'ㅋㅋㅋㅋ': 'ㅋㅋ',
    'ㅠㅠ': '',
    'ㅜㅜ': '',
    'ㅡㅡ': '',
    'ㅗㅗ': '',
    'ㄷㄷ': '',
    'ㅇㅇ': '응',
    'ㄴㄴ': '아니',
}


class KoreanProcessor:
    """
    Processes Korean text with appropriate normalization.
    Preserves Korean character structure and meaning.
    """

    def __init__(self, slang_dict: Dict[str, str] = None):
        self.slang_dict = slang_dict or DEFAULT_SLANG_MAPPINGS

    def normalize_slang(self, text: str) -> str:
        """Replace common Korean slang with standard forms."""
        for slang, replacement in self.slang_dict.items():
            text = text.replace(slang, replacement)
        return text

    def clean_text(self, text: str) -> str:
        """Apply Korean-specific cleaning operations."""
        if is_empty_or_invalid(text):
            return ""

        # Step 1: Unicode normalization (critical for Korean)
        text = normalize_unicode(text)

        # Step 2: Remove unwanted special characters
        text = remove_special_chars_korean(text)

        # Step 3: Normalize slang
        text = self.normalize_slang(text)

        # Step 4: Normalize punctuation
        text = normalize_punctuation(text)

        # Step 5: Normalize whitespace
        text = remove_extra_whitespace(text)
        return text

    def process_single(self, text: str) -> str:
        """Process a single Korean text."""
        return self.clean_text(text)

    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of Korean texts."""
        return [self.process_single(text) for text in texts]

    def __call__(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """Make processor callable."""
        if isinstance(texts, str):
            return self.process_single(texts)
        return self.process_batch(texts)
