from typing import List, Union
from .text_cleaner import (
    remove_extra_whitespace,
    normalize_punctuation,
    remove_special_chars_english,
    is_empty_or_invalid
)


class EnglishProcessor:
    """
    Processes English text with minimal preprocessing.
    Designed to preserve information for cased multilingual models.
    """

    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """Apply cleaning operations in sequence."""
        if is_empty_or_invalid(text):
            return ""
        
        text = remove_special_chars_english(text)
        text = normalize_punctuation(text)
        text = remove_extra_whitespace(text)
        return text

    def process_single(self, text: str) -> str:
        """Process a single English text."""
        return self.clean_text(text)

    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of English texts."""
        return [self.process_single(text) for text in texts]

    def __call__(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(texts, str):
            return self.process_single(texts)
        return self.process_batch(texts)
