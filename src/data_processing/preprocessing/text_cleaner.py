import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to NFC form."""
    return unicodedata.normalize('NFC', text)


def remove_extra_whitespace(text: str) -> str:
    """Replace multiple whitespace with single space and strip."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_punctuation(text: str) -> str:
    """Reduce excessive punctuation repetition."""
    # Convert !!! or ?????? to !!
    return re.sub(r'([!?.]){3,}', r'\1\1', text)


def remove_special_chars_english(text: str) -> str:
    """Remove special characters while preserving English text structure."""
    # Keep: letters, numbers, basic punctuation
    pattern = re.compile(r'[^a-zA-Z0-9\s.,?!()\'\":-]')
    return pattern.sub('', text)


def remove_special_chars_korean(text: str) -> str:
    """Remove special characters while preserving Korean text structure."""
    # Keep: Korean (Hangul), English letters, numbers, basic punctuation
    pattern = re.compile(r'[^\uAC00-\uD7A3a-zA-Z0-9\s.,?!()\'\":-]')
    return pattern.sub('', text)


def is_empty_or_invalid(text: str) -> bool:
    """Check if text is empty, None, or just whitespace."""
    if not text or not isinstance(text, str):
        return True
    return len(text.strip()) == 0


def has_minimum_length(text: str, min_length: int = 2) -> bool:
    """Check if text meets minimum length requirement."""
    return len(str(text).strip()) >= min_length
