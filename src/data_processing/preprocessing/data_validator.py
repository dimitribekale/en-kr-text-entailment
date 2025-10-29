import pandas as pd
from typing import List, Set


VALID_LANGUAGES = {'en', 'ko'}
PROBLEMATIC_STRINGS = {'nan', 'na', 'hm'}


def has_required_columns(df: pd.DataFrame, required: List[str]) -> bool:
    """Check if dataframe has all required columns."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def is_valid_language(language: str, valid_set: Set[str] = None) -> bool:
    """Check if language code is valid."""
    valid_set = valid_set or VALID_LANGUAGES
    return language in valid_set


def is_problematic_hypothesis(text: str, min_length: int = 5) -> bool:
    """
    Check if hypothesis is problematic and should be removed.

    Criteria:
    - NaN or null values
    - Single character
    - Short text with problematic strings
    """
    if pd.isna(text) or text == 'nan':
        return True

    text_str = str(text).strip()
    text_lower = text_str.lower()
    text_length = len(text_str)

    # Single character hypotheses are invalid
    if text_length == 1:
        return True

    # Short hypotheses with problematic strings
    if text_length < min_length:
        if any(prob in text_lower for prob in PROBLEMATIC_STRINGS):
            return True

    return False


def get_invalid_language_mask(df: pd.DataFrame, language_col: str = 'language') -> pd.Series:
    """Get boolean mask for rows with invalid languages."""
    return ~df[language_col].isin(VALID_LANGUAGES)


def get_invalid_hypothesis_mask(df: pd.DataFrame, hypothesis_col: str = 'hypothesis') -> pd.Series:
    """Get boolean mask for rows with invalid hypotheses."""
    return df[hypothesis_col].apply(is_problematic_hypothesis)


def count_nulls(df: pd.DataFrame) -> pd.Series:
    """Count null values in each column."""
    return df.isnull().sum()


def get_language_distribution(df: pd.DataFrame, language_col: str = 'language') -> pd.Series:
    """Get language distribution counts."""
    return df[language_col].value_counts()
