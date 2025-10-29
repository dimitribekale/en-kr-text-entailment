import pandas as pd
from typing import Dict
from .data_validator import (
    has_required_columns,
    get_invalid_language_mask,
    get_invalid_hypothesis_mask,
    get_language_distribution
)


class DataCleaner:
    """Clean and filter multilingual textual entailment datasets."""

    def __init__(self):
        self.stats = {}

    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        self.stats['original_size'] = len(df)
        print(f"[OK] Loaded {len(df):,} rows from {filepath}")
        return df

    def validate_structure(self, df: pd.DataFrame) -> None:
        """Validate dataset has required columns."""
        required = ['language', 'hypothesis', 'premise', 'label']
        has_required_columns(df, required)
        print(f"[OK] Validated required columns: {required}")

    def remove_invalid_languages(self, df: pd.DataFrame) -> pd.DataFrame:
        
        print("\n→ Filtering languages...")

        # Show distribution before
        lang_dist = get_language_distribution(df)
        print(f"  Before: {dict(lang_dist)}")

        # Filter
        invalid_mask = get_invalid_language_mask(df)
        removed_count = invalid_mask.sum()
        df_clean = df[~invalid_mask].copy()

        # Show distribution after
        lang_dist_after = get_language_distribution(df_clean)
        print(f"  After: {dict(lang_dist_after)}")
        print(f"  Removed {removed_count:,} rows with invalid languages")

        self.stats['removed_by_language'] = removed_count
        return df_clean

    def remove_invalid_hypotheses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with problematic hypotheses."""
        print("\n→ Cleaning hypotheses...")

        df['hypothesis'] = df['hypothesis'].astype(str)

        invalid_mask = get_invalid_hypothesis_mask(df)
        removed_count = invalid_mask.sum()

        if removed_count > 0:
            print(f"  Found {removed_count:,} problematic hypotheses")
            samples = df[invalid_mask].head(3)
            for _, row in samples.iterrows():
                print(f"    ID {row['ID']}: '{row['hypothesis']}'")

        df_clean = df[~invalid_mask].copy()
        self.stats['removed_by_hypothesis'] = removed_count
        return df_clean

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate premise-hypothesis pairs."""
        print("\n→ Removing duplicates...")

        before = len(df)
        df_clean = df.drop_duplicates(subset=['premise', 'hypothesis']).copy()
        after = len(df_clean)
        removed = before - after

        print(f"  Removed {removed:,} duplicate pairs")
        self.stats['removed_duplicates'] = removed
        return df_clean

    def reset_index(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.reset_index(drop=True)

    def print_summary(self) -> None:
        """Print cleaning summary statistics."""
        print("\n" + "="*60)
        print("CLEANING SUMMARY")
        print("="*60)
        print(f"Original size:      {self.stats.get('original_size', 0):,}")
        print(f"By language:       -{self.stats.get('removed_by_language', 0):,}")
        print(f"By hypothesis:     -{self.stats.get('removed_by_hypothesis', 0):,}")
        print(f"By duplicates:     -{self.stats.get('removed_duplicates', 0):,}")
        print(f"Final size:         {self.stats.get('final_size', 0):,}")
        total_removed = (self.stats.get('original_size', 0) -
                        self.stats.get('final_size', 0))
        pct = (total_removed / self.stats.get('original_size', 1)) * 100
        print(f"Total removed:      {total_removed:,} ({pct:.2f}%)")
        print("="*60)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute complete cleaning pipeline."""
        print("\nStarting data cleaning pipeline...")

        self.validate_structure(df)
        df = self.remove_invalid_languages(df)
        df = self.remove_invalid_hypotheses(df)
        df = self.remove_duplicates(df)
        df = self.reset_index(df)

        self.stats['final_size'] = len(df)
        self.print_summary()

        return df

    def get_stats(self) -> Dict:
        """Get cleaning statistics."""
        return self.stats
