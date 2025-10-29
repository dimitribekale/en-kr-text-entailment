import pandas as pd
from .data_cleaner import DataCleaner
from .english_processor import EnglishProcessor
from .korean_processor import KoreanProcessor


class PreprocessingPipeline:
    """
    Orchestrates the complete preprocessing workflow.
    Handles cleaning, language-specific processing, and output.
    """

    def __init__(self):
        """Initialize pipeline with all processors."""
        self.cleaner = DataCleaner()
        self.english_processor = EnglishProcessor()
        self.korean_processor = KoreanProcessor()

    def load(self, input_path: str) -> pd.DataFrame:
        """Load dataset from file."""
        print(f"\n{'='*60}")
        print("LOADING DATA")
        print(f"{'='*60}")
        return self.cleaner.load_data(input_path)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data cleaning operations."""
        print(f"\n{'='*60}")
        print("CLEANING DATA")
        print(f"{'='*60}")
        return self.cleaner.clean(df)

    def process_english(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process English text samples."""
        print(f"\n{'='*60}")
        print("PROCESSING ENGLISH TEXT")
        print(f"{'='*60}")

        english_mask = df['language'] == 'en'
        count = english_mask.sum()

        if count > 0:
            print(f"Processing {count:,} English samples...")
            df.loc[english_mask, 'premise'] = self.english_processor(
                df.loc[english_mask, 'premise'].tolist()
            )
            df.loc[english_mask, 'hypothesis'] = self.english_processor(
                df.loc[english_mask, 'hypothesis'].tolist()
            )
            print(f"[OK] Completed English processing")
        else:
            print("No English samples found")

        return df

    def process_korean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Korean text samples."""
        print(f"\n{'='*60}")
        print("PROCESSING KOREAN TEXT")
        print(f"{'='*60}")

        korean_mask = df['language'] == 'ko'
        count = korean_mask.sum()

        if count > 0:
            print(f"Processing {count:,} Korean samples...")
            df.loc[korean_mask, 'premise'] = self.korean_processor(
                df.loc[korean_mask, 'premise'].tolist()
            )
            df.loc[korean_mask, 'hypothesis'] = self.korean_processor(
                df.loc[korean_mask, 'hypothesis'].tolist()
            )
            print(f"[OK] Completed Korean processing")
        else:
            print("No Korean samples found")

        return df

    def save(self, df: pd.DataFrame, output_path: str) -> None:
        """Save processed dataset to file."""
        print(f"\n{'='*60}")
        print("SAVING DATA")
        print(f"{'='*60}")

        df.to_csv(output_path, index=False)
        print(f"[OK] Saved {len(df):,} rows to {output_path}")

    def show_sample(self, df: pd.DataFrame, n: int = 3) -> None:
        """Display sample of processed data."""
        print(f"\n{'='*60}")
        print(f"SAMPLE OUTPUT (first {n} rows)")
        print(f"{'='*60}")

        for idx, row in df.head(n).iterrows():
            print(f"\nRow {idx}:")
            print(f"  Language: {row['language']}")
            print(f"  Premise: {row['premise'][:100]}...")
            print(f"  Hypothesis: {row['hypothesis'][:100]}...")
            print(f"  Label: {row['label']}")

    def run(self,
            input_path: str,
            output_path: str,
            show_samples: bool = True) -> pd.DataFrame:
        """
        Execute complete preprocessing pipeline.

        Steps:
        1. Load data
        2. Clean data (remove invalid rows)
        3. Process English text
        4. Process Korean text
        5. Save output
        6. Show samples (optional)
        """
        print(f"\n{'#'*60}")
        print("PREPROCESSING PIPELINE STARTED")
        print(f"{'#'*60}")

        df = self.load(input_path)
        df = self.clean(df)
        df = self.process_english(df)
        df = self.process_korean(df)
        self.save(df, output_path)

        if show_samples:
            self.show_sample(df)

        print(f"\n{'#'*60}")
        print("PREPROCESSING PIPELINE COMPLETED")
        print(f"{'#'*60}\n")

        return df
