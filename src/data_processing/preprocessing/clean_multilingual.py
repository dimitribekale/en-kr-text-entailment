import pandas as pd
from typing import List, Optional, Dict, Any

class MultilingualCleaner:
    """
    A comprehensive class for cleaning multilingual textual entailment datasets.
    Handles language filtering and hypothesis cleaning with detailed reporting.
    """
    
    def __init__(self, valid_languages: List[str] = None, problematic_strings: List[str] = None):
        """
        Initialize the MultilingualCleaner.
        
        Args:
            valid_languages (List[str]): List of valid language codes (default: ['en', 'ko'])
            problematic_strings (List[str]): List of strings to remove from short hypotheses 
                                           (default: ['nan', 'na', 'hm'])
        """
        self.valid_languages = valid_languages or ['en', 'ko']
        self.problematic_strings = problematic_strings or ['nan', 'na', 'hm']
        self.original_size = 0
        self.cleaned_size = 0
        self.removed_by_language = 0
        self.removed_by_hypothesis = 0
        self.cleaning_stats = {}
        
    def load_dataset(self, input_path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Args:
            input_path (str): Path to input CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(input_path)
            self.original_size = len(df)
            print(f"Loaded dataset with {len(df):,} rows and {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {input_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def validate_dataset(self, df: pd.DataFrame) -> None:
        """
        Validate that the dataset has required columns.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['language', 'hypothesis']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Dataset must contain columns: {missing_columns}")
    
    def filter_by_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataset to keep only valid languages.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Language-filtered dataset
        """
        print("\n1. Filtering languages...")
        
        # Show current language distribution
        print("Current language distribution:")
        lang_dist = df['language'].value_counts()
        print(lang_dist)
        
        # Filter for valid languages
        df_filtered = df[df['language'].isin(self.valid_languages)].copy()
        self.removed_by_language = len(df) - len(df_filtered)
        
        print(f"Removed {self.removed_by_language:,} rows with invalid languages")
        
        # Show which languages were removed
        if self.removed_by_language > 0:
            removed_languages = df[~df['language'].isin(self.valid_languages)]['language'].value_counts()
            print("Languages that were removed:")
            print(removed_languages)
        
        return df_filtered
    
    def should_remove_hypothesis(self, text: Any) -> bool:
        """
        Check if hypothesis should be removed based on criteria.
        
        Args:
            text (Any): Hypothesis text to check
            
        Returns:
            bool: True if hypothesis should be removed
        """
        if pd.isna(text) or text == 'nan':
            return True
        
        text_str = str(text).strip()
        text_lower = text_str.lower()
        text_length = len(text_str)
        
        # Remove if length == 1
        if text_length == 1:
            return True
        
        # Remove if length < 5 AND contains specified strings
        if text_length < 5:
            if any(substring in text_lower for substring in self.problematic_strings):
                return True
        
        return False
    
    def clean_hypothesis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean hypothesis rows based on specified criteria.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Hypothesis-cleaned dataset
        """
        print("\n2. Cleaning hypothesis rows...")
        
        # Ensure hypothesis is string type
        df['hypothesis'] = df['hypothesis'].astype(str)
        
        # Apply removal criteria
        removal_mask = df['hypothesis'].apply(self.should_remove_hypothesis)
        rows_to_remove = df[removal_mask]
        
        print(f"Found {len(rows_to_remove):,} rows to remove based on hypothesis criteria")
        
        # Show sample problematic hypotheses
        if len(rows_to_remove) > 0:
            print("\nSample problematic hypotheses:")
            sample_problematic = rows_to_remove[['ID', 'hypothesis']].head(10)
            for _, row in sample_problematic.iterrows():
                print(f"  ID {row['ID']}: '{row['hypothesis']}'")
        
        # Remove problematic rows
        df_cleaned = df[~removal_mask].copy()
        self.removed_by_hypothesis = len(df) - len(df_cleaned)
        
        return df_cleaned
    
    def generate_cleaning_stats(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive cleaning statistics.
        
        Args:
            df_original (pd.DataFrame): Original dataset
            df_cleaned (pd.DataFrame): Cleaned dataset
            
        Returns:
            Dict[str, Any]: Cleaning statistics
        """
        self.cleaned_size = len(df_cleaned)
        total_removed = self.original_size - self.cleaned_size
        removal_percentage = (total_removed / self.original_size) * 100
        
        # Language distribution statistics
        original_lang_dist = df_original['language'].value_counts().to_dict()
        final_lang_dist = df_cleaned['language'].value_counts().to_dict()
        
        # Hypothesis length statistics
        hypothesis_lengths = df_cleaned['hypothesis'].str.len()
        
        self.cleaning_stats = {
            'original_size': self.original_size,
            'cleaned_size': self.cleaned_size,
            'total_removed': total_removed,
            'removal_percentage': removal_percentage,
            'removed_by_language': self.removed_by_language,
            'removed_by_hypothesis': self.removed_by_hypothesis,
            'original_language_distribution': original_lang_dist,
            'final_language_distribution': final_lang_dist,
            'hypothesis_length_stats': {
                'min': hypothesis_lengths.min(),
                'max': hypothesis_lengths.max(),
                'mean': hypothesis_lengths.mean(),
                'median': hypothesis_lengths.median()
            }
        }
        
        return self.cleaning_stats
    
    def print_summary(self) -> None:
        """Print comprehensive cleaning summary."""
        print(f"\n" + "="*50)
        print("FINAL CLEANING SUMMARY")
        print("="*50)
        print(f"Original size: {self.original_size:,} rows")
        print(f"Removed by language filtering: {self.removed_by_language:,} rows")
        print(f"Removed by hypothesis cleaning: {self.removed_by_hypothesis:,} rows")
        print(f"Final size: {self.cleaned_size:,} rows")
        print(f"Total removed: {self.original_size - self.cleaned_size:,} rows ({((self.original_size - self.cleaned_size)/self.original_size*100):.2f}%)")
        
        # Show final language distribution
        if 'final_language_distribution' in self.cleaning_stats:
            print(f"\nFinal language distribution:")
            for lang, count in self.cleaning_stats['final_language_distribution'].items():
                percentage = (count / self.cleaned_size) * 100
                print(f"  {lang}: {count:,} ({percentage:.1f}%)")
        
        # Show hypothesis length statistics
        if 'hypothesis_length_stats' in self.cleaning_stats:
            stats = self.cleaning_stats['hypothesis_length_stats']
            print(f"\nHypothesis length statistics in cleaned data:")
            print(f"  Min length: {stats['min']}")
            print(f"  Max length: {stats['max']}")
            print(f"  Mean length: {stats['mean']:.1f}")
            print(f"  Median length: {stats['median']:.1f}")
    
    def save_dataset(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save cleaned dataset to CSV file.
        
        Args:
            df (pd.DataFrame): Cleaned dataset
            output_path (str): Output file path
        """
        try:
            df.to_csv(output_path, index=False)
            print(f"\nCleaned dataset saved to: {output_path}")
        except Exception as e:
            print(f"Error saving dataset: {str(e)}")
    
    def clean_dataset(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Main method to clean the entire dataset.
        
        Args:
            input_path (str): Path to input CSV file
            output_path (Optional[str]): Path to save cleaned CSV file
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        # Load dataset
        df = self.load_dataset(input_path)
        
        # Validate dataset
        self.validate_dataset(df)
        
        # Step 1: Filter by language
        df_lang_filtered = self.filter_by_language(df)
        
        # Step 2: Clean hypothesis
        df_cleaned = self.clean_hypothesis(df_lang_filtered)
        
        # Reset index
        df_cleaned = df_cleaned.reset_index(drop=True)
        
        # Generate statistics
        self.generate_cleaning_stats(df, df_cleaned)
        
        # Print summary
        self.print_summary()
        
        # Save if output path provided
        if output_path:
            self.save_dataset(df_cleaned, output_path)
        
        return df_cleaned
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """
        Get cleaning statistics.
        
        Returns:
            Dict[str, Any]: Cleaning statistics dictionary
        """
        return self.cleaning_stats
    
    def reset_stats(self) -> None:
        """Reset all cleaning statistics."""
        self.original_size = 0
        self.cleaned_size = 0
        self.removed_by_language = 0
        self.removed_by_hypothesis = 0
        self.cleaning_stats = {}