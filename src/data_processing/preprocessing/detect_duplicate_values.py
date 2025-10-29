import pandas as pd
from typing import Optional, Dict, Any

class DuplicateValuesDetector:
    """
    A clean and efficient class for detecting duplicate values in a DataFrame.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DuplicateValuesDetector with a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing the dataset
        """
        self.df = df
        self._validate_dataframe()
    
    def _validate_dataframe(self) -> None:
        """Validate that the DataFrame has required columns."""
        required_columns = ['ID']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"DataFrame must contain columns: {missing_columns}")
    
    def get_duplicate_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive duplicate statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing all duplicate statistics
        """
        stats = {
            'total_duplicates': self.df.duplicated().sum(),
            'duplicate_ids': self.df['ID'].duplicated().sum(),
            'premise_hypothesis_duplicates': 0,
            'premise_duplicates': 0,
            'hypothesis_duplicates': 0
        }
        
        # Only check text columns if they exist
        if 'premise' in self.df.columns and 'hypothesis' in self.df.columns:
            stats['premise_hypothesis_duplicates'] = self.df.duplicated(subset=['premise', 'hypothesis']).sum()
            stats['premise_duplicates'] = self.df.duplicated(subset=['premise']).sum()
            stats['hypothesis_duplicates'] = self.df['hypothesis'].duplicated().sum()
        
        return stats
    
    def _format_duplicate_report(self) -> str:
        """Format duplicate analysis report as a string."""
        lines = [
            "=" * 50,
            "DUPLICATE VALUES ANALYSIS",
            "=" * 50
        ]
        
        stats = self.get_duplicate_stats()
        
        # Total duplicate rows
        lines.append(f"Total duplicate rows: {stats['total_duplicates']}")
        
        if stats['total_duplicates'] > 0:
            lines.append("\nSample duplicate rows:")
            duplicate_rows = self.df[self.df.duplicated(keep=False)].sort_values('ID')
            lines.append(duplicate_rows.head(10).to_string())
        
        # Duplicate IDs
        lines.append(f"\nDuplicate IDs: {stats['duplicate_ids']}")
        
        if stats['duplicate_ids'] > 0:
            lines.append("Rows with duplicate IDs:")
            dup_id_rows = self.df[self.df['ID'].duplicated(keep=False)].sort_values('ID')
            if 'premise' in self.df.columns and 'hypothesis' in self.df.columns:
                lines.append(dup_id_rows[['ID', 'premise', 'hypothesis', 'label']].head(10).to_string())
            else:
                lines.append(dup_id_rows.head(10).to_string())
        
        # Text-based duplicates (only if columns exist)
        if 'premise' in self.df.columns and 'hypothesis' in self.df.columns:
            lines.append(f"\nDuplicate premise-hypothesis pairs: {stats['premise_hypothesis_duplicates']}")
            
            if stats['premise_hypothesis_duplicates'] > 0:
                lines.append("Sample duplicate premise-hypothesis pairs:")
                dup_pairs = self.df[self.df.duplicated(subset=['premise', 'hypothesis'], keep=False)]
                lines.append(dup_pairs[['ID', 'premise', 'hypothesis', 'label']].head(10).to_string())
            
            lines.append(f"\nDuplicate premises (same premise, different hypothesis): {stats['premise_duplicates']}")
            lines.append(f"Duplicate premise texts: {self.df['premise'].duplicated().sum()}")
            lines.append(f"Duplicate hypothesis texts: {stats['hypothesis_duplicates']}")
        
        return "\n".join(lines)
    
    def check_duplicates(self) -> Dict[str, Any]:
        """
        Perform comprehensive duplicate analysis and return statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing duplicate statistics
        """
        return self.get_duplicate_stats()
    
    def print_report(self) -> None:
        """Print the duplicate analysis report."""
        print(self._format_duplicate_report())
    
    def get_report(self) -> str:
        """
        Get the duplicate analysis report as a string.
        
        Returns:
            str: Formatted duplicate analysis report
        """
        return self._format_duplicate_report()
    
    def save_report(self, filepath: str) -> None:
        """
        Save the duplicate analysis report to a file.
        
        Args:
            filepath (str): Path to save the report
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self._format_duplicate_report())
    
    def remove_duplicates(self, subset: Optional[list] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.
        
        Args:
            subset (Optional[list]): Column names to consider for duplicates
            keep (str): Which duplicates to keep ('first', 'last', False)
            
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        return self.df.drop_duplicates(subset=subset, keep=keep)
    
    def __call__(self) -> None:
        """Print duplicate analysis when called."""
        self.print_report()
    
    def __str__(self) -> str:
        """String representation of the duplicate analysis."""
        return self._format_duplicate_report()
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        stats = self.get_duplicate_stats()
        return f"DuplicateValuesDetector(total_duplicates={stats['total_duplicates']})"