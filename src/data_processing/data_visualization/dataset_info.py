import io
import pandas as pd
from typing import Dict, Any

class DatasetInfoReporter:
    """A comprehensive dataset information reporter with flexible output options."""
    
    def __init__(self, df: pd.DataFrame, show_head: bool = True, head_rows: int = 5):
        """
        Initialize DatasetInfoReporter with customizable options.
        
        Args:
            df (pd.DataFrame): DataFrame containing the dataset
            show_head (bool): Whether to include head rows in output
            head_rows (int): Number of head rows to display
        """
        self.df = df
        self.show_head = show_head
        self.head_rows = head_rows
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic dataset statistics as a dictionary."""
        return {
            'shape': self.df.shape,
            'total_rows': self.df.shape[0],
            'total_columns': self.df.shape[1],
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'column_info': self._get_column_info(),
            'head_data': self.df.head(self.head_rows) if self.show_head else None
        }
    
    def _get_column_info(self) -> str:
        """Get column information as string."""
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()
    
    def format_report(self) -> str:
        """Format complete dataset report."""
        stats = self.get_basic_stats()
        
        report_lines = [
            "=" * 50,
            "BASIC DATASET INFORMATION",
            "=" * 50,
            f"Dataset shape: {stats['shape']}",
            f"Total rows: {stats['total_rows']:,}",
            f"Total columns: {stats['total_columns']}",
            f"Memory usage: {stats['memory_usage_mb']:.2f} MB",
            "",
            "Column Information:",
            stats['column_info']
        ]
        
        if self.show_head and stats['head_data'] is not None:
            report_lines.extend([
                "",
                f"First {self.head_rows} rows:",
                str(stats['head_data'])
            ])
        
        return "\n".join(report_lines)
    
    def print_report(self) -> None:
        """Print the formatted report."""
        print(self.format_report())
    
    def save_report(self, filepath: str) -> None:
        """Save report to a text file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.format_report())
    
    def __call__(self) -> None:
        """Print report when called."""
        self.print_report()
    
    def __str__(self) -> str:
        """String representation."""
        return self.format_report()