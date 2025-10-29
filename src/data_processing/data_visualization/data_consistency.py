import pandas as pd
from typing import List, Optional, Dict, Any


class DatasetQualityChecker:
    """
    A clean and efficient class for checking dataset quality and consistency.
    This module provides a comprehensive DatasetQualityChecker class for analyzing dataset quality,
    including label distribution, language consistency, and text length analysis.

    """
    
    def __init__(self, df: pd.DataFrame, expected_labels: List[int] = None, expected_languages: List[str] = None):
        """
        Initialize DatasetQualityChecker with a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing the dataset
            expected_labels (List[int]): Expected label values (default: [0, 1, 2])
            expected_languages (List[str]): Expected language codes (default: ['en', 'ko'])
        """
        self.df = df.copy()  # Work with a copy to avoid modifying original
        self.expected_labels = expected_labels or [0, 1, 2]
        self.expected_languages = expected_languages or ['en', 'ko']
        self._validate_dataframe()
    
    def _validate_dataframe(self) -> None:
        """Validate that the DataFrame has required columns."""
        required_columns = ['label']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"DataFrame must contain columns: {missing_columns}")
    
    def get_label_analysis(self) -> Dict[str, Any]:
        """Analyze label distribution and validity."""
        label_counts = self.df['label'].value_counts().sort_index()
        label_percentages = (label_counts / len(self.df) * 100).round(2)
        invalid_labels = self.df[~self.df['label'].isin(self.expected_labels)]
        
        return {
            'distribution': label_counts,
            'percentages': label_percentages,
            'invalid_labels': invalid_labels,
            'invalid_count': len(invalid_labels)
        }
    
    def get_language_analysis(self) -> Optional[Dict[str, Any]]:
        """Analyze language distribution if language column exists."""
        if 'language' not in self.df.columns:
            return None
        
        lang_counts = self.df['language'].value_counts()
        lang_percentages = (lang_counts / len(self.df) * 100).round(2)
        unexpected_langs = self.df[~self.df['language'].isin(self.expected_languages)]
        
        return {
            'distribution': lang_counts,
            'percentages': lang_percentages,
            'unexpected_languages': unexpected_langs,
            'unexpected_count': len(unexpected_langs)
        }
    
    def get_text_length_analysis(self, short_threshold: int = 5, long_threshold: int = 500) -> Dict[str, Any]:
        """Analyze text lengths for premise and hypothesis."""
        # Work with a temporary copy to avoid modifying self.df
        temp_df = self.df.copy()
        temp_df['premise_length'] = temp_df['premise'].astype(str).str.len()
        temp_df['hypothesis_length'] = temp_df['hypothesis'].astype(str).str.len()
        
        analysis = {
            'premise_stats': {
                'min': temp_df['premise_length'].min(),
                'max': temp_df['premise_length'].max(),
                'mean': temp_df['premise_length'].mean()
            },
            'hypothesis_stats': {
                'min': temp_df['hypothesis_length'].min(),
                'max': temp_df['hypothesis_length'].max(),
                'mean': temp_df['hypothesis_length'].mean()
            },
            'short_premises': temp_df[temp_df['premise_length'] < short_threshold],
            'short_hypotheses': temp_df[temp_df['hypothesis_length'] < short_threshold],
            'long_premises': temp_df[temp_df['premise_length'] > long_threshold],
            'long_hypotheses': temp_df[temp_df['hypothesis_length'] > long_threshold]
        }
        
        return analysis
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive quality statistics."""
        stats = {
            'total_rows': len(self.df),
            'label_analysis': self.get_label_analysis(),
            'language_analysis': self.get_language_analysis(),
            'text_analysis': self.get_text_length_analysis()
        }
        
        return stats
    
    def _format_quality_report(self) -> str:
        """Format comprehensive quality report as a string."""
        lines = [
            "=" * 50,
            "DATA CONSISTENCY ANALYSIS",
            "=" * 50
        ]
        
        stats = self.get_comprehensive_stats()
        
        # Label analysis
        label_analysis = stats['label_analysis']
        lines.extend([
            "",
            "Label Distribution:",
            str(label_analysis['distribution']),
            "Label percentages:",
            str(label_analysis['percentages'])
        ])
        
        if label_analysis['invalid_count'] > 0:
            lines.extend([
                f"\nInvalid labels (not in {self.expected_labels}): {label_analysis['invalid_count']}",
                "Rows with invalid labels:",
                str(label_analysis['invalid_labels'][['ID', 'label']].head())
            ])
        else:
            lines.append(f"\nInvalid labels: {label_analysis['invalid_count']}")
        
        # Language analysis
        lang_analysis = stats['language_analysis']
        if lang_analysis:
            lines.extend([
                "",
                "Language Distribution:",
                str(lang_analysis['distribution']),
                "Language percentages:",
                str(lang_analysis['percentages'])
            ])
            
            if lang_analysis['unexpected_count'] > 0:
                lines.extend([
                    f"\nUnexpected languages: {lang_analysis['unexpected_count']}",
                    "Rows with unexpected languages:",
                    str(lang_analysis['unexpected_languages'][['ID', 'language', 'premise']].head())
                ])
            else:
                lines.append(f"\nUnexpected languages: {lang_analysis['unexpected_count']}")
        
        # Text length analysis
        text_analysis = stats['text_analysis']
        premise_stats = text_analysis['premise_stats']
        hypothesis_stats = text_analysis['hypothesis_stats']
        
        lines.extend([
            "",
            "Text Length Analysis:",
            f"Premise length - Min: {premise_stats['min']}, Max: {premise_stats['max']}, Mean: {premise_stats['mean']:.1f}",
            f"Hypothesis length - Min: {hypothesis_stats['min']}, Max: {hypothesis_stats['max']}, Mean: {hypothesis_stats['mean']:.1f}"
        ])
        
        # Short texts
        short_premises = text_analysis['short_premises']
        short_hypotheses = text_analysis['short_hypotheses']
        
        lines.append(f"\nVery short premises (< 5 chars): {len(short_premises)}")
        if len(short_premises) > 0:
            lines.extend([
                "Sample short premises:",
                str(short_premises[['ID', 'premise', 'premise_length']].head())
            ])
        
        lines.append(f"\nVery short hypotheses (< 5 chars): {len(short_hypotheses)}")
        if len(short_hypotheses) > 0:
            lines.extend([
                "Sample short hypotheses:",
                str(short_hypotheses[['ID', 'hypothesis', 'hypothesis_length']].head())
            ])
        
        # Long texts
        long_premises = text_analysis['long_premises']
        long_hypotheses = text_analysis['long_hypotheses']
        
        lines.extend([
            f"\nVery long premises (> 500 chars): {len(long_premises)}",
            f"Very long hypotheses (> 500 chars): {len(long_hypotheses)}"
        ])
        
        return "\n".join(lines)
    
    def check_data_consistency(self) -> Dict[str, Any]:
        """
        Check for data consistency issues and return comprehensive statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing all quality statistics
        """
        return self.get_comprehensive_stats()
    
    def print_report(self) -> None:
        """Print the quality analysis report."""
        print(self._format_quality_report())
    
    def get_report(self) -> str:
        """
        Get the quality analysis report as a string.
        
        Returns:
            str: Formatted quality analysis report
        """
        return self._format_quality_report()
    
    def save_report(self, filepath: str) -> None:
        """
        Save the quality analysis report to a file.
        
        Args:
            filepath (str): Path to save the report
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self._format_quality_report())
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get a summary of quality issues with severity assessment."""
        stats = self.get_comprehensive_stats()
        
        summary = {
            'severity': 'low',
            'issues': [],
            'recommendations': []
        }
        
        # Check for critical issues
        label_analysis = stats['label_analysis']
        if label_analysis['invalid_count'] > 0:
            summary['severity'] = 'critical'
            summary['issues'].append(f"Invalid labels found: {label_analysis['invalid_count']}")
            summary['recommendations'].append("Fix or remove rows with invalid labels")
        
        # Check language issues
        lang_analysis = stats['language_analysis']
        if lang_analysis and lang_analysis['unexpected_count'] > 0:
            summary['severity'] = 'high' if summary['severity'] != 'critical' else 'critical'
            summary['issues'].append(f"Unexpected languages found: {lang_analysis['unexpected_count']}")
            summary['recommendations'].append("Review and handle unexpected language entries")
        
        # Check text length issues
        text_analysis = stats['text_analysis']
        short_text_count = len(text_analysis['short_premises']) + len(text_analysis['short_hypotheses'])
        if short_text_count > 0:
            summary['issues'].append(f"Short texts found: {short_text_count}")
            summary['recommendations'].append("Review and clean very short text entries")
        
        return summary
    
    def __call__(self) -> None:
        """Print quality analysis when called."""
        self.print_report()
    
    def __str__(self) -> str:
        """String representation of the quality analysis."""
        return self._format_quality_report()
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"DatasetQualityChecker(rows={len(self.df)}, columns={len(self.df.columns)})"