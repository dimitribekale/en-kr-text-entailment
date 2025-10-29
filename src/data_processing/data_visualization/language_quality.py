import re
import pandas as pd
from typing import List, Dict, Any

class LanguageQualityChecker:
    """
    A clean and efficient class for checking language-specific data quality.
    """
    
    def __init__(self, df: pd.DataFrame, text_columns: List[str] = None, target_languages: List[str] = None):
        """
        Initialize LanguageQualityChecker with a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing the dataset
            text_columns (List[str]): List of text columns to analyze (default: ['premise', 'hypothesis'])
            target_languages (List[str]): List of target languages (default: ['en', 'ko'])
        """
        self.df = df.copy()  # Work with a copy to avoid modifying original
        self.text_columns = text_columns or ['premise', 'hypothesis']
        self.target_languages = target_languages or ['en', 'ko']
        self._validate_dataframe()
    
    def _validate_dataframe(self) -> None:
        """Validate that the DataFrame has required columns."""
        if 'language' not in self.df.columns:
            raise ValueError("DataFrame must contain a 'language' column")
        
        missing_text_cols = [col for col in self.text_columns if col not in self.df.columns]
        if missing_text_cols:
            raise ValueError(f"DataFrame must contain text columns: {missing_text_cols}")
    
    def _contains_korean(self, text: Any) -> bool:
        """Check if text contains Korean characters."""
        if pd.isna(text):
            return False
        return bool(re.search(r'[\uAC00-\uD7A3]', str(text)))
    
    def _contains_english(self, text: Any) -> bool:
        """Check if text contains English characters."""
        if pd.isna(text):
            return False
        return bool(re.search(r'[a-zA-Z]', str(text)))
    
    def _has_special_chars(self, text: Any) -> bool:
        """Check if text contains unusual Unicode characters."""
        if pd.isna(text):
            return False
        # Check for unusual Unicode characters (excluding Korean, English, numbers, and common punctuation)
        return bool(re.search(r'[^\w\s\uAC00-\uD7A3.,!?()\'\":-]', str(text)))
    
    def get_language_distribution(self) -> Dict[str, Any]:
        """Get language distribution statistics."""
        lang_counts = self.df['language'].value_counts()
        total_samples = len(self.df)
        
        distribution = {}
        for lang in self.target_languages:
            count = lang_counts.get(lang, 0)
            distribution[lang] = {
                'count': count,
                'percentage': (count / total_samples * 100) if total_samples > 0 else 0
            }
        
        # Add other languages
        other_langs = lang_counts[~lang_counts.index.isin(self.target_languages)]
        if len(other_langs) > 0:
            distribution['other'] = {
                'languages': dict(other_langs),
                'total_count': other_langs.sum(),
                'percentage': (other_langs.sum() / total_samples * 100) if total_samples > 0 else 0
            }
        
        return distribution
    
    def analyze_language_mixing(self) -> Dict[str, Any]:
        """Analyze potential language misclassification."""
        analysis = {}
        
        for lang in self.target_languages:
            lang_df = self.df[self.df['language'] == lang]
            
            if lang == 'ko':
                # Korean sentences that might contain English
                mixed_rows = lang_df[
                    lang_df[self.text_columns].apply(
                        lambda row: any(self._contains_english(row[col]) for col in self.text_columns), 
                        axis=1
                    )
                ]
                analysis['korean_with_english'] = {
                    'count': len(mixed_rows),
                    'percentage': (len(mixed_rows) / len(lang_df) * 100) if len(lang_df) > 0 else 0,
                    'samples': mixed_rows[['ID'] + self.text_columns].head(5) if len(mixed_rows) > 0 else None
                }
            
            elif lang == 'en':
                # English sentences that might contain Korean
                mixed_rows = lang_df[
                    lang_df[self.text_columns].apply(
                        lambda row: any(self._contains_korean(row[col]) for col in self.text_columns), 
                        axis=1
                    )
                ]
                analysis['english_with_korean'] = {
                    'count': len(mixed_rows),
                    'percentage': (len(mixed_rows) / len(lang_df) * 100) if len(lang_df) > 0 else 0,
                    'samples': mixed_rows[['ID'] + self.text_columns].head(5) if len(mixed_rows) > 0 else None
                }
        
        return analysis
    
    def analyze_special_characters(self) -> Dict[str, Any]:
        """Analyze special characters and encoding issues."""
        # Find rows with special characters
        special_char_mask = self.df[self.text_columns].apply(
            lambda row: any(self._has_special_chars(row[col]) for col in self.text_columns), 
            axis=1
        )
        
        special_char_rows = self.df[special_char_mask]
        
        analysis = {
            'total_count': len(special_char_rows),
            'percentage': (len(special_char_rows) / len(self.df) * 100) if len(self.df) > 0 else 0,
            'by_language': {},
            'samples': special_char_rows[['ID', 'language'] + self.text_columns].head(10) if len(special_char_rows) > 0 else None
        }
        
        # Analyze by language
        for lang in self.target_languages:
            lang_special = special_char_rows[special_char_rows['language'] == lang]
            lang_total = len(self.df[self.df['language'] == lang])
            
            analysis['by_language'][lang] = {
                'count': len(lang_special),
                'percentage': (len(lang_special) / lang_total * 100) if lang_total > 0 else 0
            }
        
        return analysis
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive language quality analysis."""
        return {
            'language_distribution': self.get_language_distribution(),
            'language_mixing': self.analyze_language_mixing(),
            'special_characters': self.analyze_special_characters(),
            'total_samples': len(self.df)
        }
    
    def _format_language_report(self) -> str:
        """Format comprehensive language quality report as a string."""
        lines = [
            "=" * 50,
            "LANGUAGE-SPECIFIC QUALITY ANALYSIS",
            "=" * 50
        ]
        
        analysis = self.get_comprehensive_analysis()
        
        # Language distribution
        distribution = analysis['language_distribution']
        lines.append("\nLanguage Distribution:")
        
        for lang, stats in distribution.items():
            if lang != 'other':
                lines.append(f"  {lang.upper()}: {stats['count']:,} ({stats['percentage']:.1f}%)")
        
        if 'other' in distribution:
            other_stats = distribution['other']
            lines.append(f"  Other languages: {other_stats['total_count']:,} ({other_stats['percentage']:.1f}%)")
            if other_stats['languages']:
                lines.append(f"    Details: {dict(other_stats['languages'])}")
        
        # Language mixing analysis
        mixing = analysis['language_mixing']
        lines.append("\nPotential Language Misclassification:")
        
        if 'korean_with_english' in mixing:
            ko_en = mixing['korean_with_english']
            lines.append(f"  Korean sentences with English characters: {ko_en['count']} ({ko_en['percentage']:.1f}%)")
            
            if ko_en['samples'] is not None and len(ko_en['samples']) > 0:
                lines.append("    Sample Korean sentences with English:")
                lines.append(ko_en['samples'].to_string(index=False))
        
        if 'english_with_korean' in mixing:
            en_ko = mixing['english_with_korean']
            lines.append(f"  English sentences with Korean characters: {en_ko['count']} ({en_ko['percentage']:.1f}%)")
            
            if en_ko['samples'] is not None and len(en_ko['samples']) > 0:
                lines.append("    Sample English sentences with Korean:")
                lines.append(en_ko['samples'].to_string(index=False))
        
        # Special characters analysis
        special_chars = analysis['special_characters']
        lines.append(f"\nSpecial Characters Analysis:")
        lines.append(f"  Total sentences with special characters: {special_chars['total_count']} ({special_chars['percentage']:.1f}%)")
        
        lines.append("  By language:")
        for lang, stats in special_chars['by_language'].items():
            lines.append(f"    {lang.upper()}: {stats['count']} ({stats['percentage']:.1f}%)")
        
        if special_chars['samples'] is not None and len(special_chars['samples']) > 0:
            lines.append("  Sample sentences with special characters:")
            lines.append(special_chars['samples'].to_string(index=False))
        
        return "\n".join(lines)
    
    def check_language_quality(self) -> Dict[str, Any]:
        """
        Check language-specific data quality and return comprehensive analysis.
        
        Returns:
            Dict[str, Any]: Dictionary containing all language quality statistics
        """
        return self.get_comprehensive_analysis()
    
    def print_report(self) -> None:
        """Print the language quality analysis report."""
        print(self._format_language_report())
    
    def get_report(self) -> str:
        """
        Get the language quality analysis report as a string.
        
        Returns:
            str: Formatted language quality analysis report
        """
        return self._format_language_report()
    
    def save_report(self, filepath: str) -> None:
        """
        Save the language quality analysis report to a file.
        
        Args:
            filepath (str): Path to save the report
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self._format_language_report())
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get a summary of language quality issues with severity assessment."""
        analysis = self.get_comprehensive_analysis()
        
        summary = {
            'severity': 'good',
            'issues': [],
            'recommendations': []
        }
        
        # Check for language mixing issues
        mixing = analysis['language_mixing']
        
        if 'korean_with_english' in mixing and mixing['korean_with_english']['count'] > 0:
            ko_en_pct = mixing['korean_with_english']['percentage']
            if ko_en_pct > 10:
                summary['severity'] = 'moderate'
                summary['issues'].append(f"High Korean-English mixing: {ko_en_pct:.1f}%")
                summary['recommendations'].append("Review Korean sentences with English content")
        
        if 'english_with_korean' in mixing and mixing['english_with_korean']['count'] > 0:
            en_ko_pct = mixing['english_with_korean']['percentage']
            if en_ko_pct > 5:
                summary['severity'] = 'high' if summary['severity'] != 'critical' else 'critical'
                summary['issues'].append(f"English-Korean mixing detected: {en_ko_pct:.1f}%")
                summary['recommendations'].append("Review and correct language classification")
        
        # Check for special character issues
        special_chars = analysis['special_characters']
        if special_chars['percentage'] > 5:
            summary['issues'].append(f"High special character usage: {special_chars['percentage']:.1f}%")
            summary['recommendations'].append("Clean special characters before tokenization")
        
        return summary
    
    def __call__(self) -> None:
        """Print language quality analysis when called."""
        self.print_report()
    
    def __str__(self) -> str:
        """String representation of the language quality analysis."""
        return self._format_language_report()
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"LanguageQualityChecker(samples={len(self.df)}, languages={self.target_languages})"
