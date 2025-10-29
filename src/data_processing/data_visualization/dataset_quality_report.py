import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

class DataQualityReporter:
    """
    A comprehensive data quality reporter with advanced analysis capabilities.
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with customizable configuration.
        
        Args:
            df (pd.DataFrame): DataFrame containing the dataset
            config (Optional[Dict[str, Any]]): Configuration dictionary
        """
        self.df = df.copy()
        self.config = config or {
            'expected_labels': [0, 1, 2],
            'text_columns': ['premise', 'hypothesis'],
            'required_columns': ['ID', 'label'],
            'severity_thresholds': {
                'missing_values': 0.05,
                'duplicate_rows': 0.01,
                'empty_text': 0.02,
                'outlier_text_length': 0.05
            },
            'text_length_thresholds': {
                'min_length': 5,
                'max_length': 1000
            }
        }
        self.report_timestamp = datetime.now()
        self._validate_dataframe()
    
    def _validate_dataframe(self) -> None:
        """Validate DataFrame structure."""
        missing_columns = [col for col in self.config['required_columns'] if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame must contain columns: {missing_columns}")
    
    def check_text_length_outliers(self) -> Dict[str, Any]:
        """Check for text length outliers."""
        outliers = {}
        total_outliers = 0
        
        for col in self.config['text_columns']:
            if col in self.df.columns:
                text_lengths = self.df[col].astype(str).str.len()
                
                too_short = (text_lengths < self.config['text_length_thresholds']['min_length']).sum()
                too_long = (text_lengths > self.config['text_length_thresholds']['max_length']).sum()
                
                outliers[col] = {
                    'too_short': too_short,
                    'too_long': too_long,
                    'mean_length': text_lengths.mean(),
                    'median_length': text_lengths.median(),
                    'std_length': text_lengths.std()
                }
                
                total_outliers += too_short + too_long
        
        outlier_percentage = (total_outliers / len(self.df)) * 100
        
        return {
            'outliers_by_column': outliers,
            'total_outliers': total_outliers,
            'outlier_percentage': outlier_percentage,
            'has_outliers': total_outliers > 0,
            'severity': 'high' if outlier_percentage > self.config['severity_thresholds']['outlier_text_length'] * 100 else 'low'
        }
    
    def generate_quality_score(self) -> Dict[str, Any]:
        """Generate an overall quality score."""
        analysis = self.get_comprehensive_analysis()
        
        # Scoring weights
        weights = {
            'missing_values': 0.25,
            'duplicate_rows': 0.20,
            'duplicate_ids': 0.25,
            'label_validity': 0.20,
            'empty_text': 0.10
        }
        
        scores = {}
        
        # Calculate individual scores (0-100)
        scores['missing_values'] = max(0, 100 - analysis['missing_values']['missing_percentage'] * 10)
        scores['duplicate_rows'] = max(0, 100 - analysis['duplicate_rows']['duplicate_percentage'] * 50)
        scores['duplicate_ids'] = 0 if analysis['duplicate_ids']['has_duplicate_ids'] else 100
        scores['label_validity'] = 0 if analysis['label_validity']['has_invalid_labels'] else 100
        scores['empty_text'] = max(0, 100 - analysis['empty_text']['empty_percentage'] * 20)
        
        # Calculate weighted overall score
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        
        # Determine grade
        if overall_score >= 90:
            grade = 'A'
        elif overall_score >= 80:
            grade = 'B'
        elif overall_score >= 70:
            grade = 'C'
        elif overall_score >= 60:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'overall_score': overall_score,
            'grade': grade,
            'individual_scores': scores,
            'weights': weights
        }
    
    def generate_executive_summary(self) -> str:
        """Generate an executive summary of data quality."""
        analysis = self.get_comprehensive_analysis()
        quality_score = self.generate_quality_score()
        
        summary_lines = [
            "EXECUTIVE SUMMARY",
            "=" * 50,
            f"Dataset Quality Score: {quality_score['overall_score']:.1f}/100 (Grade: {quality_score['grade']})",
            f"Total Records: {analysis['dataset_info']['total_rows']:,}",
            f"Assessment Date: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        # Key findings
        issues = self.generate_issue_list()
        if issues:
            summary_lines.extend([
                "",
                f"Key Issues ({len(issues)}):",
                *[f"  • {issue}" for issue in issues[:5]]  # Top 5 issues
            ])
        else:
            summary_lines.append("\n✅ No critical data quality issues detected")
        
        # Readiness assessment
        quality_assessment = self.assess_overall_quality()
        readiness_status = "READY" if quality_assessment['is_ready_for_training'] else "NOT READY"
        summary_lines.append(f"\nTraining Readiness: {readiness_status}")
        
        return "\n".join(summary_lines)
    
    def __call__(self) -> None:
        """Print comprehensive quality report when called."""
        print(self.generate_executive_summary())
        print("\n" + "=" * 60)
        print(self._format_quality_report())