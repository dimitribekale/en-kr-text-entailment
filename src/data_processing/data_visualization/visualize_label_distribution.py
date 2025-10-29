import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any

class LabelDistributionVisualizer:
    """
    A comprehensive label distribution analyzer with visualization capabilities.
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
            'label_mapping': {0: 'Entailment', 1: 'Neutral', 2: 'Contradiction'},
            'imbalance_threshold': 3.0,
            'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'figure_size': (12, 8)
        }
        self._validate_dataframe()
    
    def _validate_dataframe(self) -> None:
        """Validate DataFrame structure."""
        if 'label' not in self.df.columns:
            raise ValueError("DataFrame must contain a 'label' column")
    
    def create_distribution_data(self) -> Dict[str, Any]:
        """Create data structure suitable for visualization."""
        # Overall distribution
        overall_dist = self.df['label'].value_counts().sort_index()
        
        # Prepare data for visualization
        viz_data = {
            'labels': [self.config['label_mapping'].get(label, f'Label {label}') for label in overall_dist.index],
            'counts': overall_dist.values.tolist(),
            'percentages': (overall_dist / len(self.df) * 100).values.tolist()
        }
        
        # Language-specific data if available
        if 'language' in self.df.columns:
            lang_data = {}
            for lang in self.df['language'].unique():
                lang_subset = self.df[self.df['language'] == lang]
                lang_dist = lang_subset['label'].value_counts().sort_index()
                
                # Ensure all labels are represented
                all_labels = sorted(self.df['label'].unique())
                lang_counts = [lang_dist.get(label, 0) for label in all_labels]
                lang_percentages = [(count / len(lang_subset) * 100) if len(lang_subset) > 0 else 0 
                                  for count in lang_counts]
                
                lang_data[lang] = {
                    'counts': lang_counts,
                    'percentages': lang_percentages,
                    'total_samples': len(lang_subset)
                }
            
            viz_data['by_language'] = lang_data
        
        return viz_data
    
    def generate_chart_data(self) -> Dict[str, Any]:
        """Generate data formatted for chart creation."""
        viz_data = self.create_distribution_data()
        
        # Overall distribution chart data
        overall_chart_data = {
            'labels': viz_data['labels'],
            'values': viz_data['counts'],
            'type': 'bar'
        }
        
        chart_data = {
            'overall_distribution': overall_chart_data,
            'title': 'Label Distribution Analysis',
            'subtitle': f'Total samples: {len(self.df):,}'
        }
        
        # Language comparison data if available
        if 'by_language' in viz_data:
            lang_comparison_data = []
            for lang, data in viz_data['by_language'].items():
                lang_comparison_data.append({
                    'name': lang.upper(),
                    'values': data['percentages']
                })
            
            chart_data['language_comparison'] = {
                'categories': viz_data['labels'],
                'series': lang_comparison_data,
                'type': 'grouped_bar'
            }
        
        return chart_data
    
    def get_statistical_summary(self) -> Dict[str, Any]:
        """Get statistical summary of label distribution."""
        overall_dist = self.df['label'].value_counts(normalize=True) * 100
        
        summary = {
            'entropy': -sum(p/100 * np.log2(p/100) for p in overall_dist if p > 0),
            'gini_coefficient': 1 - sum((p/100)**2 for p in overall_dist),
            'most_frequent_label': overall_dist.idxmax(),
            'least_frequent_label': overall_dist.idxmin(),
            'balance_ratio': overall_dist.max() / overall_dist.min() if overall_dist.min() > 0 else float('inf')
        }
        
        # Add label names
        summary['most_frequent_label_name'] = self.config['label_mapping'].get(
            summary['most_frequent_label'], f"Label {summary['most_frequent_label']}"
        )
        summary['least_frequent_label_name'] = self.config['label_mapping'].get(
            summary['least_frequent_label'], f"Label {summary['least_frequent_label']}"
        )
        
        return summary
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive report with statistical insights."""
        lines = [
            "=" * 60,
            "COMPREHENSIVE LABEL DISTRIBUTION ANALYSIS",
            "=" * 60
        ]
        
        # Basic statistics
        overall_dist = self.df['label'].value_counts().sort_index()
        lines.extend([
            "",
            "BASIC STATISTICS:",
            f"  Total samples: {len(self.df):,}",
            f"  Unique labels: {len(overall_dist)}",
            f"  Labels: {list(overall_dist.index)}"
        ])
        
        # Distribution details
        lines.append("\nLABEL DISTRIBUTION:")
        for label, count in overall_dist.items():
            percentage = (count / len(self.df)) * 100
            label_name = self.config['label_mapping'].get(label, f'Label {label}')
            lines.append(f"  {label} ({label_name}): {count:,} ({percentage:.1f}%)")
        
        # Statistical measures
        stats = self.get_statistical_summary()
        lines.extend([
            "",
            "STATISTICAL MEASURES:",
            f"  Entropy: {stats['entropy']:.3f}",
            f"  Gini coefficient: {stats['gini_coefficient']:.3f}",
            f"  Balance ratio: {stats['balance_ratio']:.2f}",
            f"  Most frequent: {stats['most_frequent_label_name']}",
            f"  Least frequent: {stats['least_frequent_label_name']}"
        ])
        
        # Language-specific analysis
        if 'language' in self.df.columns:
            lines.append("\nLANGUAGE-SPECIFIC ANALYSIS:")
            
            for lang in self.df['language'].unique():
                lang_subset = self.df[self.df['language'] == lang]
                lang_dist = lang_subset['label'].value_counts(normalize=True) * 100
                
                lines.append(f"  {lang.upper()} ({len(lang_subset):,} samples):")
                for label, percentage in lang_dist.items():
                    label_name = self.config['label_mapping'].get(label, f'Label {label}')
                    lines.append(f"    {label} ({label_name}): {percentage:.1f}%")
                
                # Check for imbalance
                min_pct = lang_dist.min()
                max_pct = lang_dist.max()
                if max_pct / min_pct > self.config['imbalance_threshold']:
                    lines.append(f"    ⚠️  Imbalance detected: {max_pct:.1f}% vs {min_pct:.1f}%")
        
        # Recommendations
        recommendations = self.get_recommendations()
        if recommendations:
            lines.append("\nRECOMMENDATIONS:")
            for rec in recommendations:
                lines.append(f"  • {rec}")
        
        return "\n".join(lines)
    
    def get_recommendations(self) -> List[str]:
        """Get actionable recommendations based on analysis."""
        recommendations = []
        stats = self.get_statistical_summary()
        
        # Balance recommendations
        if stats['balance_ratio'] > 10:
            recommendations.extend([
                "Severe class imbalance detected - consider data augmentation",
                "Use stratified sampling for train/validation splits",
                "Apply class weights during model training"
            ])
        elif stats['balance_ratio'] > 3:
            recommendations.extend([
                "Moderate class imbalance - monitor minority class performance",
                "Consider using class weights during training"
            ])
        
        # Entropy recommendations
        if stats['entropy'] < 1.0:
            recommendations.append("Low entropy indicates potential data collection bias")
        elif stats['entropy'] > 1.5:
            recommendations.append("High entropy suggests good class diversity")
        
        return recommendations
    
    def __call__(self) -> None:
        """Print comprehensive analysis when called."""
        print(self.generate_comprehensive_report())
    
    def __str__(self) -> str:
        """String representation."""
        return self.generate_comprehensive_report()