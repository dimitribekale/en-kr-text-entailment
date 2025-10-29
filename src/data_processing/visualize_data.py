import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from data_visualization.dataset_quality_report import DataQualityReporter
from data_visualization.data_consistency import DatasetQualityChecker
from data_visualization.dataset_info import DatasetInfoReporter
from data_visualization.language_quality import LanguageQualityChecker
from data_visualization.visualize_label_distribution import LabelDistributionVisualizer
from preprocessing.detect_duplicate_values import DuplicateValuesDetector
from preprocessing.clean_multilingual import MultilingualCleaner


pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Show full column width
pd.set_option('display.max_rows', None)  # Show all rows


"""
=============================================================================================

This file helps to visualize the data and have a deeper understanding of the data distribution.
The dataset is loaded using a csv file. If you are willing to use a different format (e.g: parquet)
Please make sure to replace the read_csv method to the target format method in all the imported modules. 

============================================================================================

Uncommment the commented lines as needed.

"""
# Load the dataset
df = pd.read_csv("your_dataset.csv")

reporter = DatasetInfoReporter(df, show_head=True, head_rows=10)
duplicate_detector = DuplicateValuesDetector(df)

# # Print to console
# reporter()  # or reporter.print_info()
# detector()  # or detector.print_report()

# # Save to file
# reporter.save_report("dataset_info.txt")


# # Get duplicate groups
# duplicate_groups = detector.identify_duplicate_groups()
# print(f"Found {len(duplicate_groups)} duplicate entries")

# # Remove duplicates
# cleaned_df = detector.remove_duplicates(subset=['premise', 'hypothesis'])

# # Custom text columns
# custom_detector = DuplicateValuesDetector(df, text_columns=['premise', 'hypothesis', 'custom_text'])
# custom_detector.print_report()

# # Save report to file
# detector.save_report("duplicate_analysis.txt")



# Example usage of the DatasetQualityChecker class
# checker = DatasetQualityChecker(df)

# # Print comprehensive report
# checker()  # or checker.print_report()

# # Get statistics programmatically
# stats = checker.get_comprehensive_stats()
# print(f"Invalid labels: {stats['label_analysis']['invalid_count']}")

# # Custom configuration
# custom_config = {
#     'expected_labels': [0, 1, 2],
#     'expected_languages': ['en', 'ko', 'ja'],
#     'short_text_threshold': 3,
#     'long_text_threshold': 1000
# }
# custom_checker = DatasetQualityChecker(df, config=custom_config)

# # Get quality summary
# summary = checker.get_quality_summary()
# if summary['severity'] == 'critical':
#     print("⚠️ Critical quality issues found!")

# # Save report to file
# checker.save_report("quality_analysis.txt")


# # Basic usage
# df = pd.read_csv("your_dataset.csv")
# checker = LanguageQualityChecker(df)

# # Print comprehensive report
# checker()  # or checker.print_report()

# # Get statistics programmatically
# analysis = checker.get_comprehensive_analysis()
# mixing = analysis['language_mixing']

# if 'english_with_korean' in mixing:
#     print(f"English-Korean mixing: {mixing['english_with_korean']['count']} cases")

# # Custom configuration
# custom_config = {
#     'text_columns': ['premise', 'hypothesis'],
#     'target_languages': ['en', 'ko', 'ja'],
#     'mixing_threshold': 3.0,
#     'special_char_threshold': 2.0
# }
# custom_checker = LanguageQualityChecker(df, config=custom_config)

# # Get quality summary
# summary = checker.get_quality_summary()
# if summary['severity'] in ['high', 'critical']:
#     print("⚠️ Language quality issues detected!")
#     for issue in summary['issues']:
#         print(f"  - {issue}")

# # Save detailed report
# checker.save_report("language_quality_analysis.txt")

# # Get character distribution analysis
# char_analysis = checker.analyze_character_distribution()
# for lang, stats in char_analysis.items():
#     print(f"{lang}: {stats['korean_chars']['percentage']:.1f}% Korean, {stats['english_chars']['percentage']:.1f}% English")


# # Basic usage
# df = pd.read_csv("your_dataset.csv")
# reporter = DataQualityReporter(df)

# # Print comprehensive report
# reporter()  # or reporter.print_report()

# # Get analysis programmatically
# report_data = reporter.generate_quality_report()
# analysis = report_data['analysis']
# quality_assessment = report_data['quality_assessment']

# print(f"Overall quality: {quality_assessment['overall_severity']}")
# print(f"Ready for training: {quality_assessment['is_ready_for_training']}")

# # Custom configuration
# custom_config = {
#     'expected_labels': [0, 1, 2],
#     'text_columns': ['premise', 'hypothesis'],
#     'required_columns': ['ID', 'label'],
#     'severity_thresholds': {
#         'missing_values': 0.02,  # Stricter threshold
#         'duplicate_rows': 0.005,
#         'empty_text': 0.01
#     }
# }
# custom_reporter = DataQualityReporter(df, config=custom_config)

# # Get quality score
# quality_score = custom_reporter.generate_quality_score()
# print(f"Quality score: {quality_score['overall_score']:.1f}/100 (Grade: {quality_score['grade']})")

# # Save reports
# reporter.save_report("quality_report.txt")
# reporter.export_issues_csv("quality_issues.csv")

# # Get executive summary
# summary = reporter.generate_executive_summary()
# print(summary)



# Example usage:
# Uncomment the following lines to use the MultilingualCleaner class

#  Basic usage
# cleaner = MultilingualCleaner()
# df_cleaned = cleaner.clean_dataset(
#     input_path="datasets/dataset_with_language.csv",
#     output_path="datasets/cleaned_dataset.csv"
# )

# # Custom configuration
# cleaner_custom = MultilingualCleaner(
#     valid_languages=['en', 'ko', 'ja'],  # Include Japanese
#     problematic_strings=['nan', 'na', 'hm', 'null', 'none']  # Custom problematic strings
# )
# df_cleaned_custom = cleaner_custom.clean_dataset(
#     input_path="datasets/dataset_with_language.csv",
#     output_path="datasets/cleaned_dataset_custom.csv"
# )

# # Access cleaning statistics
# stats = cleaner.get_cleaning_stats()
# print("Cleaning Statistics:")
# for key, value in stats.items():
#     print(f"  {key}: {value}")

# # Clean without saving
# cleaner_no_save = MultilingualCleaner()
# df_cleaned_no_save = cleaner_no_save.clean_dataset("datasets/dataset_with_language.csv")
