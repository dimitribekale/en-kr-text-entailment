import pandas as pd
from preprocessing.korean_preprocessing import KoreanTextPreprocessor
from preprocessing.english_preprocessing import EnglishTextPreprocessor


df = pd.read_csv("datasets/dataset_with_language.csv")

# Initialize preprocessors
korean_preprocessor = KoreanTextPreprocessor()
english_preprocessor = EnglishTextPreprocessor(lowercase=True, remove_extra_whitespace=True)

# Create a copy for processing
df_processed = df.copy()

# Process Korean sentences
korean_mask = df_processed['language'] == 'ko'
if korean_mask.any():
    df_processed.loc[korean_mask, 'premise'] = korean_preprocessor(
        df_processed.loc[korean_mask, 'premise'].tolist()
    )
    df_processed.loc[korean_mask, 'hypothesis'] = korean_preprocessor(
        df_processed.loc[korean_mask, 'hypothesis'].tolist()
    )

# Process English sentences
english_mask = df_processed['language'] == 'en'
if english_mask.any():
    df_processed.loc[english_mask, 'premise'] = english_preprocessor(
        df_processed.loc[english_mask, 'premise'].tolist()
    )
    df_processed.loc[english_mask, 'hypothesis'] = english_preprocessor(
        df_processed.loc[english_mask, 'hypothesis'].tolist()
    )

# Save preprocessed dataset
df_processed.to_csv("datasets/preprocessed_dataset.csv", index=False)
