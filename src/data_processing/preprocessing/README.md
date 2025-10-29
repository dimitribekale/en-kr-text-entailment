# Preprocessing Module

Preprocessing pipeline for multilingual textual entailment data.

## Design Principles

- **No lowercasing** - preserves casing for multilingual models (XLM-RoBERTa)
- **No contraction expansion** - modern tokenizers handle contractions natively

## Module Structure

```
preprocessing/
├── text_cleaner.py          # Basic text cleaning utilities
├── data_validator.py        # Data quality validation
├── data_cleaner.py          # Dataset cleaning operations
├── english_processor.py     # English text processing
├── korean_processor.py      # Korean text processing
├── pipeline.py              # Main pipeline orchestrator
└── README.md               # This file
```

## Module Descriptions

### `text_cleaner.py`
Basic text cleaning operations used by all processors:
- Unicode normalization
- Whitespace normalization
- Punctuation cleanup
- Special character removal
- Validation utilities

### `data_validator.py`
Data quality checks and validation:
- Column validation
- Language code validation
- Hypothesis quality checks
- Statistical summaries

### `data_cleaner.py`
Dataset-level cleaning operations:
- Remove invalid languages
- Remove problematic hypotheses
- Remove duplicates
- Generate cleaning statistics

### `english_processor.py`
English text processing (optimized for XLM-RoBERTa):
- **No lowercasing** - preserves case information
- **No contraction expansion** - model handles it
- Special character cleanup
- Punctuation normalization

### `korean_processor.py`
Korean text processing:
- Unicode NFC normalization (critical for Korean)
- Slang normalization
- Hangul character preservation
- Korean-specific cleaning

### `pipeline.py`
Main orchestrator that chains all operations:
1. Load data
2. Clean data
3. Process English text
4. Process Korean text
5. Save output

## Usage

```python
from preprocessing.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()
df = pipeline.run(
    input_path="input.csv",
    output_path="output.csv",
    show_samples=True
)
```

Or use the convenience script:
```bash
python run_preprocessing.py
```


## Testing Individual Processors

```python
# Test English processor
from preprocessing.english_processor import EnglishProcessor
processor = EnglishProcessor()
text = processor("This is a TEST!!! Can't wait.")
print(text)  # "This is a TEST!! Can't wait." (case preserved!)

# Test Korean processor
from preprocessing.korean_processor import KoreanProcessor
processor = KoreanProcessor()
text = processor("안녕하세요ㅋㅋㅋ")
print(text)  # "안녕하세요ㅋㅋ"
```
