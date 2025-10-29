# Data Processing Pipeline

**Purpose:** Generate the `text_entailment.csv` dataset from the original multilingual data.

## Quick Start

```bash
# 1. Configure paths in run_preprocessing.py
INPUT_PATH = "PATH_TO_DATASET"
OUTPUT_PATH = "OUTPUT_PATH"

# 2. Run preprocessing
python run_preprocessing.py
```

## What This Does

Transforms the original multilingual dataset into a clean, model-ready format:

1. **Loads** raw data with language labels
2. **Cleans** invalid entries (bad languages, empty hypotheses, duplicates)
3. **Processes English** text (preserves case, removes special chars)
4. **Processes Korean** text (Unicode normalization, slang handling)
5. **Saves** cleaned dataset as `OUTPUT_PATH`

## Pipeline Components

### `preprocessing/` Module

```
preprocessing/
├── text_cleaner.py       - Basic cleaning utilities
├── data_validator.py     - Validation and quality checks
├── data_cleaner.py       - Dataset cleaning operations
├── english_processor.py  - English text processing
├── korean_processor.py   - Korean text processing
├── pipeline.py          - Main orchestrator
└── README.md            - Detailed module docs
```

### Entry Point

- **`run_preprocessing.py`** - Simple script to run the entire pipeline

## Key Features

✅ **Case Preservation** - No lowercasing (optimal for XLM-RoBERTa)
✅ **Modular Design** - Small, focused functions
✅ **Language-Specific** - Separate processors for English & Korean
✅ **Quality Checks** - Validates data at each step
✅ **Progress Tracking** - Clear output messages

## Output

**File:** `OUTPUT_PATH`

**Contents:**
- ~407k samples
- English (94%) & Korean (6%)
- Balanced labels (33% each: entailment, neutral, contradiction)
- Clean, properly encoded text

## Preprocessing Details

### English Processing
- Preserves original case (no lowercasing)
- Removes special characters
- Normalizes punctuation
- Cleans whitespace

### Korean Processing
- Unicode NFC normalization
- Slang replacement
- Hangul character preservation
- Special character cleanup

### Data Cleaning
- Removes invalid languages (keeps only en/ko)
- Removes problematic hypotheses (empty, too short, invalid)
- Removes duplicate premise-hypothesis pairs
- Validates all columns

## Statistics

**Typical run:**
- Original: 417,700 rows
- Removed: ~10,700 rows (2.5%)
- Final: ~407,000 rows
- Processing time: ~30-60 seconds

## Usage in Training

After generating `OUTPUT_PATH`, use it in model training:

```python
# In model-training/config.py
TRAINING_FILE = "PATH_TO_PROCESSED_DATA"
```

## Documentation

- **Module details:** See `preprocessing/README.md`

## Requirements

- pandas
- numpy
- Python 3.8+

No external NLP libraries needed - uses pure Python regex and built-in functions.
