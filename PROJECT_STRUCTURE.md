# Project Structure Guide

Complete overview of the project organization and file purposes.

## 📁 Directory Tree

```
text-entailment/
│
├── README.md                          # Main project documentation
├── QUICKSTART.md                      # Quick start guide (5 min setup)
├── PROJECT_STRUCTURE.md              # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore patterns
│
├── src/                              # Source code
│   │
│   ├── data_processing/              # Data preprocessing pipeline
│   │   ├── README.md                 # Data processing guide
│   │   ├── 
│   │   ├── download_data.py          # Download datasets from Google Drive
│   │   ├── run_preprocessing.py      # Main preprocessing entry point
│   │   │
│   │   └── preprocessing/            # Modular preprocessing components
│   │       ├── README.md             # Module documentation
│   │       ├── text_cleaner.py       # Basic text cleaning utilities
│   │       ├── data_validator.py     # Data validation functions
│   │       ├── data_cleaner.py       # Dataset cleaning operations
│   │       ├── english_processor.py  # English text processing
│   │       ├── korean_processor.py   # Korean text processing
│   │       └── pipeline.py          # Main pipeline orchestrator
│   │
│   └── model-training/               # Model training code
│       ├── 
│       ├── config.py                 # Training configuration
│       ├── train.py                  # Main training script
│       ├── class_model.py            # Model architecture wrapper
│       ├── load_dataset.py           # Dataset loading utilities
│       ├── utils.py                  # Training utilities + HF upload functions
│       ├── upload_to_hub.py          # Upload trained model to HF Hub
│       └── example_upload.py         # Example upload script
│
└── datasets-text-entailment/         # Data directory (created separately)
    ├── original_dataset_train.csv    # Original raw data
    ├── dataset_with_language.csv     # Data with language labels
    └── text_entailment.csv          # Processed, training-ready data
```

---

## 📄 File Descriptions

### Root Level

| File | Purpose | When to Use |
|------|---------|-------------|
| `README.md` | Comprehensive project documentation | First thing to read |
| `QUICKSTART.md` | 5-minute setup guide | Want to start quickly |
| `PROJECT_STRUCTURE.md` | This file - explains organization | Understanding structure |
| `requirements.txt` | Python package dependencies | Installation |
| `.gitignore` | Files to exclude from Git | Git setup |

---

### Data Processing (`src/data_processing/`)

#### Main Files

| File | Lines | Purpose |
|------|-------|---------|
| `download_data.py` | ~30 | Download datasets from Google Drive |
| `run_preprocessing.py` | ~35 | Entry point to run preprocessing pipeline |
| `README.md` | - | Data processing documentation |

#### Preprocessing Module (`preprocessing/`)

| File | Lines | Purpose |
|------|-------|---------|
| `pipeline.py` | ~150 | Main orchestrator - chains all operations |
| `data_cleaner.py` | ~130 | Dataset cleaning (remove invalid/duplicates) |
| `data_validator.py` | ~70 | Validation rules and quality checks |
| `english_processor.py` | ~50 | English text processing (case-preserving) |
| `korean_processor.py` | ~85 | Korean text processing (Unicode, slang) |
| `text_cleaner.py` | ~50 | Basic cleaning utilities (shared) |
| `README.md` | - | Module documentation |

**Key Design Principles:**
- ✅ Case preservation (no lowercasing)


---

### Model Training (`src/model-training/`)

| File | Lines | Purpose |
|------|-------|---------|
| `train.py` | ~140 | Main training script with early stopping |
| `config.py` | ~35 | All configuration (hyperparameters, paths, HF settings) |
| `class_model.py` | ~25 | Model wrapper for XLM-RoBERTa |
| `load_dataset.py` | ~55 | PyTorch Dataset and DataLoader |
| `utils.py` | ~530 | Training utilities + HF upload functions |
| `upload_to_hub.py` | ~110 | Interactive script to upload models |
| `example_upload.py` | ~130 | Example code for uploading |
| `HF_UPLOAD_GUIDE.md` | - | Detailed upload guide |

**Key Features:**
- ✅ Mixed precision (FP16)
- ✅ Gradient accumulation
- ✅ Early stopping
- ✅ F1-based model selection
- ✅ Hugging Face integration

---

## 🔄 Data Flow

```
1. Download Data
   download_data.py
   ↓
   dataset_with_language.csv

2. Preprocess
   run_preprocessing.py → preprocessing/pipeline.py
   ↓
   text_entailment.csv

3. Train
   train.py (config.py, class_model.py, load_dataset.py, utils.py)
   ↓
   best_model.bin

4. Upload (Optional)
   upload_to_hub.py
   ↓
   Hugging Face Hub
```

---

## 🎯 Common Tasks

### Task 1: Use Pre-trained Model

**Files needed:**
- None (model is on Hugging Face)

**See:**
- `README.md` → "Using the Trained Model"
- `QUICKSTART.md`

### Task 2: Download Dataset

**Files involved:**
1. `src/data_processing/download_data.py` (configure OUTPUT_PATH)

**Steps:**
```bash
cd src/data_processing
# Edit download_data.py to set OUTPUT_PATH
python download_data.py
```

### Task 3: Preprocess Data

**Files involved:**
1. `src/data_processing/run_preprocessing.py` (configure paths)
2. `src/data_processing/preprocessing/` (all modules)

**Steps:**
```bash
cd src/data_processing
# Edit run_preprocessing.py to set INPUT_PATH and OUTPUT_PATH
python run_preprocessing.py
```

### Task 4: Train Model

**Files involved:**
1. `src/model-training/config.py` (configure everything)
2. `src/model-training/train.py` (run training)
3. `src/model-training/class_model.py` (model definition)
4. `src/model-training/load_dataset.py` (data loading)
5. `src/model-training/utils.py` (training utilities)

**Steps:**
```bash
cd src/model-training
# Edit config.py to set TRAINING_FILE and hyperparameters
python train.py
```

### Task 5: Upload to Hugging Face

**Files involved:**
1. `src/model-training/config.py` (set HF_TOKEN, HF_REPO_NAME)
2. `src/model-training/upload_to_hub.py` (upload script)
3. `src/model-training/utils.py` (upload functions)

**Steps:**
```bash
cd src/model-training
# Edit config.py to set HF credentials
python upload_to_hub.py
```

**Documentation:**
- See `src/model-training/HF_UPLOAD_GUIDE.md`

---

## 📊 File Size Summary

| Component | Files | Total Lines | Purpose |
|-----------|-------|-------------|---------|
| **Root Documentation** | 4 files | ~1000 lines | Project docs |
| **Data Processing** | 8 files | ~600 lines | Data pipeline |
| **Model Training** | 8 files | ~1100 lines | Training + upload |
| **Total** | **20 files** | **~2700 lines** | Complete project |

---

## 🔍 Finding Things

### "I want to change preprocessing"
→ `src/data_processing/preprocessing/`

### "I want to change training hyperparameters"
→ `src/model-training/config.py`

### "I want to understand data processing"
→ `src/data_processing/README.md`

### "I want to upload my model"
→ `src/model-training/HF_UPLOAD_GUIDE.md`

### "I want to use the pre-trained model"
→ `README.md` or `QUICKSTART.md`

### "I want to see training code"
→ `src/model-training/train.py`

### "I want to modify the model architecture"
→ `src/model-training/class_model.py`

---

## 🛠️ Modifying the Project

### Add New Language Processing

1. Create `src/data_processing/preprocessing/spanish_processor.py`
2. Follow pattern from `english_processor.py`
3. Update `pipeline.py` to include new processor

### Change Model Architecture

1. Edit `src/model-training/class_model.py`
2. Modify `Roberta4TextEntailment` class
3. Update `config.py` if new parameters needed

### Add New Features to Training

1. Edit `src/model-training/train.py`
2. Add to training loop
3. Update `config.py` for any new hyperparameters

### Customize Data Cleaning

1. Edit `src/data_processing/preprocessing/data_cleaner.py`
2. Add new cleaning methods
3. Update `pipeline.py` to call them

---

## 🧪 Testing Structure

To test individual components:

```python
# Test English processor
from data_processing.preprocessing.english_processor import EnglishProcessor
processor = EnglishProcessor()
result = processor("Test TEXT!!")
print(result)  # "Test TEXT!!" (case preserved)

# Test Korean processor
from data_processing.preprocessing.korean_processor import KoreanProcessor
processor = KoreanProcessor()
result = processor("테스트ㅋㅋㅋ")
print(result)  # "테스트ㅋㅋ" (slang normalized)

# Test data cleaner
from data_processing.preprocessing.data_cleaner import DataCleaner
import pandas as pd
cleaner = DataCleaner()
df = pd.DataFrame(...)
clean_df = cleaner.clean(df)
```

---

## 📚 Documentation Hierarchy

```
README.md (Start here - comprehensive overview)
    ├── QUICKSTART.md (5-minute quick start)
    ├── PROJECT_STRUCTURE.md (This file - organization)
    │
    ├── src/data_processing/
    │   ├── README.md (Data processing guide)
    │   ├── CLEANUP_SUMMARY.md (What was removed)
    │   └── preprocessing/README.md (Module details)
    │
    └── src/model-training/
        └── HF_UPLOAD_GUIDE.md (Upload to Hugging Face)
```

**Reading Order:**
1. `README.md` - Overview
2. `QUICKSTART.md` - Get started fast
3. Task-specific docs as needed

---

## 🎓 Code Quality

All code follows these principles:

✅ **Modular**: Small, focused files
✅ **Documented**: Comments and docstrings
✅ **Clean**: No long functions (< 30 lines)
✅ **Tested**: All components work independently
✅ **Maintainable**: Clear naming and structure

---

## 🔗 External Resources

- **Model**: [bekalebendong/xlm-roberta-large-text-entailment-88](https://huggingface.co/bekalebendong/xlm-roberta-large-text-entailment-88)
- **Base Model**: [xlm-roberta-large](https://huggingface.co/xlm-roberta-large)
- **Framework**: [Transformers](https://github.com/huggingface/transformers)

---

**This structure is designed for easy navigation, maintenance, and reproduction!** 🚀
