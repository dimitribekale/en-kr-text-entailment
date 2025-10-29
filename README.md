# Multilingual Textual Entailment with XLM-RoBERTa

A high-performance multilingual Natural Language Inference (NLI) system for English and Korean, achieving **88% F1-score** on textual entailment tasks.

[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/bekalebendong/xlm-roberta-large-text-entailment-88)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Model Performance](#model-performance)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Using the Trained Model](#using-the-trained-model)
- [Results](#results)
- [Technical Details](#technical-details)
- [Citation](#citation)

---

## 🎯 Overview

This project implements a state-of-the-art multilingual textual entailment system using **XLM-RoBERTa-large**. Given a premise and hypothesis pair, the model predicts:

- **Entailment (0)**: Hypothesis is necessarily true given the premise
- **Neutral (1)**: Hypothesis might be true given the premise
- **Contradiction (2)**: Hypothesis is necessarily false given the premise

### Key Features

✅ **High Performance**: 88% F1-score on test set
✅ **Multilingual**: Supports English and Korean
✅ **Production-Ready**: Clean, modular, well-documented code
✅ **Optimized Training**: Mixed precision (FP16), gradient accumulation, early stopping
✅ **Easy to Use**: Pre-trained model available on Hugging Face Hub

---

## 🏆 Model Performance

| Metric | Score |
|--------|-------|
| **F1-Score (weighted)** | **88.0%** |
| **Accuracy** | **88.0%** |
| **Entailment F1** | 86% |
| **Neutral F1** | 84% |
| **Contradiction F1** | 89% |

**Model**: [bekalebendong/xlm-roberta-large-text-entailment-88](https://huggingface.co/bekalebendong/xlm-roberta-large-text-entailment-88)

---

## 🚀 Quick Start

### Using the Pre-trained Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model from Hugging Face Hub
model_name = "bekalebendong/xlm-roberta-large-text-entailment-88"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example inference
premise = "The Orsay museum is air-conditioned."
hypothesis = "The museum has air conditioning."

inputs = tokenizer(premise, hypothesis, return_tensors="pt", max_length=192)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)
    label_id = torch.argmax(predictions, dim=1).item()

labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
print(f"Prediction: {labels[label_id]}")
print(f"Confidence: {predictions[0][label_id]:.4f}")
```

### Using Pipeline API

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="bekalebendong/xlm-roberta-large-text-entailment-88"
)

result = classifier("The museum is air-conditioned.", "The museum has AC.")
print(result)
```

---

## 📁 Project Structure

```
text-entailment/
├── README.md                          # This file
├── src/
│   ├── data_processing/              # Data preprocessing pipeline
│   │   ├── download_data.py          # Download datasets from Google Drive
│   │   ├── run_preprocessing.py      # Main preprocessing script
│   │   ├── preprocessing/            # Modular preprocessing modules
│   │   │   ├── text_cleaner.py       # Basic cleaning utilities
│   │   │   ├── data_validator.py     # Validation logic
│   │   │   ├── data_cleaner.py       # Cleaning operations
│   │   │   ├── english_processor.py  # English text processing
│   │   │   ├── korean_processor.py   # Korean text processing
│   │   │   ├── pipeline.py          # Pipeline orchestrator
│   │   │   └── README.md            # Module documentation
│   │   └── README.md                 # Processing guide
│   │
│   └── model-training/               # Model training code
│       ├── config.py                 # Training configuration
│       ├── train.py                  # Main training script
│       ├── class_model.py            # Model architecture
│       ├── load_dataset.py           # Data loading utilities
│       ├── utils.py                  # Training utilities + HF upload
│       ├── upload_to_hub.py          # Upload to Hugging Face
│       └── example_upload.py         # Upload examples
│
└── datasets-text-entailment/         # Datasets (downloaded separately)
    ├── original_dataset_train.csv    # Original data
    ├── dataset_with_language.csv     # Data with language labels
    └── text_entailment.csv          # Processed, ready for training
```

---

## 🔧 Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: NVIDIA H100/RTX 4060 Ti or better)
- 16GB+ RAM
- 10GB+ disk space

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/text-entailment.git
cd text-entailment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=2.0.0
transformers>=4.30.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
tqdm>=4.65.0
huggingface_hub>=0.16.0
gdown>=4.7.0  # For dataset download
```

---

## 📊 Dataset

### Download

The project uses a multilingual textual entailment dataset with **~407K samples** (English and Korean).

#### Option 1: Download via Script

```bash
cd src/data_processing
python download_data.py
```

Edit `download_data.py` to set:
- `OUTPUT_PATH`: Where to save the dataset
- `ORIGINAL_DATA_ID` or `PROCESSED_DATA_ID`: Choose which version

#### Option 2: Manual Download

- **Original Data**: [Google Drive Link](https://drive.google.com/file/d/1utmqvJlvnEGHfz6bHs_fQg0fgby0leA)
- **Processed Data**: [Google Drive Link](https://drive.google.com/file/d/1bE3AkPbOsH_9cfrIN9QZiw7zAGGu4FW4)

Place downloaded files in `datasets-text-entailment/` folder.

### Dataset Statistics

| Attribute | Value |
|-----------|-------|
| Total Samples | ~407,000 |
| English Samples | 382,000 (94%) |
| Korean Samples | 25,000 (6%) |
| Classes | 3 (balanced) |
| Train/Val/Test Split | 80/10/10 |

### Data Format

```csv
ID,premise,hypothesis,label,language
0,"The museum is air-conditioned.","The museum has AC.",0,en
1,"안녕하세요.","반갑습니다.",1,ko
```

Labels: `0=entailment`, `1=neutral`, `2=contradiction`

---

## 🔄 Data Processing

Process raw data into training-ready format with optimized preprocessing.

### Key Features

- ✅ **Case Preservation**: No lowercasing (optimal for cased models)
- ✅ **No Contraction Expansion**: Modern tokenizers handle it
- ✅ **Unicode Normalization**: Proper Korean text handling
- ✅ **Quality Filtering**: Removes invalid/duplicate entries
- ✅ **Modular Design**: Easy to extend/modify

### Usage

```bash
cd src/data_processing

# Configure paths in run_preprocessing.py
# INPUT_PATH = "path/to/dataset_with_language.csv"
# OUTPUT_PATH = "path/to/text_entailment.csv"

python run_preprocessing.py
```

### What It Does

1. **Load** raw data with language labels
2. **Clean** invalid entries (bad languages, empty hypotheses)
3. **Process English**: Preserve case, clean special characters
4. **Process Korean**: Unicode normalization, slang handling
5. **Remove** duplicates
6. **Save** cleaned dataset

### Output

- **File**: `text_entailment.csv`
- **Samples**: ~407,000
- **Processing time**: ~30-60 seconds

**See detailed documentation**: `src/data_processing/README.md`

---

## 🎓 Model Training

Train your own model or fine-tune the existing one.

### Configuration

Edit `src/model-training/config.py`:

```python
# Model Configuration
ROBERTA_PATH = "xlm-roberta-large"
MAX_LEN = 192
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size: 128
LEARNING_RATE = 1e-5
LABEL_SMOOTHING = 0.1
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4

# Data
TRAINING_FILE = "../datasets-text-entailment/text_entailment.csv"

# Hardware
DEVICE = "cuda"
```

### Training

```bash
cd src/model-training
python train.py
```

### Training Features

✅ **Mixed Precision (FP16)**: 2x faster training
✅ **Gradient Accumulation**: Large effective batch size
✅ **Early Stopping**: Stops when F1 plateaus
✅ **Label Smoothing**: Better generalization
✅ **Stratified Splitting**: Balanced train/val/test
✅ **F1-based Model Selection**: Saves best checkpoint

### Training Output

```
Training Configuration:
  Device: cuda
  Model: xlm-roberta-large
  Max sequence length: 192
  Batch size: 32
  Effective batch size: 128
  Learning rate: 1e-5
  Mixed precision: Enabled (FP16)
  Early stopping patience: 4 epochs

Epoch 1/15
----------
Train loss 0.7990 accuracy 0.6212
Validating: 100%|████████████| 403/403 [00:15<00:00]
Val   loss 0.5249 accuracy 0.7894 F1-score 0.7893

Best model saved with F1-score: 0.7893
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 12GB | H100 80GB |
| RAM | 16GB | 32GB+ |
| Training Time | ~4 hours | ~60-90 min |

---

## 🎯 Using the Trained Model

### From Hugging Face Hub

```python
from transformers import pipeline

# Initialize classifier
nli_classifier = pipeline(
    "text-classification",
    model="bekalebendong/xlm-roberta-large-text-entailment-88"
)

# English example
result = nli_classifier(
    "The Eiffel Tower is in Paris.",
    "Paris has the Eiffel Tower."
)
print(result)  # [{'label': 'LABEL_0', 'score': 0.95}]  # Entailment

# Korean example
result = nli_classifier(
    "박물관은 에어컨이 있습니다.",
    "박물관이 시원합니다."
)
print(result)  # Entailment
```

### Batch Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "bekalebendong/xlm-roberta-large-text-entailment-88"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Batch of premise-hypothesis pairs
pairs = [
    ("The cat is sleeping.", "The cat is awake."),
    ("It's raining outside.", "The weather is wet."),
    ("The store is closed.", "You can shop now.")
]

labels_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

for premise, hypothesis in pairs:
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", max_length=192)

    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax().item()

    print(f"Premise: {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Prediction: {labels_map[pred]}\n")
```

### REST API Example

```python
# Using FastAPI
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
classifier = pipeline(
    "text-classification",
    model="bekalebendong/xlm-roberta-large-text-entailment-88"
)

@app.post("/predict")
def predict(premise: str, hypothesis: str):
    result = classifier(premise, hypothesis)
    return {"prediction": result[0]['label'], "confidence": result[0]['score']}
```

---

## 📈 Results

### Overall Performance

| Split | F1-Score | Accuracy |
|-------|----------|----------|
| Training | 95.35% | 95.35% |
| Validation | 82.96% | 82.96% |
| **Test** | **88.0%** | **88.0%** |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Entailment | 0.86 | 0.90 | 0.88 | 8,606 |
| Neutral | 0.84 | 0.77 | 0.80 | 8,530 |
| Contradiction | 0.90 | 0.88 | 0.89 | 8,600 |
| **Weighted Avg** | **0.88** | **0.88** | **0.88** | **25,736** |

### Training History

```
Epoch 1:  Val F1: 0.7893
Epoch 2:  Val F1: 0.8193
Epoch 3:  Val F1: 0.8252
Epoch 4:  Val F1: 0.8267
Epoch 5:  Val F1: 0.8293
Epoch 6:  Val F1: 0.8273
Epoch 7:  Val F1: 0.8276
Epoch 8:  Val F1: 0.8296 ← Best model
Epoch 9:  Val F1: 0.8282
Epoch 10: Val F1: 0.8281
```

---

## 🔬 Technical Details

### Model Architecture

- **Base Model**: `xlm-roberta-large`
- **Parameters**: 560M
- **Layers**: 24 transformer layers
- **Hidden Size**: 1024
- **Attention Heads**: 16
- **Vocabulary Size**: 250K tokens
- **Max Sequence Length**: 192

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-5 |
| Batch Size | 32 |
| Gradient Accumulation | 4 steps |
| Effective Batch Size | 128 |
| Epochs | 15 (early stopped at 10) |
| Warmup Steps | 10% of total |
| Weight Decay | 0.01 |
| Label Smoothing | 0.1 |
| Max Seq Length | 192 |
| Optimizer | AdamW |
| LR Scheduler | Linear with warmup |
| Mixed Precision | FP16 |

### Optimization Techniques

1. **Mixed Precision Training**: 2x speedup, 50% memory reduction
2. **Gradient Accumulation**: Large effective batch size on limited VRAM
3. **Early Stopping**: Prevents overfitting (patience: 4 epochs)
4. **Label Smoothing**: Better generalization (ε=0.1)
5. **Stratified Splitting**: Maintains class balance
6. **F1-based Selection**: Optimizes for classification performance

### Data Processing

**Key Decisions:**
- ✅ **Case Preserved**: No lowercasing (XLM-RoBERTa is cased)
- ✅ **No Contraction Expansion**: Model tokenizer handles it
- ✅ **Unicode NFC Normalization**: Critical for Korean
- ✅ **Minimal Preprocessing**: Preserves semantic information

---

## 🚀 Uploading Models to Hugging Face

### Configure

Edit `src/model-training/config.py`:

```python
HF_TOKEN = "hf_xxxxxxxxxxxxx"  # Your Hugging Face token
HF_REPO_NAME = "your-username/model-name"
HF_PRIVATE = False  # True for private repos
```

### Upload

```bash
cd src/model-training
python upload_to_hub.py
```

**See detailed guide**: `src/model-training/HF_UPLOAD_GUIDE.md`

---

## 📚 Citation

If you use this model or code in your research, please cite:

```bibtex
@misc{xlm-roberta-text-entailment-2025,
  author = {Beka Bendong},
  title = {Multilingual Textual Entailment with XLM-RoBERTa},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/bekalebendong/xlm-roberta-large-text-entailment-88}},
  note = {88\% F1-score on English-Korean textual entailment}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

---

## 📝 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 🙏 Acknowledgments

- **Base Model**: [XLM-RoBERTa](https://huggingface.co/xlm-roberta-large) by FacebookAI
- **Framework**: [Hugging Face Transformers](https://github.com/huggingface/transformers)
- **Optimization**: PyTorch mixed precision training
- **Hardware**: NVIDIA H100 GPU

---

## 📧 Contact

- **Model**: [bekalebendong/xlm-roberta-large-text-entailment-88](https://huggingface.co/bekalebendong/xlm-roberta-large-text-entailment-88)
- **Issues**: [GitHub Issues](https://github.com/yourusername/text-entailment/issues)

---

## 📖 Additional Documentation

- **Data Processing**: `src/data_processing/README.md`
- **Preprocessing Modules**: `src/data_processing/preprocessing/README.md`
- **Hugging Face Upload Guide**: `src/model-training/HF_UPLOAD_GUIDE.md`
- **Refactoring Summary**: `src/data_processing/REFACTORING_SUMMARY.md`

---

**Built with ❤️ for multilingual NLP research**
