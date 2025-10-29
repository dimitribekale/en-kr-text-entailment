# Quick Start Guide

Get up and running with the multilingual textual entailment model in 5 minutes!

## 🎯 Using the Pre-trained Model (Easiest)

### Install Dependencies

```bash
pip install transformers torch
```

### Run Inference

```python
from transformers import pipeline

# Load model
classifier = pipeline(
    "text-classification",
    model="bekalebendong/xlm-roberta-large-text-entailment-88"
)

# Predict
result = classifier(
    "The museum is air-conditioned.",
    "The museum has AC."
)

print(result)
# Output: [{'label': 'LABEL_0', 'score': 0.95}]
# LABEL_0 = entailment
```

**Done!** That's all you need to use the model.

---

## 🔧 Training Your Own Model

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/text-entailment.git
cd text-entailment
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

```bash
cd src/data_processing

# Edit download_data.py to set OUTPUT_PATH
# Then run:
python download_data.py
```

### 4. Preprocess Data

```bash
# Edit run_preprocessing.py to set paths
# INPUT_PATH = "path/to/downloaded/data"
# OUTPUT_PATH = "path/to/save/processed/data"

python run_preprocessing.py
```

### 5. Configure Training

Edit `src/model-training/config.py`:

```python
TRAINING_FILE = "path/to/processed/text_entailment.csv"
DEVICE = "cuda"  # or "cpu"
```

### 6. Train

```bash
cd ../model-training
python train.py
```

**That's it!** Training will start and save the best model as `best_model.bin`.

---

## ⚡ Common Use Cases

### 1. Check if Hypothesis Follows from Premise

```python
from transformers import pipeline

nli = pipeline(
    "text-classification",
    model="bekalebendong/xlm-roberta-large-text-entailment-88"
)

premise = "John went to the store."
hypothesis = "John left his house."

result = nli(premise, hypothesis)
print(f"Relationship: {result[0]['label']}")
```

### 2. Fact Checking

```python
claim = "The Earth is flat."
evidence = "The Earth is a spherical planet."

result = nli(evidence, claim)
if result[0]['label'] == 'LABEL_2':  # Contradiction
    print("Claim contradicts evidence!")
```

### 3. Question Answering Verification

```python
context = "Paris is the capital of France."
answer = "Paris is in France."

result = nli(context, answer)
if result[0]['label'] == 'LABEL_0':  # Entailment
    print("Answer is supported by context")
```

### 4. Korean Text

```python
premise = "오늘 날씨가 좋습니다."
hypothesis = "날씨가 맑습니다."

result = nli(premise, hypothesis)
print(result)  # Works with Korean too!
```

---

## 🔍 Label Mapping

The model outputs numerical labels:

| Label | Meaning | Description |
|-------|---------|-------------|
| `LABEL_0` | **Entailment** | Hypothesis must be true |
| `LABEL_1` | **Neutral** | Hypothesis might be true |
| `LABEL_2` | **Contradiction** | Hypothesis must be false |

To get human-readable labels:

```python
label_map = {
    'LABEL_0': 'entailment',
    'LABEL_1': 'neutral',
    'LABEL_2': 'contradiction'
}

result = nli(premise, hypothesis)
readable = label_map[result[0]['label']]
print(f"Prediction: {readable}")
```

---

## 🚨 Troubleshooting

### "CUDA out of memory"
```python
# Use CPU instead
classifier = pipeline(
    "text-classification",
    model="bekalebendong/xlm-roberta-large-text-entailment-88",
    device=-1  # Force CPU
)
```

### "Model download is slow"
```python
# Download once, cache locally
from transformers import AutoModel
AutoModel.from_pretrained(
    "bekalebendong/xlm-roberta-large-text-entailment-88",
    cache_dir="./models"  # Local cache
)
```

### "Import errors"
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt
```

---

## 📚 Next Steps

- **Full Documentation**: See [README.md](README.md)
- **Data Processing**: See `src/data_processing/README.md`
- **Training Details**: See training configuration in `src/model-training/config.py`
- **Upload to Hub**: See `src/model-training/HF_UPLOAD_GUIDE.md`

---

## 🎓 Examples

### Batch Processing

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "bekalebendong/xlm-roberta-large-text-entailment-88"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

pairs = [
    ("The cat sleeps.", "The cat is awake."),
    ("It's raining.", "The weather is wet."),
    ("Store is open.", "You can shop.")
]

for premise, hypothesis in pairs:
    inputs = tokenizer(premise, hypothesis, return_tensors="pt")
    outputs = model(**inputs)
    pred = outputs.logits.argmax().item()
    print(f"{premise} → {hypothesis}: {['Entailment', 'Neutral', 'Contradiction'][pred]}")
```

### With Confidence Scores

```python
import torch

result = classifier(premise, hypothesis)
confidence = result[0]['score']

if confidence > 0.9:
    print("High confidence prediction")
elif confidence > 0.7:
    print("Medium confidence prediction")
else:
    print("Low confidence - manual review recommended")
```

---

**Happy coding! 🚀**
