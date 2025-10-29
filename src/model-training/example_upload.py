"""
================================================================
================================================================
AI-Generated code. It worked just fine for me, but you might
want to double check before using it.
================================================================
================================================================

"""


import torch
import config
from class_model import Roberta4TextEntailment
from utils import push_to_huggingface_hub


# ============================================================
# CONFIGURATION
# ============================================================
# Set these in config.py instead of here:
# config.HF_TOKEN = "hf_xxxxxxxxxxxxx"
# config.HF_REPO_NAME = "your-username/model-name"

# Or override here for testing:
REPO_NAME = "john-doe/xlm-roberta-large-en-ko-nli"  # Your repository
TOKEN = None  # Will use config.HF_TOKEN or HF_TOKEN env variable
PRIVATE = False  # Set True for private repository


# ============================================================
# LOAD YOUR TRAINED MODEL
# ============================================================
print("Loading model...")
model = Roberta4TextEntailment(config.N_CLASSES)
model.load_state_dict(torch.load('best_model.bin', map_location='cpu'))
print("Model loaded successfully!")


# ============================================================
# PUSH TO HUB
# ============================================================
print("\nUploading to Hugging Face Hub...")

success = push_to_huggingface_hub(
    model=model,
    tokenizer=config.TOKENIZER,
    model_path='best_model.bin',

    # Repository settings
    repo_name=REPO_NAME,  # Or use config.HF_REPO_NAME
    token=TOKEN,           # Or use config.HF_TOKEN
    private=PRIVATE,       # Or use config.HF_PRIVATE

    # Model metrics (optional - for model card)
    f1_score=0.88,         # Your best F1-score
    accuracy=0.88,         # Your best accuracy

    # Classification report (optional)
    classification_report="""
               precision    recall  f1-score   support

   entailment       0.86      0.90      0.88      8606
      neutral       0.84      0.77      0.80      8530
contradiction       0.90      0.88      0.89      8600

     accuracy                           0.88     25736
    macro avg       0.87      0.85      0.86     25736
 weighted avg       0.88      0.88      0.88     25736
    """,

    # Training config (optional - auto-detected from config.py)
    training_config=None  # Leave as None to auto-detect
)

if success:
    print(f"\n✓ Model successfully uploaded!")
    print(f"View at: https://huggingface.co/{REPO_NAME}")
else:
    print("\n✗ Upload failed. Check error messages above.")


# ============================================================
# TEST YOUR UPLOADED MODEL
# ============================================================
print("\n" + "="*60)
print("TESTING UPLOADED MODEL")
print("="*60)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

print(f"\nLoading from Hub: {REPO_NAME}")
tokenizer = AutoTokenizer.from_pretrained(REPO_NAME)
model_from_hub = AutoModelForSequenceClassification.from_pretrained(REPO_NAME)

# Test inference
premise = "The Orsay museum is air-conditioned."
hypothesis = "The museum has air conditioning."

print(f"\nPremise: {premise}")
print(f"Hypothesis: {hypothesis}")

inputs = tokenizer(premise, hypothesis, return_tensors="pt", max_length=192)
outputs = model_from_hub(**inputs)

prediction = outputs.logits.argmax().item()
confidence = torch.softmax(outputs.logits, dim=1)[0][prediction].item()

labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
print(f"\nPrediction: {labels[prediction]}")
print(f"Confidence: {confidence:.4f}")

print("\n✓ Model works! Ready to share with the world! 🚀")
