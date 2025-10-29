import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.amp import autocast
import config

def train_model(model, data_loader, optimizer, device, scheduler, n_examples, scaler, gradient_accumulation_steps=1):
    model.train()
    losses = []
    correct_predictions = 0

    optimizer.zero_grad()

    for idx, d in enumerate(tqdm(data_loader, total=len(data_loader))):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        # Mixed precision forward pass
        with autocast('cuda'):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits

            #Label smoothing
            if config.LABEL_SMOOTHING > 0:
                loss = F.cross_entropy(
                    logits,
                    targets,
                    label_smoothing=config.LABEL_SMOOTHING
                )
            else:
                loss = F.cross_entropy(logits, targets)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        _, preds = torch.max(logits, dim=1)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item() * gradient_accumulation_steps)  # Unscale for logging

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Update weights every gradient_accumulation_steps
        if (idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def evaluate(model, data_loader, device, n_examples, scaler):
    model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Validating", total=len(data_loader)):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            # Mixed precision inference
            with autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                loss = F.cross_entropy(logits, targets)

            _, preds = torch.max(logits, dim=1)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = correct_predictions.double() / n_examples
    avg_loss = np.mean(losses)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted')

    return accuracy, avg_loss, f1_weighted

def get_predictions(model, data_loader, device):
    model.eval()
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Testing", total=len(data_loader)):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            # Mixed precision inference
            with autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits

            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return predictions, prediction_probs, real_values


# ============================================================
# HUGGING FACE HUB UTILITIES
#
# (AI-Generated code. It worked for me, but you might want
#  to double check before using it)
# ============================================================

def generate_model_card(
    model_name: str,
    f1_score: float,
    accuracy: float,
    training_config: dict,
    classification_report: str = None
) -> str:
    """
    Generate a comprehensive model card for Hugging Face Hub.

    Args:
        model_name: Name of the base model
        f1_score: Best F1-score achieved
        accuracy: Best accuracy achieved
        training_config: Dictionary with training configuration
        classification_report: Optional classification report text

    Returns:
        Markdown-formatted model card
    """

    card = f"""---
language:
- en
- ko
license: mit
tags:
- text-classification
- textual-entailment
- natural-language-inference
- multilingual
- xlm-roberta
datasets:
- snli
- multinli
- klue-nli
metrics:
- f1
- accuracy
model-index:
- name: {model_name}
  results:
  - task:
      type: text-classification
      name: Textual Entailment
    metrics:
    - type: f1
      value: {f1_score:.4f}
      name: F1 Score
    - type: accuracy
      value: {accuracy:.4f}
      name: Accuracy
---

# {model_name} - Multilingual Textual Entailment

## Model Description

This model is a fine-tuned version of `{training_config.get('base_model', 'xlm-roberta-large')}` for multilingual textual entailment (Natural Language Inference) on English and Korean text.

**Task:** Given a premise and a hypothesis, predict whether the hypothesis is:
- **Entailment (0)**: The hypothesis is necessarily true given the premise
- **Neutral (1)**: The hypothesis might be true given the premise
- **Contradiction (2)**: The hypothesis is necessarily false given the premise

## Intended Uses & Limitations

### Intended Uses
- Textual entailment / Natural Language Inference tasks
- English and Korean language pairs
- Research and educational purposes
- Building NLU applications

### Limitations
- Trained primarily on English (94%) and Korean (6%) data
- May not generalize well to other languages
- Performance may vary on out-of-domain text
- Not suitable for tasks requiring deep reasoning or external knowledge

## Training Data

The model was trained on a multilingual dataset containing:
- **English samples:** ~382k
- **Korean samples:** ~25k
- **Total samples:** ~407k premise-hypothesis pairs
- **Label distribution:** Balanced (33% each class)

Data preprocessing:
- Case preservation (no lowercasing) for optimal performance with cased models
- Unicode normalization (NFC) for Korean text
- Special character cleanup
- Duplicate removal

## Training Procedure

### Training Hyperparameters

- **Base model:** `{training_config.get('base_model', 'xlm-roberta-large')}`
- **Learning rate:** {training_config.get('learning_rate', 1e-5)}
- **Batch size:** {training_config.get('batch_size', 32)}
- **Gradient accumulation:** {training_config.get('gradient_accumulation', 4)} steps
- **Effective batch size:** {training_config.get('effective_batch_size', 128)}
- **Epochs:** {training_config.get('epochs', 15)}
- **Max sequence length:** {training_config.get('max_len', 192)}
- **Label smoothing:** {training_config.get('label_smoothing', 0.1)}
- **Weight decay:** 0.01
- **Warmup ratio:** 10%
- **Mixed precision:** FP16
- **Early stopping patience:** {training_config.get('early_stopping_patience', 4)} epochs

### Hardware

- **GPU:** {training_config.get('gpu', 'NVIDIA H100')}
- **Training time:** ~{training_config.get('training_time', '60-90 minutes')}

## Evaluation Results

### Overall Performance

| Metric | Score |
|--------|-------|
| **F1 Score (weighted)** | **{f1_score:.4f}** |
| **Accuracy** | **{accuracy:.4f}** |

### Per-Class Performance

{classification_report if classification_report else 'See detailed classification report in training logs.'}

## Usage

### Direct Usage (Transformers)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "{training_config.get('repo_name', 'your-username/model-name')}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example inference
premise = "Orsay is one of the few Paris museums that is air-conditioned."
hypothesis = "The Orsay museum has air conditioning."

# Tokenize
inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=192)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(predictions, dim=1).item()

# Map to label
label_map = {{0: "entailment", 1: "neutral", 2: "contradiction"}}
print(f"Prediction: {{label_map[label]}} (confidence: {{predictions[0][label]:.4f}})")
```

### Pipeline Usage

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="{training_config.get('repo_name', 'your-username/model-name')}",
    tokenizer="{training_config.get('repo_name', 'your-username/model-name')}"
)

result = classifier(
    "Orsay is one of the few Paris museums that is air-conditioned.",
    "The Orsay museum has air conditioning."
)
print(result)
```

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{xlm-roberta-text-entailment,
  author = {{{training_config.get('author', 'Your Name')}}},
  title = {{Multilingual Textual Entailment with XLM-RoBERTa}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{training_config.get('repo_name', 'your-username/model-name')}}}}}
}}
```

## Acknowledgments

- Base model: [xlm-roberta-large](https://huggingface.co/xlm-roberta-large) by FacebookAI
- Training framework: PyTorch + Hugging Face Transformers
- Optimization: Mixed precision training (FP16) with gradient accumulation

## Model Card Authors

{training_config.get('author', 'Your Name')}

## Model Card Contact

For questions or issues, please open an issue on the [model repository](https://huggingface.co/{training_config.get('repo_name', 'your-username/model-name')}).
"""

    return card


def push_to_huggingface_hub(
    model,
    tokenizer,
    model_path: str,
    repo_name: str = None,
    token: str = None,
    private: bool = False,
    f1_score: float = None,
    accuracy: float = None,
    classification_report: str = None,
    training_config: dict = None
) -> bool:
    """
    Push a trained model to Hugging Face Hub.

    Args:
        model: The trained model instance
        tokenizer: The tokenizer used for training
        model_path: Path to the saved model checkpoint (e.g., 'best_model.bin')
        repo_name: HuggingFace repo name (e.g., 'username/model-name')
        token: HuggingFace API token (or set HF_TOKEN env variable)
        private: Whether to make the repository private
        f1_score: Best F1-score achieved
        accuracy: Best accuracy achieved
        classification_report: Text classification report
        training_config: Dictionary with training configuration

    Returns:
        bool: True if successful, False otherwise

    Example:
        >>> from class_model import Roberta4TextEntailment
        >>> import config
        >>>
        >>> # Load trained model
        >>> model = Roberta4TextEntailment(config.N_CLASSES)
        >>> model.load_state_dict(torch.load('best_model.bin'))
        >>>
        >>> # Push to hub
        >>> push_to_huggingface_hub(
        ...     model=model,
        ...     tokenizer=config.TOKENIZER,
        ...     model_path='best_model.bin',
        ...     repo_name='username/xlm-roberta-text-entailment',
        ...     f1_score=0.88,
        ...     accuracy=0.88
        ... )
    """
    import os
    from huggingface_hub import HfApi, create_repo

    # Validate inputs
    if repo_name is None:
        import config as cfg
        repo_name = cfg.HF_REPO_NAME

    if repo_name is None:
        raise ValueError(
            "Repository name not provided. Set repo_name parameter or "
            "configure config.HF_REPO_NAME"
        )

    # Get token from parameter, config, or environment
    if token is None:
        import config as cfg
        token = cfg.HF_TOKEN

    if token is None:
        token = os.environ.get('HF_TOKEN')

    if token is None:
        raise ValueError(
            "HuggingFace token not provided. Either:\n"
            "1. Pass token parameter\n"
            "2. Set config.HF_TOKEN in config.py\n"
            "3. Set HF_TOKEN environment variable\n"
            "Get your token from: https://huggingface.co/settings/tokens"
        )

    print(f"\n{'='*60}")
    print("PUSHING MODEL TO HUGGING FACE HUB")
    print(f"{'='*60}")
    print(f"Repository: {repo_name}")
    print(f"Private: {private}")

    try:
        # Create repository if it doesn't exist
        print("\n→ Creating/verifying repository...")
        api = HfApi()
        create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print("[OK] Repository ready")

        # Load model state dict if needed
        if isinstance(model_path, str) and os.path.exists(model_path):
            print(f"\n→ Loading model from {model_path}...")
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print("[OK] Model loaded")

        # Prepare training config
        if training_config is None:
            import config as cfg
            training_config = {
                'base_model': cfg.ROBERTA_PATH,
                'learning_rate': cfg.LEARNING_RATE,
                'batch_size': cfg.BATCH_SIZE,
                'gradient_accumulation': cfg.GRADIENT_ACCUMULATION_STEPS,
                'effective_batch_size': cfg.BATCH_SIZE * cfg.GRADIENT_ACCUMULATION_STEPS,
                'epochs': cfg.EPOCHS,
                'max_len': cfg.MAX_LEN,
                'label_smoothing': cfg.LABEL_SMOOTHING,
                'early_stopping_patience': cfg.EARLY_STOPPING_PATIENCE,
                'repo_name': repo_name,
                'gpu': 'NVIDIA H100/RTX 4060 Ti',
                'author': 'Your Name'  # Update this!
            }

        # Generate model card
        if f1_score is not None and accuracy is not None:
            print("\n→ Generating model card...")
            model_card = generate_model_card(
                model_name=repo_name.split('/')[-1],
                f1_score=f1_score,
                accuracy=accuracy,
                training_config=training_config,
                classification_report=classification_report
            )

            # Save model card temporarily
            with open('README.md', 'w', encoding='utf-8') as f:
                f.write(model_card)
            print("[OK] Model card generated")

        # Push model and tokenizer
        print("\n→ Pushing model to Hub...")
        model.roberta.push_to_hub(
            repo_id=repo_name,
            token=token,
            commit_message=f"Upload XLM-RoBERTa model (F1: {f1_score:.4f})" if f1_score else "Upload model"
        )
        print("[OK] Model pushed")

        print("\n→ Pushing tokenizer to Hub...")
        tokenizer.push_to_hub(
            repo_id=repo_name,
            token=token,
            commit_message="Upload tokenizer"
        )
        print("[OK] Tokenizer pushed")

        # Upload model card if generated
        if f1_score is not None and os.path.exists('README.md'):
            print("\n→ Uploading model card...")
            api.upload_file(
                path_or_fileobj='README.md',
                path_in_repo='README.md',
                repo_id=repo_name,
                token=token,
                commit_message="Add model card"
            )
            os.remove('README.md')  # Cleanup
            print("[OK] Model card uploaded")

        print(f"\n{'='*60}")
        print("✓ SUCCESSFULLY PUSHED TO HUGGING FACE HUB!")
        print(f"{'='*60}")
        print(f"\nView your model at:")
        print(f"https://huggingface.co/{repo_name}")
        print(f"\nUse it with:")
        print(f'  from transformers import AutoModelForSequenceClassification')
        print(f'  model = AutoModelForSequenceClassification.from_pretrained("{repo_name}")')

        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to push model to Hub: {e}")
        print("\nTroubleshooting:")
        print("1. Check your token has write permissions")
        print("2. Verify repository name format: 'username/repo-name'")
        print("3. Ensure you have internet connection")
        print("4. Try: pip install --upgrade huggingface_hub")
        return False
