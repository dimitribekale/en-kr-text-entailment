import numpy as np
import transformers

MAX_LEN = 192  
BATCH_SIZE = 32 
GRADIENT_ACCUMULATION_STEPS = 4 
EPOCHS = 15 
EARLY_STOPPING_PATIENCE = 4 
N_CLASSES = 3

ROBERTA_PATH = "xlm-roberta-large" 
LEARNING_RATE = 1e-5 
LABEL_SMOOTHING = 0.1 
TRAINING_FILE = "PATH_TO_DATASET"
TOKENIZER = transformers.AutoTokenizer.from_pretrained(ROBERTA_PATH)

DEVICE = "cuda"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# HUGGING FACE HUB CONFIGURATION
# ============================================================

HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN"
HF_REPO_NAME = "YOUR_REPOSITORY_NAME"
HF_PRIVATE = False  # Set to True to make the model repository private

