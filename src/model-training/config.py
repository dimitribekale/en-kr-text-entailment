import numpy as np
import transformers

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10
N_CLASSES = 3

ROBERTA_PATH = "xlm-roberta-large"
TRAINING_FILE = r"C:\Users\bekal\OneDrive\Desktop\AI4SE\GitHub-Projects\text-entailment\datasets\cleaned_dataset.csv"
TOKENIZER = transformers.AutoTokenizer.from_pretrained(ROBERTA_PATH)

DEVICE = "cuda"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

