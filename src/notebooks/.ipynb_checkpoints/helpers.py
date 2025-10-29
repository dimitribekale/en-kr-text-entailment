import torch
import numpy as np
from sklearn.metrics import (accuracy_score, 
                             precision_recall_fscore_support,
                             )

def check_gpu_availability():
    """Safe GPU detection that works with CUDA 12.9"""
    try:
        if torch.cuda.is_available():
            print(f"CUDA available: True")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            device = torch.device('cuda')
        else:
            print("CUDA available: False")
            print("Running on CPU")
            device = torch.device('cpu')
    except Exception as e:
        print(f"GPU detection failed: {e}")
        print("Falling back to CPU")
        device = torch.device('cpu')
    
    return device

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Use zero_division parameter to handle undefined metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, 
        average='weighted', 
        zero_division=0  # Set undefined metrics to 0
    )
    accuracy = accuracy_score(labels, predictions)
    
    # Also get per-class metrics for debugging
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, 
        average=None, 
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support.tolist()
    }