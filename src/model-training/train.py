import utils
import load_dataset
import config
from class_model import Roberta4TextEntailment

import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from sklearn.utils import resample
from collections import defaultdict
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def run():

    df = pd.read_csv(config.TRAINING_FILE)
    df.drop(columns=["ID"], inplace=True)
    print("[SUCCESS] Data successfully imported.")
    df = resample(df, random_state=config.RANDOM_SEED)
    df['premise'] = df['premise'].astype(str)
    df['hypothesis'] = df['hypothesis'].astype(str)

    df.drop_duplicates(subset=['premise', 'hypothesis'], inplace=True)

    df_train, df_val = train_test_split(
        df,
        test_size=0.2,
        random_state=config.RANDOM_SEED
    )
    df_val, df_test = train_test_split(
        df_val,
        test_size=0.5,
        random_state=config.RANDOM_SEED
    )

    train_data_loader = load_dataset.create_data_loader(df_train, config.TOKENIZER, config.MAX_LEN, config.BATCH_SIZE)
    val_data_loader = load_dataset.create_data_loader(df_val, config.TOKENIZER, config.MAX_LEN, config.BATCH_SIZE)
    test_data_loader = load_dataset.create_data_loader(df_test, config.TOKENIZER, config.MAX_LEN, config.BATCH_SIZE)

    device = torch.device(config.DEVICE)
    model = Roberta4TextEntailment(config.N_CLASSES)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_data_loader) * config.EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    history = defaultdict(list)
    best_accuracy = 0

    print("\nTraining...\n")
    print(f"The device: {device}.\n")
    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        print("-" * 10)

        train_acc, train_loss = utils.train_model(
            model,
            train_data_loader,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = utils.evaluate(
            model,
            val_data_loader,
            device,
            len(df_val)
        )
        print(f'Val   loss {val_loss} accuracy {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model.bin')
            best_accuracy = val_acc

    print("Training completed...")

    print("Testing...\n")
    model.load_state_dict(torch.load('best_model.bin'))
    y_pred, y_pred_probs, y_test = utils.get_predictions(
        model,
        test_data_loader,
        device
    )

    print(classification_report(y_test, y_pred, target_names=['entailment', 'neutral', 'contradiction']))

if __name__ == '__main__':
    run()