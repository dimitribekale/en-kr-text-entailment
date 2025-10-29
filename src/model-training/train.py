import utils
import load_dataset
import config
from class_model import Roberta4TextEntailment

import torch
import pandas as pd
from torch.optim import AdamW
from torch.amp import GradScaler
from sklearn.utils import resample
from collections import defaultdict
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report


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
        random_state=config.RANDOM_SEED,
        stratify=df['label']
    )
    df_val, df_test = train_test_split(
        df_val,
        test_size=0.5,
        random_state=config.RANDOM_SEED,
        stratify=df_val['label']
    )

    train_data_loader = load_dataset.create_data_loader(df_train, config.TOKENIZER, config.MAX_LEN, config.BATCH_SIZE, shuffle=True)
    val_data_loader = load_dataset.create_data_loader(df_val, config.TOKENIZER, config.MAX_LEN, config.BATCH_SIZE, shuffle=False)
    test_data_loader = load_dataset.create_data_loader(df_test, config.TOKENIZER, config.MAX_LEN, config.BATCH_SIZE, shuffle=False)

    device = torch.device(config.DEVICE)
    model = Roberta4TextEntailment(config.N_CLASSES)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)

    # Adjust total steps for gradient accumulation
    total_steps = (len(train_data_loader) // config.GRADIENT_ACCUMULATION_STEPS) * config.EPOCHS
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Mixed precision training scaler
    scaler = GradScaler('cuda')

    history = defaultdict(list)
    best_f1 = 0
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        print("-" * 10)

        train_acc, train_loss = utils.train_model(
            model,
            train_data_loader,
            optimizer,
            device,
            scheduler,
            len(df_train),
            scaler,
            config.GRADIENT_ACCUMULATION_STEPS
        )
        print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')

        val_acc, val_loss, val_f1 = utils.evaluate(
            model,
            val_data_loader,
            device,
            len(df_val),
            scaler
        )
        print(f'Val   loss {val_loss:.4f} accuracy {val_acc:.4f} F1-score {val_f1:.4f}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        if val_f1 > best_f1:
            torch.save(model.state_dict(), 'best_model.bin')
            best_f1 = val_f1
            epochs_no_improve = 0
            print(f'\nBest model saved with F1-score: {best_f1:.4f}')
        else:
            epochs_no_improve += 1
            print(f'\nNo improvement for {epochs_no_improve} epoch(s)')

            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f'\nEarly stopping triggered! No improvement for {config.EARLY_STOPPING_PATIENCE} epochs.')
                print(f'   Best F1-score: {best_f1:.4f}')
                early_stop = True
                break

    if early_stop:
        print(f"\nTraining stopped early at epoch {epoch + 1}/{config.EPOCHS}")
    else:
        print("\nTraining completed all epochs")

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