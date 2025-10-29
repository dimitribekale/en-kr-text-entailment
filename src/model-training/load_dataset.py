import torch
from torch.utils.data import DataLoader, Dataset


class TextEntData(Dataset):

    def __init__(self, premises, hypotheses, targets, tokenizer, max_len):
        self.premises = premises
        self.hypotheses = hypotheses
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, item):
        premise = str(self.premises[item])
        hypothesis = str(self.hypotheses[item])

        encoding = self.tokenizer.encode_plus(
            premise, 
            hypothesis, 
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'premise_text': premise,
            'hypothesis_text': hypothesis,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(self.targets[item], dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size, shuffle=True):
    ds = TextEntData(
        premises=df.premise.to_numpy(),
        hypotheses=df.hypothesis.to_numpy(),
        targets=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle
    )
