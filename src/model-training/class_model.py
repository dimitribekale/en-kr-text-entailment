import config
import transformers
import torch.nn as nn


class Roberta4TextEntailment(nn.Module):

    def __init__(self, n_classes):
        super(Roberta4TextEntailment, self).__init__()
        self.roberta = transformers.XLMRobertaForSequenceClassification.from_pretrained(
            config.ROBERTA_PATH,
            num_labels=n_classes
        )

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output
