import config
import transformers
import torch.nn as nn


class Roberta4TextEntailment(nn.Module):

    def __init__(self, n_classes):
        super(Roberta4TextEntailment, self).__init__()
        # Configure model with label smoothing for better generalization
        model_config = transformers.XLMRobertaConfig.from_pretrained(
            config.ROBERTA_PATH,
            num_labels=n_classes,
            problem_type="single_label_classification"
        )
        self.roberta = transformers.XLMRobertaForSequenceClassification.from_pretrained(
            config.ROBERTA_PATH,
            config=model_config
        )
        self.label_smoothing = config.LABEL_SMOOTHING

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output
