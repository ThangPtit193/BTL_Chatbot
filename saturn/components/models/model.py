import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers import AutoModel
from saturn.components.models.module import SimilarityFunction

class BiencoderRobertaModel(nn.Module):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args
        self.roberta = AutoModel.from_pretrained(config)

    def get_output(
            self,
            input_ids
            attention_mask
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        return outputs

    def forward(
            self,
            input_ids
            attention_mask
            input_ids_positive
            attention_mask_positive
    ):
        outputs = self.get_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        output_positive = self.get_output(
            input_ids=input_ids_positive,
            attention_mask=attention_mask_positive,
        )

        sim_fn = SimilarityFunction(self.args.sim_fn)
        scores = sim_fn(outputs, output_positive)

        labels = torch.arange(scores.size(0)).long()
        loss = nn.CrossEntropyLoss()
        loss = loss(scores, labels)
        return loss
