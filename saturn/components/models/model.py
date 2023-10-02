import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaPreTrainedModel,
)
from saturn.components.models.module import SimilarityFunction

class BiencoderRobertaModel(RobertaPreTrainedModel):
    def __init__(self, config: PretrainedConfig, args):
        super().__init__(config)
        self.args = args
        self.roberta = RobertaModel(config)

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
        scores = sim_fn(pooled_output, pooled_output_positive)

        labels = torch.arange(scores.size(0)).long().to(pooled_output.device)
        loss_fct = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)

        loss_ct = loss_fct(scores, labels)

        return loss_ct
