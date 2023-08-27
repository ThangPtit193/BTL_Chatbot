from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead,
    RobertaModel,
    RobertaPreTrainedModel,
)

from saturn.components.losses import AlignmentLoss, UniformityLoss
from saturn.components.models.module import Pooler, SimilarityFunction


class BiencoderRobertaModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"lm_head.decoder.weight",
        r"lm_head.decoder.bias",
    ]

    # _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: PretrainedConfig, args):
        super().__init__(config)

        self.args = args

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # self.lm_head = RobertaLMHead(config)

        self.pooler = Pooler(self.args.pooler_type)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    # def get_output_embeddings(self):
    #     return self.lm_head.decoder

    # def set_output_embeddings(self, new_embeddings):
    #     self.lm_head.decoder = new_embeddings

    def get_output(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            **kwargs
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        pooled_output = self.pooler(attention_mask, outputs)

        return outputs, pooled_output

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            input_ids_positive: Optional[torch.LongTensor] = None,
            attention_mask_positive: Optional[torch.FloatTensor] = None,
            input_ids_negative: Optional[torch.LongTensor] = None,
            attention_mask_negative: Optional[torch.FloatTensor] = None,
            **kwargs
    ):
        _, pooled_output = self.get_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if not kwargs.get("is_train", ""):
            return pooled_output

        # init loss
        total_loss = 0.0
        loss_ct = 0.0
        loss_ct_dpi_query = 0.0
        loss_ct_dpi_positive = 0.0
        loss_alignment = 0.0
        loss_uniformity = 0.0

        _, pooled_output_positive = self.get_output(
            input_ids=input_ids_positive,
            attention_mask=attention_mask_positive,
        )

        sim_fn = SimilarityFunction(self.args.sim_fn)
        scores = sim_fn(pooled_output, pooled_output_positive)

        if self.args.use_negative:
            _, pooled_output_negative = self.get_output(
                input_ids=input_ids_negative,
                attention_mask=attention_mask_negative,
            )
            scores_negative = sim_fn(pooled_output, pooled_output_negative)
            # hard negative in batch negative
            scores = torch.cat([scores, scores_negative], 1)

            weights = torch.tensor(
                [
                    [0.0] * (scores.size(-1) - scores_negative.size(-1))
                    + [0.0] * i
                    + [self.args.weight_hard_negative]
                    + [0.0] * (scores_negative.size(-1) - i - 1)
                    for i in range(scores_negative.size(-1))
                ]
            ).to(pooled_output.device)
            scores = scores + weights

        # Contrastice Learning Stratege - InBatch
        labels = torch.arange(scores.size(0)).long().to(pooled_output.device)
        loss_fct = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)

        loss_ct = loss_fct(scores, labels)
        total_loss += loss_ct

        if self.args.dpi_query:
            _, pooled_output_dropout = self.get_output(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            scores = sim_fn(pooled_output, pooled_output_dropout)

            labels = torch.arange(scores.size(0)).long().to(pooled_output.device)
            loss_ct_dpi_query = loss_fct(scores, labels)
            total_loss += self.args.coff_dpi_query * loss_ct_dpi_query

        if self.args.dpi_positive:
            _, pooled_output_positive_dropout = self.get_output(
                input_ids=input_ids_positive,
                attention_mask=attention_mask_positive,
            )
            scores = sim_fn(
                pooled_output_positive,
                pooled_output_positive_dropout,
            )

            labels = (
                torch.arange(scores.size(0)).long().to(pooled_output_positive.device)
            )
            loss_ct_dpi_positive = loss_fct(scores, labels)
            total_loss += self.args.coff_dpi_positive * loss_ct_dpi_positive

        if self.args.use_align_loss:
            align_fn = AlignmentLoss()
            loss_alignment = align_fn(pooled_output, pooled_output_positive)
            total_loss += self.args.coff_alignment * loss_alignment

        if self.args.use_uniformity_loss:
            uniformity_fn = UniformityLoss()
            loss_uniformity = -0.5 * uniformity_fn(
                pooled_output
            ) + -0.5 * uniformity_fn(pooled_output_positive)
            total_loss += self.args.coff_uniformity * loss_uniformity

        return (
            total_loss,
            loss_ct,
            loss_ct_dpi_query,
            loss_ct_dpi_positive,
            loss_alignment,
            loss_uniformity,
        )
