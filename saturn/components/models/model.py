import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from saturn.components.models.module import CosineSimilarity
from transformers import PretrainedConfig, RobertaTokenizer
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)
from transformers.models.bert.modeling_bert import (
    BertLMPredictionHead,
    BertModel,
    BertPreTrainedModel,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead,
    RobertaModel,
    RobertaPreTrainedModel,
)

sim_fn = CosineSimilarity()


class BiencoderRobertaModel(RobertaPreTrainedModel):
    def __init__(self, config: PretrainedConfig, args, *inputs, **kwargs):
        super(BiencoderRobertaModel, self).__init__(config, *inputs, **kwargs)

        self.args = args
        self.roberta = RobertaModel(config)

        # Other head

        # MLM

        # KLD

        # Harnegative

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        input_ids_positive=None,
        attention_mask_positive=None,
        token_type_ids_positive=None,
        is_trainalbe=True,
        return_dict=None,
    ):
        batch_size = input_ids.shape[0]

        labels = torch.arange(
            0, batch_size, dtype=torch.long, device=input_ids.device
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        pooled_output = outputs[1]  # [CLS]

        if not is_trainalbe:
            return pooled_output

        outputs_positive = self.roberta(
            input_ids_positive,
            attention_mask=attention_mask_positive,
            token_type_ids=token_type_ids_positive,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        pooled_output_positive = outputs_positive[1]  # [CLS]

        # Contrastive Loss
        scores = sim_fn(
            pooled_output.unsqueeze(1), pooled_output_positive.unsqueeze(0)
        )

        loss = torch.nn.functional.cross_entropy(scores, labels)  # TODO label_smoothing

        if not return_dict:
            output = (scores,) + (pooled_output,) + (pooled_output_positive,)
            return (loss,) + output

        return SequenceClassifierOutput(
            loss=loss,
            logits=loss,
            pooled_output=pooled_output,
            pooled_output_positive=pooled_output_positive,
        )


class BiencoderBertModel(BertPreTrainedModel):
    pass
