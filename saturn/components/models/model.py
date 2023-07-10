import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from components.models.module import CosineSimilarity
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
        input_ids_query=None,
        attention_mask_query=None,
        token_type_ids_query=None,
        input_ids_document=None,
        attention_mask_document=None,
        token_type_ids_document=None,
        return_dict=None,
    ):
        batch_size = input_ids_query.shape[0]

        labels = torch.arange(
            0, batch_size, dtype=torch.long, device=input_ids_query.device
        )

        outputs_query = self.roberta(
            input_ids_query,
            attention_mask=attention_mask_query,
            token_type_ids=token_type_ids_query,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        pooled_output_query = outputs_query[1]  # [CLS]

        outputs_document = self.roberta(
            input_ids_document,
            attention_mask=attention_mask_document,
            token_type_ids=token_type_ids_document,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        pooled_output_document = outputs_document[1]  # [CLS]

        # Contrastive Loss
        scores = sim_fn(
            pooled_output_query.unsqueeze(1), pooled_output_document.unsqueeze(0)
        )

        loss = torch.nn.functional.cross_entropy(scores, labels)  # TODO label_smoothing

        if not return_dict:
            output = (scores,) + (pooled_output_query,) + (pooled_output_document,)
            return (loss,) + output

        return SequenceClassifierOutput(
            loss=loss,
            logits=loss,
            pooled_output_query=pooled_output_query,
            pooled_output_document=pooled_output_document,
        )


class BiencoderBertModel(BertPreTrainedModel):
    pass
