import os

import torch
from transformers import (
    AutoModel,
)


class AutoModelForSentenceEmbedding(torch.nn.Module):
    def __init__(self, model_name, tokenizer, normalize=True):
        super(AutoModelForSentenceEmbedding, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.tokenizer = tokenizer

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)

    def save_pretrained(self, output_path):
        self.tokenizer.save_pretrained(output_path)
        self.model.config.save_pretrained(output_path)
        torch.save(self.model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))