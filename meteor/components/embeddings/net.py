import os
import os
from typing import *

import torch
import transformers
from sentence_transformers import losses
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
)
from venus.sentence_embedding.sentence_embedding import SentenceEmbedding


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


class CustomSentenceEmbedding(SentenceEmbedding):
    def train_negative_ranking(
        self,
        train_dataset,
        dev_evaluator=None,
        batch_size=16,
        epochs=10,
        warmup_steps=1000,
        model_save_path='models',
        evaluation_steps=5000,
        use_amp=True,
        optimizer_params: Dict[str, object] = {'lr': 2e-5},
        optimizer_class: Type[Optimizer] = transformers.AdamW,
        scheduler='WarmupLinear',
        weight_decay: float = 0.01,
        max_grad_norm: float = 1,
        save_best_model=True,
        show_progress_bar: bool = True,
    ):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        # train_loss = losses.MultipleNegativesRankingLoss(model=self.model)
        train_loss = losses.TripletLoss(model=self.model)

        # Train the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            evaluation_steps=evaluation_steps,
            use_amp=use_amp,
            optimizer_params=optimizer_params,
            optimizer_class=optimizer_class,
            scheduler=scheduler,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            save_best_model=save_best_model,
            show_progress_bar=show_progress_bar
        )
        self.model.save(model_save_path)
