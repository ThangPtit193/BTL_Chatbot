from typing import *

from comet.components.embeddings.embedding_models import BertEmbedder


class SentenceEmbedder(BertEmbedder):

    def __init__(self):
        pass

    def initialize(self, **kwargs):
        for k, val in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, val)

    def load_model(self, cache_path=None, pretrained_name_or_abspath=None):
        super(SentenceEmbedder, self).__init__(cache_path, pretrained_name_or_abspath)

    @staticmethod
    def train(
        triplets_data: List[Dict],
        pretrained_model="bert-base-uncased",
        model_save_path: Text = None,
        n_samples: int = 5,
        batch_size: int = 128,
        epochs: int = 5,
        warmup_steps: int = 5000,
        evaluation_steps: int = 2000,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
        save_best_model: bool = True,
        show_progress_bar: bool = True
    ):
        """
        Train the sentence embedding model
        :param triplets_data: The triplets data
        :param pretrained_model: The pretrained model
        :param model_save_path: Where to save the model
        :param n_samples:
        :param batch_size:
        :param epochs:
        :param warmup_steps:
        :param evaluation_steps:
        :param weight_decay:
        :param max_grad_norm:
        :param use_amp:
        :param save_best_model:
        :param show_progress_bar:
        :return:
        """
        from venus.dataset_reader.TripletDataset import TripletsDataset
        from venus.sentence_embedding.sentence_embedding import SentenceEmbedding

        sentence_embedding = SentenceEmbedding.from_pretrained(
            model_name_or_path=pretrained_model
        )

        sentence_embedding.model.max_seq_length = 128

        train_dataset = TripletsDataset(
            triplet_examples=triplets_data,
            query_key="query",
            pos_key="pos",
            neg_key="neg"
        )
        sentence_embedding.train_negative_ranking(
            train_dataset=train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            warmup_steps=warmup_steps,
            model_save_path=model_save_path,
            evaluation_steps=evaluation_steps,
            use_amp=use_amp,
            scheduler='WarmupLinear',
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            save_best_model=save_best_model,
            show_progress_bar=show_progress_bar,
        )
