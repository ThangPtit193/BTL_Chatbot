import os
import time
from abc import abstractmethod
from typing import *

import torch
import torch.multiprocessing as mp
import transformers
from comet.components.embeddings.embedding_models import BertEmbedder
from comet.lib import file_util
from sentence_transformers import losses
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .dataset import data_producer
from .net import AutoModelForSentenceEmbedding, CustomSentenceTransformer

__all__ = []
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class BaseEmbedder(BertEmbedder):
    def __init__(self):
        pass

    def initialize(self, **kwargs):
        for k, val in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, val)

    def train(self, trainer_config: Dict):
        self._train(**trainer_config)

    @abstractmethod
    def _train(self, *args, **kwargs):
        raise NotImplementedError


class NaiveEmbedder(BaseEmbedder):
    def initialize(self, **kwargs):
        for k, val in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, val)

    def load_model(self, cache_path=None, pretrained_name_or_abspath=None):
        super(BaseEmbedder, self).__init__(cache_path, pretrained_name_or_abspath)

    def _train(
        self,
        data_config_path: Text,
        datasets_per_batch: int = 2,
        num_same_dataset: int = 2,
        batch_size: int = 32,
        pretrained_model='microsoft/MiniLM-L12-H384-uncased',
        steps: int = 20000,
        max_length: int = 128,
        scale: int = 20,
        save_steps: int = 10000,
        model_save_path: Text = "models",
        **kwargs
    ):
        # Load data config
        queue = mp.Queue(maxsize=100)

        # Loading data config conent
        assert os.path.exists(data_config_path), f"Data config path {data_config_path} does not exist"
        data_config = file_util.load_json(data_config_path)

        filepaths = []
        dataset_indices = []
        for idx, data in enumerate(data_config):
            filepaths.append(data['name'])
            dataset_indices.extend([idx] * data['weight'])
        filepaths = []
        dataset_indices = []
        for idx, data in enumerate(data_config):
            filepaths.append(data['name'])
            dataset_indices.extend([idx] * data['weight'])

        # Start data producer
        data_produce_config = {
            "datasets_per_batch": datasets_per_batch,
            "num_same_dataset": num_same_dataset,
            "batch_size": batch_size,
        }
        p = mp.Process(target=data_producer, args=(queue, filepaths, dataset_indices, data_produce_config))
        p.start()

        # Start training process
        self._fit(
            queue, pretrained_model=pretrained_model, steps=steps, max_length=max_length, scale=scale,
            save_steps=save_steps, model_save_path=model_save_path
        )

        # Terminate data procedure
        p.terminate()
        exit()

    @staticmethod
    def _fit(
        queue,
        pretrained_model='microsoft/MiniLM-L12-H384-uncased',
        steps: int = 20000,
        max_length: int = 128,
        scale: int = 20,
        save_steps: int = 10000,
        model_save_path: Text = "models",
    ):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        model = AutoModelForSentenceEmbedding(pretrained_model, tokenizer)

        model = model.to(device)

        # Instantiate optimizer
        optimizer = AdamW(params=model.parameters(), lr=2e-5, correct_bias=True)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=steps,
        )

        # Now we train the model
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        max_grad_norm = 1
        model.train()

        print(f"Starting training process")
        for global_step in tqdm(range(steps)):
            # Get the batch data
            while queue.empty():
                time.sleep(0.5)
                print(f"Waiting for data will put into queue")
            batch = queue.get()
            # For the dataset containing a tuple (anchor, positive)
            if len(batch[0]) == 2:
                text1 = tokenizer([b[0] for b in batch], return_tensors="pt",
                                  max_length=max_length, truncation=True, padding="max_length")
                text2 = tokenizer([b[1] for b in batch], return_tensors="pt",
                                  max_length=max_length, truncation=True, padding="max_length")

                # Computing the embeddings of two sentences
                embeddings_a = model(**text1.to(device))
                embeddings_b = model(**text2.to(device))

                # Compute similarity scores 512 x 512
                scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * scale

                # Computing cross-entropy loss
                labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)

                # Symmetric loss as in CLIP
                loss = (cross_entropy_loss(scores, labels) +
                        cross_entropy_loss(scores.transpose(0, 1), labels)) / 2

            # For the dataset containing a tuple (anchor, positive, negative)
            else:
                text1 = tokenizer([b[0] for b in batch], return_tensors="pt",
                                  max_length=max_length, truncation=True, padding="max_length")
                text2 = tokenizer([b[1] for b in batch], return_tensors="pt",
                                  max_length=max_length, truncation=True, padding="max_length")
                text3 = tokenizer([b[2] for b in batch], return_tensors="pt",
                                  max_length=max_length, truncation=True, padding="max_length")
                embeddings_a = model(**text1.to(device))
                embeddings_b1 = model(**text2.to(device))
                embeddings_b2 = model(**text3.to(device))

                embeddings_b = torch.cat([embeddings_b1, embeddings_b2])

                # Compute similarity scores 512 x 1024
                scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * scale

                # Computing cross-entropy loss
                labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)

                # Computing One-way loss
                loss = cross_entropy_loss(scores, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Save model
            if (global_step + 1) % save_steps == 0:
                output_path = os.path.join(model_save_path, str(global_step + 1))
                model.save_pretrained(output_path)
                print(f"Saved checkpoint at step: {global_step} at {output_path}")

        print(f"Save final checkpoint at: {model_save_path}/final")
        output_path = os.path.join(model_save_path, "final")
        model.save_pretrained(output_path)


class SentenceEmbedder(BaseEmbedder):

    def __init__(self, model_name_or_path: Text = "microsoft/MiniLM-L12-H384-uncased"):
        super(SentenceEmbedder, self).__init__()
        self.model_name_or_path: Text = model_name_or_path
        self._model: CustomSentenceTransformer = None

    @property
    def model(self):
        if not self._model:
            self._model = CustomSentenceTransformer.from_pretrained(self.model_name_or_path)
        return self._model

    def _train(
        self,
        triplets_data_path: List[Text],
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
        show_progress_bar: bool = True,
        checkpoint_path: Text = None,
        checkpoint_save_epoch: int = None,
        checkpoint_save_total_limit: int = None,
        resume_from_checkpoint: Text = None,
        **kwargs,
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
        # from venus.sentence_embedding.sentence_embedding import SentenceEmbedding
        # Producing data
        triplets_data = []
        if isinstance(triplets_data_path, str):
            triplets_data_path = [triplets_data_path]
        for path in triplets_data_path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"triplets data path {path} does not exist")
            triplets_data.extend(file_util.load_json(path)['data'])

        # sentence_embedding = CustomSentenceTransformer.from_pretrained(
        #     model_name_or_path=pretrained_model
        # )

        # sentence_embedding.model.max_seq_length = 128

        train_dataset = TripletsDataset(
            triplet_examples=triplets_data,
            query_key="query",
            pos_key="pos",
            neg_key="neg"
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        train_loss = losses.TripletLoss(model=self.model)

        # Train the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=None,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            evaluation_steps=evaluation_steps,
            use_amp=use_amp,
            optimizer_params={'lr': 2e-5},
            optimizer_class=transformers.AdamW,
            scheduler='WarmupLinear',
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            save_best_model=save_best_model,
            show_progress_bar=show_progress_bar,
            checkpoint_path=checkpoint_path,
            checkpoint_save_epoch=checkpoint_save_epoch,
            checkpoint_save_total_limit=checkpoint_save_total_limit,
            resume_from_checkpoint=resume_from_checkpoint
        )
        self.model.save(model_save_path)
