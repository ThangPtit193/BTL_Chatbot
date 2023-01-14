import math
import os
import shutil
import time
from abc import abstractmethod
from typing import *

import torch
import torch.multiprocessing as mp
import transformers
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from comet.components.embeddings.embedding_models import BertEmbedder
from comet.lib import file_util, logger
from saturn.abstract_method.staturn_abstract import SaturnAbstract
from .dataset import data_producer
from .net import AutoModelForSentenceEmbedding, CustomSentenceTransformer

__all__ = []
_logger = logger.get_logger(__name__)


class SemanticSimilarity(SaturnAbstract):
    def __init__(self, config, pretrained_name_or_abspath, **kwargs):
        super(SemanticSimilarity, self).__init__(
            config=config, **kwargs
        )
        self.cache_path = ".embeddings_cache"
        self.pretrained_name_or_abspath = pretrained_name_or_abspath
        self._embedder: Optional[BertEmbedder] = None

    def initialize(self, **kwargs):
        for k, val in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, val)

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = BertEmbedder(
                pretrained_name_or_abspath=self.pretrained_name_or_abspath, device=self.device
            )
        return self._embedder

    def load_model(self, cache_path=None, pretrained_name_or_abspath=None, **kwargs):
        self._embedder = BertEmbedder(
            cache_path=cache_path, pretrained_name_or_abspath=pretrained_name_or_abspath, device=self.device
        )

    def train(self, trainer_config: Dict):
        self._train(**trainer_config)

    @abstractmethod
    def _train(self, *args, **kwargs):
        raise NotImplementedError


class NaiveSemanticSimilarity(SemanticSimilarity):

    # def load_model(self, cache_path=None, pretrained_name_or_abspath=None):
    #     super(BaseEmbedder, self).__init__(cache_path, pretrained_name_or_abspath)

    def __init__(self, config, pretrained_name_or_abspath, **kwargs):
        super(NaiveSemanticSimilarity, self).__init__(
            config=config, pretrained_name_or_abspath=pretrained_name_or_abspath, **kwargs
        )

    def initialize(self, **kwargs):
        for k, val in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, val)

    def _train(
        self,
        data_config_path: Text,
        datasets_per_batch: int = 2,
        num_same_dataset: int = 2,
        batch_size: int = 32,
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
            queue, pretrained_model=self.pretrained_name_or_abspath, steps=steps, max_length=max_length, scale=scale,
            save_steps=save_steps, model_save_path=model_save_path, **kwargs
        )

        # Terminate data procedure
        p.terminate()
        exit()

    def _fit(
        self,
        queue,
        pretrained_model='microsoft/MiniLM-L12-H384-uncased',
        steps: int = 20000,
        scale: int = 20,
        save_steps: int = 10000,
        model_save_path: Text = "models",
        checkpoint_path: Text = None,
        checkpoint_save_step: int = None,
        checkpoint_save_total_limit: int = None,
        resume_from_checkpoint: Text = None,
        max_length: int = 128,
        **kwargs
    ):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        model = AutoModelForSentenceEmbedding(pretrained_model, tokenizer)
        print(self.device)
        model = model.to(self.device)

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

        step_offset = None
        if resume_from_checkpoint:
            if not os.path.exists(resume_from_checkpoint):
                raise ValueError(f"Resume from checkpoint {resume_from_checkpoint} does not exist")
            checkpoint = torch.load(resume_from_checkpoint, map_location=self.device)
            step_offset = checkpoint['step']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Starting training process")
        for global_step in tqdm(range(steps)):
            if step_offset and global_step < step_offset:
                continue
            # Get the batch data
            while queue.empty():
                time.sleep(0.5)
                print(f"Waiting for data will put into queue")
            batch = queue.get()
            if len(batch[0]) == 2:
                text1 = tokenizer([b[0] for b in batch], return_tensors="pt",
                                  max_length=max_length, truncation=True, padding="max_length")
                text2 = tokenizer([b[1] for b in batch], return_tensors="pt",
                                  max_length=max_length, truncation=True, padding="max_length")

                # Computing the embeddings of two sentences
                embeddings_a = model(**text1.to(self.device))
                embeddings_b = model(**text2.to(self.device))

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
                embeddings_a = model(**text1.to(self.device))
                embeddings_b1 = model(**text2.to(self.device))
                embeddings_b2 = model(**text3.to(self.device))

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
            if bool(
                checkpoint_path is not None
                and checkpoint_save_step is not None
                and checkpoint_save_step > 0
                and (global_step + 1) % checkpoint_save_step == 0
            ):
                self.save_checkpoint(
                    model, global_step, optimizer, lr_scheduler, checkpoint_path, checkpoint_save_total_limit
                )
                print(f"Saved checkpoint at step: {global_step} at {checkpoint_path}")

            # Save model
            if (global_step + 1) % save_steps == 0:
                output_path = os.path.join(model_save_path, str(global_step + 1))
                model.save_pretrained(output_path)
                print(f"Saved model at step: {global_step} at {output_path}")

        print(f"Save final checkpoint at: {model_save_path}/final")
        output_path = os.path.join(model_save_path, "final")
        model.save_pretrained(output_path)

    @staticmethod
    def save_checkpoint(model, step, optimizer, scheduler, checkpoint_path, checkpoint_save_total_limit):
        _logger.info(f"Saving checkpoint for epoch {step}")
        model_checkpoint_path = os.path.join(checkpoint_path, "step-{}".format(step))
        if not os.path.exists(model_checkpoint_path):
            os.makedirs(model_checkpoint_path)

        model_checkpoint_file = os.path.join(model_checkpoint_path, "checkpoint.pt")

        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, model_checkpoint_file)
        # Delete old checkpoints
        if checkpoint_save_total_limit is not None and checkpoint_save_total_limit > 0:
            old_checkpoints = []
            for subdir in os.listdir(checkpoint_path):
                prefix, epoch = subdir.split("-")
                if epoch.isdigit():
                    old_checkpoints.append({'step': int(step), 'path': os.path.join(checkpoint_path, subdir)})

            if len(old_checkpoints) > checkpoint_save_total_limit:
                old_checkpoints = sorted(old_checkpoints, key=lambda x: x['step'])
                shutil.rmtree(old_checkpoints[0]['path'])


class SBertSemanticSimilarity(SemanticSimilarity):

    def __init__(self, pretrained_name_or_abspath, **kwargs):
        super(SBertSemanticSimilarity, self).__init__(
            pretrained_name_or_abspath=pretrained_name_or_abspath, **kwargs)
        self._learner: CustomSentenceTransformer = None

    @property
    def learner(self):
        if not self._learner:
            self._learner = CustomSentenceTransformer.from_pretrained(
                self.pretrained_name_or_abspath, device=self.device
            )
        return self._learner

    def _train(
        self,
        triplets_data_path: List[Text] = None,
        pretrained_name_or_abspath="bert-base-uncased",
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
        checkpoint_save_epoch: int = None,
        checkpoint_save_total_limit: int = None,
        resume_from_checkpoint: Text = None,
        save_by_epoch: int = 0,
        model_save_total_limit: int = None,
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
        # Producing data
        if not triplets_data_path:
            triplets_data_path = file_util.get_all_files_in_directory(
                self.get_data_dir(), extensions=[".json"]
            )
        triplets_data = []
        if isinstance(triplets_data_path, str):
            triplets_data_path = [triplets_data_path]
        for path in triplets_data_path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"triplets data path {path} does not exist")
            triplet_data = file_util.load_json(path)
            if 'data' not in triplet_data:
                _logger.warning(f"triplets data path '{path}' is not a valid triplets data")
                continue
            triplets_data.extend(triplet_data['data'])

        train_dataset = TripletsDataset(
            triplet_examples=triplets_data,
            query_key="query",
            pos_key="pos",
            neg_key="neg"
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        train_loss = losses.TripletLoss(model=self.learner)

        # Train the model
        self.learner.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=None,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=self.get_model_dir(),
            # evaluation_steps=evaluation_steps,
            use_amp=use_amp,
            optimizer_params={'lr': 2e-5},
            optimizer_class=transformers.AdamW,
            scheduler='WarmupLinear',
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            save_best_model=save_best_model,
            show_progress_bar=show_progress_bar,
            checkpoint_path=self.get_checkpoint_dir(),
            checkpoint_save_epoch=checkpoint_save_epoch,
            checkpoint_save_total_limit=checkpoint_save_total_limit,
            resume_from_checkpoint=resume_from_checkpoint,
            save_by_epoch=save_by_epoch,
            model_save_total_limit=model_save_total_limit,
        )
        # save_best_model_path = os.path.join(model_save_path,"final_model")
        # if not os.path.exists(save_best_model_path):
        #     os.makedirs(save_best_model_path)
        # self.learner.save(save_best_model_path)


class NLISemanticSimilarity(SemanticSimilarity):

    def __init__(self, pretrained_name_or_abspath, device=None, **kwargs):
        super(NLISemanticSimilarity, self).__init__(
            pretrained_name_or_abspath=pretrained_name_or_abspath, device=device, **kwargs)
        # self.model_name_or_path: Text = model_name_or_path
        self._learner: Optional[SentenceTransformer] = None
        self.max_seq_length = kwargs.get("max_seq_length", 128)

    @property
    def learner(self):
        if not self._learner:
            word_embedding_model = models.Transformer(self.pretrained_name_or_abspath,
                                                      max_seq_length=self.max_seq_length)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')

            self._learner = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)
        return self._learner

    def _train(
        self,
        data_path_or_dir: List[Text],
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
        save_by_epoch: int = 0,
        model_save_total_limit: int = None,
        # device_name: Union[str,int] = "cpu",
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
        # from venus.sentence_embedding.sentence_embedding import SentenceEmbedding
        # Producing data
        label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        train_samples = []
        if isinstance(data_path_or_dir, str):
            data_path_or_dir = [data_path_or_dir]
        for path in data_path_or_dir:
            if not os.path.exists(path):
                raise FileNotFoundError(f"triplets data path {path} does not exist")
            datas = file_util.load_json(path)['data']
            for data in datas:
                label_id = label2int[data['label']]
                train_samples.append(InputExample(texts=[data['sentence1'], data['sentence2']], label=label_id))
        _logger.info("Train samples: {}".format(len(train_samples)))

        # Special data loader that avoid duplicates within a batch
        train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=batch_size)

        # Our training loss
        train_loss = losses.MultipleNegativesRankingLoss(self.learner)
        # Configure the training
        warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)  # 10% of train data for warm-up
        _logger.info("Warmup-steps: {}".format(warmup_steps))

        # Train the model
        self.learner.fit(
            train_objectives=[(train_dataloader, train_loss)],
            # evaluator=dev_evaluator,
            epochs=epochs,
            evaluation_steps=int(len(train_dataloader) * 0.1),
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            use_amp=False,
            checkpoint_path=checkpoint_path,
            checkpoint_save_total_limit=checkpoint_save_total_limit
        )


class QuadrupletSemanticSimilarity(SemanticSimilarity):

    def __init__(self, pretrained_name_or_abspath, device=None, **kwargs):
        super(QuadrupletSemanticSimilarity, self).__init__(pretrained_name_or_abspath, device, **kwargs)
        self._learner: CustomSentenceTransformer = None

    @property
    def learner(self):
        if not self._learner:
            self._learner = CustomSentenceTransformer.from_pretrained(
                self.pretrained_name_or_abspath, device=self.device
            )
        return self._learner

    def _train(
        self,
        quadruplet_data_path: List[Text],
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
        save_by_epoch: int = 0,
        model_save_total_limit: int = None,
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
        :param use_amp:save_by_epoch
        :param save_best_model:
        :param show_progress_bar:
        :return:
        """
        from saturn.components.dataset_reader.quadruplet_dataset import QuadrupletDataset
        # quad loss
        from saturn.components.losses.quadruplet_loss import QuadrupletLoss
        # from venus.sentence_embedding.sentence_embedding import SentenceEmbedding
        # Producing data
        quadruplets_data = []
        if isinstance(quadruplet_data_path, str):
            quadruplet_data_path = [quadruplet_data_path]
        for path in quadruplet_data_path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"triplets data path {path} does not exist")
            quadruplets_data.extend(file_util.load_json(path)['data'])

        train_dataset = QuadrupletDataset(
            quadruplet_examples=quadruplets_data,
            query_key="query",
            pos_key="pos",
            neg_key_1="neg1",
            neg_key_2="neg2"
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        train_loss = QuadrupletLoss(model=self.learner, quadruple_margin_1=5, quadruple_margin_2=4)

        # Train the model
        self.learner.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=None,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            # evaluation_steps=evaluation_steps,
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
            resume_from_checkpoint=resume_from_checkpoint,
            save_by_epoch=save_by_epoch,
            model_save_total_limit=model_save_total_limit
        )
        # self.learner.save(model_save_path)
