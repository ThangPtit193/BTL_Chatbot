import os
import math
import shutil
import transformers
from typing import *
from torch import Tensor
from numpy import ndarray
import pandas as pd
from sentence_transformers.evaluation import (
    SentenceEvaluator
)
from sentence_transformers import SentenceTransformer, evaluation, losses

from torch.optim import Optimizer
from torch.utils.data import DataLoader

from venus.utils import logger
from venus.modeling import predict_heads as head
from venus.sentence_embedding.constants import TASKS
from venus.utils.config_parser import ConfigParser
from venus.utils import constants, utils
from venus.wrapper import axiom_wrapper

_logger = logger.get_logger(__name__)


class SentenceEmbedding(object):
    def __init__(self, model: Optional[SentenceTransformer], max_seq_length=None):
        super(SentenceEmbedding, self).__init__()
        self.model = model
        self.tokenizer = None
        self.predict_head: Optional[head.PredictionHead] = None
        self.model.max_seq_length = max_seq_length
        # self.data_processor: Optional[DataProcessor] = None
        self.word_embedding_model = None
        self.data_reader = None
        self.loss_func = None
        self.evaluator = None
        self.word_embedding_layer_dims = None

    @classmethod
    def from_pretrained(cls, model_name_or_path: Text = None):
        """

        Args:
            model_name_or_path: The model name or path. If name provided,
                                It will be loaded from hugging face hub or from axiom
        """
        model = None
        if os.path.isdir(model_name_or_path):
            try:
                model = SentenceTransformer(model_name_or_path)
            except Exception as e:
                _logger.error(f"Cannot load pretrained model from {model_name_or_path}, "
                              f"Because {e}")
        else:
            model_name_or_path = axiom_wrapper.fetch_model(model_name_or_path)
            model = SentenceTransformer(model_name_or_path)
        return cls(model)

    @staticmethod
    def get_model_embedding_dim(model_name_or_path: Text = None):
        """

        Args:
            model_name_or_path: The model name or path. If name provided,
                                It will be loaded from hugging face hub or from axiom
        """
        model = None
        if os.path.isdir(model_name_or_path):
            try:
                model = SentenceTransformer(model_name_or_path)
            except Exception as e:
                _logger.error(f"Cannot load pretrained model from {model_name_or_path}, "
                              f"Because {e}")
        else:
            model_name_or_path = axiom_wrapper.fetch_model(model_name_or_path)
            model = SentenceTransformer(model_name_or_path)
        return model.get_sentence_embedding_dimension()

    @classmethod
    def from_configure(cls, configure: Union[Dict, Text]):
        """
        Load sentence embedding from configure file
        Args:
            configure: a dictionary configure for sentence embedding

        Returns:

        """
        config_parser = ConfigParser(configure)

        if config_parser.check_module_exist(constants.MODULE_SENTENCE_TRANSFORMER):
            model = config_parser.get_object(constants.MODULE_SENTENCE_TRANSFORMER)
        else:
            # Initialize sentence transformer
            language_model = config_parser.get_object(constants.MODULE_LANGUAGE_MODEL)

            # Initialize predict head
            predict_head = config_parser.get_object(
                constants.MODULE_PREDICT_HEAD,
                word_embedding_dimension=language_model.get_word_embedding_dimension()
            )

            # Initialize sentence transformer
            model = SentenceTransformer(modules=[language_model, predict_head])
        sentence_embedding = cls(model=model)

        # Initialize loss function
        loss_func = config_parser.get_object(
            constants.MODULE_LOSS,
            model=model,
            word_embedding_dimension=model.get_sentence_embedding_dimension()
        )
        setattr(sentence_embedding, "loss_func", loss_func)

        # initialize dataset
        data_reader = config_parser.get_object(constants.MODULE_DATASET)
        setattr(sentence_embedding, "data_reader", data_reader)

        # initializer evaluator
        dev_samples = data_reader.get_samples_by_name('dev')
        if dev_samples:
            evaluator_class, arguments = config_parser.get_module(constants.MODULE_EVALUATOR)
            evaluator = evaluator_class.from_input_examples(dev_samples, **arguments)
            setattr(sentence_embedding, "evaluator", evaluator)
        return sentence_embedding

    def train(
            self,
            batch_size: int = 8,
            epochs: int = 3,
            weight_decay: float = 0.01,
            scheduler: str = 'WarmupLinear',
            show_progress_bar: bool = True,
            evaluation_steps: int = 1000,
            save_best_model: bool = True,
            use_amp: bool = False,
            model_save_path: Text = "models",
            **kwargs):
        """

        Args:
            batch_size:
            epochs:
            weight_decay:
            scheduler:
            show_progress_bar:
            evaluation_steps:
            save_best_model:
            use_amp:
            model_save_path:
            **kwargs:

        Returns:

        """

        train_dataloader = self.data_reader.get_loader('train', shuffle=True, batch_size=batch_size)
        # 10% of train data for warm-up
        warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)
        _logger.info("Warmup-steps: {}".format(warmup_steps))

        # Train the model
        self.model.fit(train_objectives=[(train_dataloader, self.loss_func)],
                       evaluator=self.evaluator,
                       epochs=epochs,
                       evaluation_steps=evaluation_steps,
                       warmup_steps=warmup_steps,
                       save_best_model=save_best_model,
                       output_path=model_save_path,
                       scheduler=scheduler,
                       use_amp=use_amp,
                       show_progress_bar=show_progress_bar,
                       weight_decay=weight_decay,
                       **kwargs
                       )

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
        train_loss = losses.MultipleNegativesRankingLoss(model=self.model)
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

    def encode(self, sentences: Union[str, List[str], List[int]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:

        out_vectors = self.model.encode(
            sentences, batch_size, show_progress_bar, output_value, convert_to_numpy,
            convert_to_tensor, device, normalize_embeddings
        )
        return out_vectors

    def evaluation(self, evaluator: SentenceEvaluator, output_path: str = None):
        """

        Args:
            evaluator: the evaluator
            output_path: the evaluator can write the results to this path

        Returns:

        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self.model, output_path)

    def information_retrieval_evaluate(
            self,
            query_file,
            corpus_file,
            triplets_file,
            name='ir-eval',
            output_path='results',
            show_progress_bar=False
    ):
        dev_queries = utils.load_json(query_file)
        dev_corpus = utils.load_json(corpus_file)
        dev_triplets = pd.read_csv(triplets_file)
        dev_rel_docs = {}

        for _, row in dev_triplets.iterrows():
            qid = str(row['query_id'])
            pos_id = str(row['pos_id'])

            if qid not in dev_rel_docs:
                dev_rel_docs[qid] = set()

            dev_rel_docs[qid].add(pos_id)

        ir_evaluator = evaluation.InformationRetrievalEvaluator(
            dev_queries,
            dev_corpus,
            dev_rel_docs,
            name=name,
            show_progress_bar=show_progress_bar
        )
        self.evaluation(ir_evaluator, output_path=output_path)

    def save(self, model_directory):
        self.model.save(model_directory)

    @staticmethod
    def _check_loss_sanity(task, loss_params):
        """

        Args:
            task:
            loss_params:

        Returns:

        """
        assert task in TASKS, f"We didn't support {task} task, we only support {TASKS.keys()} for QA"
        class_name = loss_params.get('class')
        if class_name not in TASKS[task]['loss']:
            raise Exception(f"For {task} you should use {TASKS[task]['loss']} instead of "
                            f"{class_name}")

    @staticmethod
    def _check_head_sanity(config: Dict):
        """

        Args:
            task:
            head_config:

        Returns:

        """
        task = config.get("training", {}).get('task')
        predict_head = config.get('predict_head')
        class_name = predict_head.get('class')

        if class_name not in TASKS[task]['head']:
            raise Exception(f"For {task} you should use {TASKS[task]['head']} instead of "
                            f"{class_name}")
