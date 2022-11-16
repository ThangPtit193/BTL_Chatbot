import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Any, Callable, Dict, List, Union

import numpy as np
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from meteor.document_stores.base import BaseDocumentStore

from meteor.nodes.retriever._losses import _TRAINING_LOSSES
from meteor.schema import Document

if TYPE_CHECKING:
    from meteor.nodes.retriever import EmbeddingRetriever

logger = logging.getLogger(__name__)


class _BaseEmbeddingEncoder:
    @abstractmethod
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        pass

    @abstractmethod
    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param docs: List of documents to embed.
        :return: Embeddings, one per input document, shape: (documents, embedding_dim)
        """
        pass

    def train(
            self,
            training_data: List[Dict[str, Any]],
            learning_rate: float = 2e-5,
            n_epochs: int = 1,
            num_warmup_steps: Optional[int] = None,
            batch_size: int = 16,
    ):
        """
        Trains or adapts the underlying embedding model.

        Each training data example is a dictionary with the following keys:

        * question: The question string.
        * pos_doc: Positive document string (the document containing the answer).
        * neg_doc: Negative document string (the document that doesn't contain the answer).
        * score: The score margin the answer must fall within.


        :param training_data: The training data in a dictionary format. Required.
        :type training_data: List[Dict[str, Any]]
        :param learning_rate: The speed at which the model learns. Required. We recommend that you leave the default `2e-5` value.
        :type learning_rate: float
        :param n_epochs: The number of epochs (complete passes of the training data through the algorithm) that you want the model to go through. Required.
        :type n_epochs: int
        :param num_warmup_steps: The number of warmup steps for the model. Warmup steps are epochs when the learning rate is very low. You can use them at the beginning of the training to prevent early overfitting of your model. Required.
        :type num_warmup_steps: int
        :param batch_size: The batch size to use for the training. Optional. The default values is 16.
        :type batch_size: int (optional)
        """
        pass

    def save(self, save_dir: Union[Path, str]):
        """
        Save the model to the directory you specify.

        :param save_dir: The directory where the model is saved. Required.
        :type save_dir: Union[Path, str]
        """
        pass

    def _check_docstore_similarity_function(self, document_store: BaseDocumentStore, model_name: str):
        """
        Check that document_store uses a similarity function
        compatible with the embedding model
        """
        if "sentence-transformers" in model_name.lower():
            model_similarity = None
            if "-cos-" in model_name.lower():
                model_similarity = "cosine"
            elif "-dot-" in model_name.lower():
                model_similarity = "dot_product"

            if model_similarity is not None and document_store.similarity != model_similarity:
                logger.warning(
                    f"You seem to be using {model_name} model with the {document_store.similarity} function instead of the recommended {model_similarity}. "
                    f"This can be set when initializing the DocumentStore"
                )
        elif "dpr" in model_name.lower() and document_store.similarity != "dot_product":
            logger.warning(
                f"You seem to be using a DPR model with the {document_store.similarity} function. "
                f"We recommend using dot_product instead. "
                f"This can be set when initializing the DocumentStore"
            )


class _SentenceTransformersEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "EmbeddingRetriever"):
        # pretrained embedding models coming from: https://github.com/UKPLab/sentence-transformers#pretrained-models
        # e.g. 'roberta-base-nli-stsb-mean-tokens'
        try:
            from sentence_transformers import SentenceTransformer
        except (ImportError, ModuleNotFoundError) as ie:
            from meteor.utils.import_utils import _optional_component_not_installed

            _optional_component_not_installed(__name__, "sentence", ie)

        self.embedding_model = SentenceTransformer(retriever.embedding_model, device=str(retriever.devices[0]))
        self.batch_size = retriever.batch_size
        self.embedding_model.max_seq_length = retriever.max_seq_len
        self.show_progress_bar = retriever.progress_bar
        if retriever.document_store:
            self._check_docstore_similarity_function(
                document_store=retriever.document_store, model_name=retriever.embedding_model
            )

    def embed(self, texts: Union[List[str], str]) -> np.ndarray:
        # texts can be a list of strings
        # get back list of numpy embedding vectors
        emb = self.embedding_model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        return emb

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        return self.embed(queries)

    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param docs: List of documents to embed.
        :return: Embeddings, one per input document, shape: (documents, embedding_dim)
        """
        passages = [d.content for d in docs]
        return self.embed(passages)

    def train(
            self,
            training_data: List[Dict[str, Any]],
            learning_rate: float = 2e-5,
            n_epochs: int = 1,
            num_warmup_steps: Optional[int] = None,
            batch_size: int = 16,
            train_loss: str = "mnrl",
    ):

        if train_loss not in _TRAINING_LOSSES:
            raise ValueError(f"Unrecognized train_loss {train_loss}. Should be one of: {_TRAINING_LOSSES.keys()}")

        st_loss = _TRAINING_LOSSES[train_loss]

        train_examples = []
        for train_i in training_data:
            missing_attrs = st_loss.required_attrs.difference(set(train_i.keys()))
            if len(missing_attrs) > 0:
                raise ValueError(
                    f"Some training examples don't contain the fields {missing_attrs} which are necessary when using the '{train_loss}' loss."
                )

            texts = [train_i["question"], train_i["pos_doc"]]
            if "neg_doc" in train_i:
                texts.append(train_i["neg_doc"])

            if "score" in train_i:
                train_examples.append(InputExample(texts=texts, label=train_i["score"]))
            else:
                train_examples.append(InputExample(texts=texts))

        logger.info("Training/adapting %s with %s examples", self.embedding_model, len(train_examples))
        train_dataloader = DataLoader(train_examples, batch_size=batch_size, drop_last=True, shuffle=True)
        train_loss = st_loss.loss(self.embedding_model)

        # Tune the model
        self.embedding_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=n_epochs,
            optimizer_params={"lr": learning_rate},
            warmup_steps=int(len(train_dataloader) * 0.1) if num_warmup_steps is None else num_warmup_steps,
        )

    def save(self, save_dir: Union[Path, str]):
        self.embedding_model.save(path=str(save_dir))


_EMBEDDING_ENCODERS: Dict[str, Callable] = {
    "sentence_transformers": _SentenceTransformersEmbeddingEncoder,
}
