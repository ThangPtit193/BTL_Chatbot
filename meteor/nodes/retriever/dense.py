import os
from abc import abstractmethod
from typing import List, Dict, Union, Optional, Text

import logging

from numpy import ndarray

import numpy as np
from torch import Tensor
from tqdm.auto import tqdm

import torch

from meteor.errors import MeteorError
from meteor.schema import Document
from meteor.document_stores import BaseDocumentStore
from meteor.document_stores import InMemoryDocumentStore
# from meteor.document_stores.memory import InMemoryDocumentStore
from meteor.nodes.retriever.base import BaseRetriever
from meteor.nodes.retriever.sentence_embedding import SentenceEmbedding
from meteor.modelling.utils import initialize_device_settings

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Base class for all dense retrievers.
    """

    @abstractmethod
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        pass

    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param documents: List of documents to embed.
        :return: Embeddings of documents, one per input document, shape: (documents, embedding_dim)
        """
        pass

    def run_indexing(self, documents: List[Document]):
        embeddings = self.embed_documents(documents)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        output = {"documents": documents}
        return output, "output_1"


class EmbeddingRetriever(DenseRetriever):
    def __init__(
            self,
            embedding_model: str,
            document_store: Optional[BaseDocumentStore] = None,
            use_gpu: bool = True,
            batch_size: int = 32,
            top_k: int = 10,
            progress_bar: bool = True,
            devices: Optional[List[Union[str, torch.device]]] = None
    ):
        """
        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param embedding_model: Local path or name of model in Hugging Face's model hub such
                                as ``'sentence-transformers/all-MiniLM-L6-v2'``. The embedding model could also fetch from
                                Venus hub to get ready to use
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param batch_size: Number of documents to encode at once.
        :param top_k: How many documents to return per query.
        :param progress_bar: If true displays progress bar during embedding.
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
                        Note: As multi-GPU training is currently not implemented for EmbeddingRetriever,
                        training will only use the first device provided in this list.
        """
        super().__init__()

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=True)

        if batch_size < len(self.devices):
            logger.warning("Batch size is less than the number of devices.All gpus will not be utilized.")

        self.document_store = document_store
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.top_k = top_k
        self.progress_bar = progress_bar
        self.embedding_model = embedding_model

        logger.info("Init retriever using embeddings of model %s", embedding_model)
        if embedding_model:
            self.embedding_encoder = SentenceEmbedding.from_pretrained(embedding_model)
        else:
            self.embedding_encoder = None

        if document_store and document_store.similarity != "cosine":
            logger.warning(
                f"You seem to be using a Sentence Transformer with the {document_store.similarity} function."
                f"We strongly recommend using cosine instead."
                f"This can be set when initializing the DocumentStore"
            )

        if len(self.devices) > 1:
            logger.info(f"You have {self.devices} devices available")

    def retrieve(
            self,
            query: str,
            filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
            top_k: Optional[int] = None,
            index: Optional[str] = None,
            document_store: Optional[BaseDocumentStore] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:
                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
            )
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = document_store.index
        query_emb = self.embed_queries(queries=[query])
        documents = document_store.query_by_embedding(
            query_emb=query_emb[0], filters=filters, top_k=top_k, index=index
        )
        return documents

    def retrieve_batch(
            self,
            queries: List[str],
            filters: Optional[
                Union[
                    Dict[str, Union[Dict, List, str, int, float, bool]],
                    List[Dict[str, Union[Dict, List, str, int, float, bool]]],
                ]
            ] = None,
            top_k: Optional[int] = None,
            index: Optional[str] = None,
            batch_size: Optional[int] = None,
            document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number of documents
        that are most relevant to the supplied queries.

        Returns a list of lists of Documents (one per query).

        :param queries: List of query strings.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions. Can be a single filter that will be applied to each query or a list of filters
                        (one filter per query).

                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:
                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param batch_size: Number of queries to embed at a time.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve_batch() method."
            )
        if top_k is None:
            top_k = self.top_k

        if batch_size is None:
            batch_size = self.batch_size

        if isinstance(filters, list):
            if len(filters) != len(queries):
                raise MeteorError(
                    "Number of filters does not match number of queries. Please provide as many filters"
                    " as queries or a single filter that will be applied to each query."
                )
        else:
            filters = [filters] * len(queries) if filters is not None else [{}] * len(queries)

        if index is None:
            index = document_store.index

        documents = []
        query_embs: List[np.ndarray] = []
        for batch in self._get_batches(queries=queries, batch_size=batch_size):
            query_embs.extend(self.embed_queries(queries=batch))
        for query_emb, cur_filters in tqdm(
                zip(query_embs, filters), total=len(query_embs), disable=not self.progress_bar, desc="Querying"
        ):
            cur_docs = document_store.query_by_embedding(
                query_emb=query_emb,
                top_k=top_k,
                filters=cur_filters,
                index=index
            )
            documents.append(cur_docs)

        return documents

    def embed(self, texts: Union[List[List[str]], List[str], str], batch_size=8) -> Union[
        List[Tensor], ndarray, Tensor]:
        """
        Create embeddings for each text in a list of texts using the retrievers model (`self.embedding_model`)

        :param texts: Texts to embed
        :param batch_size: Batch size to embed batch texts to avoid out of memory
        :return: List of embeddings (one per input text). Each embedding is a list of floats.
        """

        # for backward compatibility: cast pure str input
        if isinstance(texts, str):
            texts = [texts]
        assert isinstance(texts, list), \
            "Expecting a list of texts, i.e. create_embeddings(texts=['text1',...])"

        # texts can be a list of strings or a list of [title, text]
        # get back list of numpy embedding vectors
        emb = self.embedding_encoder.encode(texts, batch_size=batch_size, show_progress_bar=False)
        emb = [r for r in emb]
        return emb

    def embed_queries(self, queries: List[str], batch_size: int = 8) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        # for backward compatibility: cast pure str input
        return self.embed(queries, batch_size=batch_size)

    def embed_documents(self, documents: List[Document], batch_size: int = 8) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Create embeddings for a list of documents.

        :param documents: List of documents to embed.
        :return: Embeddings, one per input document, shape: (docs, embedding_dim)
        """
        if self.embedding_model:
            passages = [[d.meta["name"] if d.meta and "name" in d.meta else "", d.content] for d in
                        documents]
        else:
            passages = [d.content for d in documents]  # type: ignore
        return self.embed(passages, batch_size=batch_size)

    # def save(self, model_directory, download_pretrained=True):
    #     retriever_dir = os.path.join(model_directory, self.__class__.__name__)
    #     logger.info(f"Save model of {self.__class__.__name__} to {retriever_dir}")
    #
    #     if not os.path.isdir(retriever_dir):
    #         os.makedirs(retriever_dir)
    #
    #     if not download_pretrained:
    #         self.embedding_encoder.save(retriever_dir)
    #     if isinstance(self.document_store, InMemoryDocumentStore):
    #         self.document_store.save(retriever_dir)
    #
    # def load(self, model_directory: str, download_pretrained: bool = True):
    #     retriever_dir = os.path.join(model_directory, self.__class__.__name__)
    #     logger.info(f"Loaded model of {self.__class__.__name__} from {retriever_dir}")
    #     model_name = retriever_dir if download_pretrained else self.embedding_model
    #     if not self.embedding_model:
    #         self.embedding_encoder = SentenceEmbedding.from_pretrained(model_name)
    #
    #     if isinstance(self.document_store, InMemoryDocumentStore):
    #         self.document_store.load(retriever_dir)
    #
    # def update_embeddings(self, index: Text = None, batch_size=8):
    #     self.document_store.update_embeddings(self, index=index, batch_size=batch_size)
