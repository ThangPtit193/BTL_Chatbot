from abc import abstractmethod
from typing import List, Dict, Union, Optional, Any

import logging
from pathlib import Path
from copy import deepcopy
from requests.exceptions import HTTPError

import numpy as np
from tqdm.auto import tqdm

import torch
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

from meteor.errors import MeteorError
from meteor.schema import Document
from meteor.document_stores import BaseDocumentStore
from meteor.nodes.retriever.base import BaseRetriever
from meteor.nodes.retriever._embedding_encoder import _EMBEDDING_ENCODERS
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
            model_version: Optional[str] = None,
            use_gpu: bool = True,
            batch_size: int = 32,
            max_seq_len: int = 512,
            model_format: Optional[str] = None,
            pooling_strategy: str = "reduce_mean",
            emb_extraction_layer: int = -1,
            top_k: int = 10,
            progress_bar: bool = True,
            devices: Optional[List[Union[str, torch.device]]] = None,
            use_auth_token: Optional[Union[str, bool]] = None,
            scale_score: bool = True,
            embed_meta_fields: List[str] = [],
            api_key: Optional[str] = None,
    ):
        """
        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param embedding_model: Local path or name of model in Hugging Face's model hub such
                                as ``'sentence-transformers/all-MiniLM-L6-v2'``. The embedding model could also
                                potentially be an OpenAI model ["ada", "babbage", "davinci", "curie"] or
                                a Cohere model ["small", "medium", "large"].
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param batch_size: Number of documents to encode at once.
        :param max_seq_len: Longest length of each document sequence. Maximum number of tokens for the document text. Longer ones will be cut down.
        :param model_format: Name of framework that was used for saving the model or model type. If no model_format is
                             provided, it will be inferred automatically from the model configuration files.
                             Options:

                             - ``'farm'`` (will use `_DefaultEmbeddingEncoder` as embedding encoder)
                             - ``'transformers'`` (will use `_DefaultEmbeddingEncoder` as embedding encoder)
                             - ``'sentence_transformers'`` (will use `_SentenceTransformersEmbeddingEncoder` as embedding encoder)
                             - ``'retribert'`` (will use `_RetribertEmbeddingEncoder` as embedding encoder)
                             - ``'openai'``: (will use `_OpenAIEmbeddingEncoder` as embedding encoder)
                             - ``'cohere'``: (will use `_CohereEmbeddingEncoder` as embedding encoder)
        :param pooling_strategy: Strategy for combining the embeddings from the model (for farm / transformers models only).
                                 Options:

                                 - ``'cls_token'`` (sentence vector)
                                 - ``'reduce_mean'`` (sentence vector)
                                 - ``'reduce_max'`` (sentence vector)
                                 - ``'per_token'`` (individual token vectors)
        :param emb_extraction_layer: Number of layer from which the embeddings shall be extracted (for farm / transformers models only).
                                     Default: -1 (very last layer).
        :param top_k: How many documents to return per query.
        :param progress_bar: If true displays progress bar during embedding.
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
                        Note: As multi-GPU training is currently not implemented for EmbeddingRetriever,
                        training will only use the first device provided in this list.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise, raw similarity scores (e.g. cosine or dot_product) will be used.
        :param embed_meta_fields: Concatenate the provided meta fields and text passage / table to a text pair that is
                                  then used to create the embedding.
                                  This approach is also used in the TableTextRetriever paper and is likely to improve
                                  performance if your titles contain meaningful information for retrieval
                                  (topic, entities etc.).
        :param api_key: The OpenAI API key or the Cohere API key. Required if one wants to use OpenAI/Cohere embeddings.
                        For more details see https://beta.openai.com/account/api-keys and https://dashboard.cohere.ai/api-keys

        """
        super().__init__()

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=True)

        if batch_size < len(self.devices):
            logger.warning("Batch size is less than the number of devices.All gpus will not be utilized.")

        self.document_store = document_store
        self.embedding_model = embedding_model
        self.model_version = model_version
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pooling_strategy = pooling_strategy
        self.emb_extraction_layer = emb_extraction_layer
        self.top_k = top_k
        self.progress_bar = progress_bar
        self.use_auth_token = use_auth_token
        self.scale_score = scale_score
        self.api_key = api_key
        self.model_format = (
            self._infer_model_format(model_name_or_path=embedding_model, use_auth_token=use_auth_token)
            if model_format is None
            else model_format
        )

        logger.info("Init retriever using embeddings of model %s", embedding_model)

        if self.model_format not in _EMBEDDING_ENCODERS.keys():
            raise ValueError(f"Unknown retriever embedding model format {model_format}")

        if (
                self.embedding_model.startswith("sentence-transformers")
                and model_format
                and model_format != "sentence_transformers"
        ):
            logger.warning(
                f"You seem to be using a Sentence Transformer embedding model but 'model_format' is set to '{self.model_format}'."
                f" You may need to set model_format='sentence_transformers' to ensure correct loading of model."
                f"As an alternative, you can let Haystack derive the format automatically by not setting the "
                f"'model_format' parameter at all."
            )

        self.embedding_encoder = _EMBEDDING_ENCODERS[self.model_format](retriever=self)
        self.embed_meta_fields = embed_meta_fields

    def retrieve(
            self,
            query: str,
            filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
            top_k: Optional[int] = None,
            index: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
            scale_score: Optional[bool] = None,
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
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                                           If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                                           Otherwise, raw similarity scores (e.g. cosine or dot_product) will be used.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
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
        if scale_score is None:
            scale_score = self.scale_score
        query_emb = self.embed_queries(queries=[query])
        documents = document_store.query_by_embedding(
            query_emb=query_emb[0], filters=filters, top_k=top_k, index=index, headers=headers, scale_score=scale_score
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
            headers: Optional[Dict[str, str]] = None,
            batch_size: Optional[int] = None,
            scale_score: Optional[bool] = None,
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
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true similarity scores (e.g. cosine or dot_product) which naturally have a different
                            value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
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
        if scale_score is None:
            scale_score = self.scale_score

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
                index=index,
                headers=headers,
                scale_score=scale_score,
            )
            documents.append(cur_docs)

        return documents

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        # for backward compatibility: cast pure str input
        if isinstance(queries, str):
            queries = [queries]
        assert isinstance(queries, list), "Expecting a list of texts, i.e. create_embeddings(texts=['text1',...])"
        return self.embedding_encoder.embed_queries(queries)

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param documents: List of documents to embed.
        :return: Embeddings, one per input document, shape: (docs, embedding_dim)
        """
        documents = self._preprocess_documents(documents)
        return self.embedding_encoder.embed_documents(documents)

    def _preprocess_documents(self, docs: List[Document]) -> List[Document]:
        """
        Turns table documents into text documents by representing the table in csv format.
        This allows us to use text embedding models for table retrieval.
        It also concatenates specified meta data fields with the text representations.

        :param docs: List of documents to linearize. If the document is not a table, it is returned as is.
        :return: List of documents with meta data + linearized tables or original documents if they are not tables.
        """
        linearized_docs = []
        for doc in docs:
            doc = deepcopy(doc)
            if doc.content_type == "table":
                if isinstance(doc.content, pd.DataFrame):
                    doc.content = doc.content.to_csv(index=False)
                else:
                    raise MeteorError("Documents of type 'table' need to have a pd.DataFrame as content field")
            meta_data_fields = [doc.meta[key] for key in self.embed_meta_fields if key in doc.meta and doc.meta[key]]
            doc.content = "\n".join(meta_data_fields + [doc.content])
            linearized_docs.append(doc)
        return linearized_docs

    @staticmethod
    def _infer_model_format(model_name_or_path: str, use_auth_token: Optional[Union[str, bool]]) -> str:
        if any(m in model_name_or_path for m in ["ada", "babbage", "davinci", "curie"]):
            return "openai"
        if model_name_or_path in ["small", "medium", "large"]:
            return "cohere"
        # Check if model name is a local directory with sentence transformers config file in it
        if Path(model_name_or_path).exists():
            if Path(f"{model_name_or_path}/config_sentence_transformers.json").exists():
                return "sentence_transformers"
        # Check if sentence transformers config file in model hub
        else:
            try:
                hf_hub_download(
                    repo_id=model_name_or_path,
                    filename="config_sentence_transformers.json",
                    use_auth_token=use_auth_token,
                )
                return "sentence_transformers"
            except HTTPError:
                pass

        # Check if retribert model
        config = AutoConfig.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)
        if config.model_type == "retribert":
            return "retribert"

        # Model is neither sentence-transformers nor retribert model -> use _DefaultEmbeddingEncoder
        return "farm"

    def train(
            self,
            training_data: List[Dict[str, Any]],
            learning_rate: float = 2e-5,
            n_epochs: int = 1,
            num_warmup_steps: Optional[int] = None,
            batch_size: int = 16,
            train_loss: str = "mnrl",
    ) -> None:
        """
        Trains/adapts the underlying embedding model.

        Each training data example is a dictionary with the following keys:

        * question: the question string
        * pos_doc: the positive document string
        * neg_doc: the negative document string
        * score: the score margin


        :param training_data: The training data
        :type training_data: List[Dict[str, Any]]
        :param learning_rate: The learning rate
        :type learning_rate: float
        :param n_epochs: The number of epochs
        :type n_epochs: int
        :param num_warmup_steps: The number of warmup steps
        :type num_warmup_steps: int
        :param batch_size: The batch size to use for the training, defaults to 16
        :type batch_size: int (optional)
        :param train_loss: The loss to use for training.
                           If you're using sentence-transformers as embedding_model (which are the only ones that currently support training),
                           possible values are 'mnrl' (Multiple Negatives Ranking Loss) or 'margin_mse' (MarginMSE).
        :type train_loss: str (optional)
        """
        self.embedding_encoder.train(
            training_data,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            num_warmup_steps=num_warmup_steps,
            batch_size=batch_size,
            train_loss=train_loss,
        )

    def save(self, save_dir: Union[Path, str]) -> None:
        """
        Save the model to the given directory

        :param save_dir: The directory where the model will be saved
        :type save_dir: Union[Path, str]
        """
        self.embedding_encoder.save(save_dir=save_dir)


class MultihopEmbeddingRetriever(EmbeddingRetriever):
    """
    Retriever that applies iterative retrieval using a shared encoder for query and passage.
    See original paper for more details:

    Xiong, Wenhan, et. al. (2020): "Answering complex open-domain questions with multi-hop dense retrieval"
    (https://arxiv.org/abs/2009.12756)
    """

    def __init__(
            self,
            embedding_model: str,
            document_store: Optional[BaseDocumentStore] = None,
            model_version: Optional[str] = None,
            num_iterations: int = 2,
            use_gpu: bool = True,
            batch_size: int = 32,
            max_seq_len: int = 512,
            model_format: str = "farm",
            pooling_strategy: str = "reduce_mean",
            emb_extraction_layer: int = -1,
            top_k: int = 10,
            progress_bar: bool = True,
            devices: Optional[List[Union[str, torch.device]]] = None,
            use_auth_token: Optional[Union[str, bool]] = None,
            scale_score: bool = True,
            embed_meta_fields: List[str] = [],
    ):
        """
        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param embedding_model: Local path or name of model in Hugging Face's model hub such as ``'sentence-transformers/all-MiniLM-L6-v2'``
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param num_iterations: The number of times passages are retrieved, i.e., the number of hops (Defaults to 2.)
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param batch_size: Number of documents to encode at once.
        :param max_seq_len: Longest length of each document sequence. Maximum number of tokens for the document text. Longer ones will be cut down.
        :param model_format: Name of framework that was used for saving the model or model type. If no model_format is
                             provided, it will be inferred automatically from the model configuration files.
                             Options:

                             - ``'farm'`` (will use `_DefaultEmbeddingEncoder` as embedding encoder)
                             - ``'transformers'`` (will use `_DefaultEmbeddingEncoder` as embedding encoder)
                             - ``'sentence_transformers'`` (will use `_SentenceTransformersEmbeddingEncoder` as embedding encoder)
                             - ``'retribert'`` (will use `_RetribertEmbeddingEncoder` as embedding encoder)
        :param pooling_strategy: Strategy for combining the embeddings from the model (for farm / transformers models only).
                                 Options:

                                 - ``'cls_token'`` (sentence vector)
                                 - ``'reduce_mean'`` (sentence vector)
                                 - ``'reduce_max'`` (sentence vector)
                                 - ``'per_token'`` (individual token vectors)
        :param emb_extraction_layer: Number of layer from which the embeddings shall be extracted (for farm / transformers models only).
                                     Default: -1 (very last layer).
        :param top_k: How many documents to return per query.
        :param progress_bar: If true displays progress bar during embedding.
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
                        Note: As multi-GPU training is currently not implemented for EmbeddingRetriever,
                        training will only use the first device provided in this list.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :param embed_meta_fields: Concatenate the provided meta fields and text passage / table to a text pair that is
                                  then used to create the embedding.
                                  This approach is also used in the TableTextRetriever paper and is likely to improve
                                  performance if your titles contain meaningful information for retrieval
                                  (topic, entities etc.).
        """
        super().__init__(
            embedding_model=embedding_model,
            document_store=document_store,
            model_version=model_version,
            use_gpu=use_gpu,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            model_format=model_format,
            pooling_strategy=pooling_strategy,
            emb_extraction_layer=emb_extraction_layer,
            top_k=top_k,
            progress_bar=progress_bar,
            devices=devices,
            use_auth_token=use_auth_token,
            scale_score=scale_score,
            embed_meta_fields=embed_meta_fields,
        )
        self.num_iterations = num_iterations

    def _merge_query_and_context(self, query: str, context: List[Document], sep: str = " "):
        return sep.join([query] + [doc.content for doc in context])

    def retrieve(
            self,
            query: str,
            filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
            top_k: Optional[int] = None,
            index: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
            scale_score: Optional[bool] = None,
            document_store: Optional[BaseDocumentStore] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number of documents
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
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                                           If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                                           Otherwise, raw similarity scores (e.g. cosine or dot_product) will be used.
        :param document_store: the document store to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        return self.retrieve_batch(
            queries=[query],
            filters=[filters] if filters is not None else None,
            top_k=top_k,
            index=index,
            headers=headers,
            scale_score=scale_score,
            batch_size=1,
        )[0]

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
            headers: Optional[Dict[str, str]] = None,
            batch_size: Optional[int] = None,
            scale_score: Optional[bool] = None,
            document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number of documents
        that are most relevant to the supplied queries.

        If you supply a single query, a single list of Documents is returned. If you supply a list of queries, a list of
        lists of Documents (one per query) is returned.

        :param queries: Single query string or list of queries.
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
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true similarity scores (e.g. cosine or dot_product) which naturally have a different
                            value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise, raw similarity scores (e.g. cosine or dot_product) will be used.
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
        if scale_score is None:
            scale_score = self.scale_score

        documents = []
        batches = self._get_batches(queries=queries, batch_size=batch_size)
        # TODO: Currently filters are applied both for final and context documents.
        # maybe they should only apply for final docs? or make it configurable with a param?
        pb = tqdm(total=len(queries), disable=not self.progress_bar, desc="Querying")
        for batch, cur_filters in zip(batches, filters):
            context_docs: List[List[Document]] = [[] for _ in range(len(batch))]
            for it in range(self.num_iterations):
                texts = [self._merge_query_and_context(q, c) for q, c in zip(batch, context_docs)]
                query_embs = self.embed_queries(texts)
                for idx, emb in enumerate(query_embs):
                    cur_docs = document_store.query_by_embedding(
                        query_emb=emb,
                        top_k=top_k,
                        filters=cur_filters,
                        index=index,
                        headers=headers,
                        scale_score=scale_score,
                    )
                    if it < self.num_iterations - 1:
                        # add doc with the highest score to context
                        if len(cur_docs) > 0:
                            context_docs[idx].append(cur_docs[0])
                    else:
                        # documents in the last iteration are final results
                        documents.append(cur_docs)
            pb.update(len(batch))
        pb.close()

        return documents
