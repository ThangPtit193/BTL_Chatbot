import copy
import csv
import json
import os
import pathlib
import sys
from itertools import chain
from pathlib import Path
from typing import Text, List, Union, Any

import yaml
from loguru import logger
from venus.document_store import ElasticsearchDocumentStore, InMemoryDocumentStore
from venus.retriever import EmbeddingRetriever
from yaml.scanner import ScannerError

from rest_api.constant import (
    DEFAULT_DOCSTORE_TYPES,
    DEFAULT_RETRIEVER_TYPES,
    DOC_STORE_ELASTICSEARCH,
    RETRIEVER_EMBEDDING,
    RETRIEVER_ELASTICSEARCH,
    DOC_STORE_INMEMORY,
    DEFAULT_FILE_TYPES
)
from rest_api.pipelines.venus_pipeline import VenusPipeline
from utils.helper import validate_document
from utils.io import deep_container_fingerprint
from utils.decorator import measure_memory, timeit

from fastapi import HTTPException, status


class VenusServices:
    def __init__(
            self,
            document_supported_types: List[Text] = DEFAULT_DOCSTORE_TYPES,
            retriever_supported_types: List[Text] = DEFAULT_RETRIEVER_TYPES,
            document_store_type: Text = DOC_STORE_INMEMORY,
            retriever_type: Text = RETRIEVER_ELASTICSEARCH,
            retriever_pretrained: Text = "va-base-distilbert-multilingual-faq-v0.1.0",
            similarity: str = "cosine",
            return_embedding: bool = False,
            top_k: int = 2,
            embedding_dim: int = 768,
            **kwargs
    ):

        self.embedding_dim = embedding_dim
        self.retriever_pretrained = retriever_pretrained
        self.document_supported_types = document_supported_types
        self.retriever_supported_types = retriever_supported_types
        # self.document_store_type = document_store_type
        assert document_store_type in self.document_supported_types, \
            f"Files of type {document_store_type} are not supported" \
            f"The supported types are: {self.document_supported_types}"

        assert retriever_type in self.retriever_supported_types, \
            f"The {retriever_type} types is not supported. {self.retriever_supported_types} are supported"

        if retriever_type == RETRIEVER_ELASTICSEARCH:
            assert document_store_type is DOC_STORE_ELASTICSEARCH, \
                f"Retriever is elasticsearch so elasticsearch document store is required"

        if retriever_type == RETRIEVER_EMBEDDING:
            assert retriever_pretrained is not None, f"`retriever_pretrained` must be provided to embed."

        self.retriever = None
        self.similarity = similarity
        self.return_embedding = return_embedding
        self.document_store = None
        self.index = kwargs["index"] if "index" in kwargs else "document"

        if document_store_type == DOC_STORE_INMEMORY:
            self.document_store = InMemoryDocumentStore(index=self.index, return_embedding=self.return_embedding)
        elif document_store_type == DOC_STORE_ELASTICSEARCH:
            # assert "host" in kwargs, f"host must be provided for {document_store_type}"
            # assert "port" in kwargs, f"port must be provided for {document_store_type}"
            host = kwargs["host"] if "host" in kwargs else "localhost"
            port = kwargs["port"] if "port" in kwargs else 9200

            self.document_store = ElasticsearchDocumentStore(
                index=self.index,
                similarity=self.similarity,
                return_embedding=self.return_embedding,
                embedding_dim=self.embedding_dim,
                host=host,
                port=port
            )

        # if retriever_type == RETRIEVER_EMBEDDING:
        self.retriever = EmbeddingRetriever(
            document_store=self.document_store,
            top_k=top_k,
            model_name_or_path=self.retriever_pretrained,
            download_pretrained=True
        )
        # elif retriever_type == RETRIEVER_ELASTICSEARCH:
        #     assert document_store_type == "elasticsearch", \
        #         f"Elasticsearch document store is required for elasticsearch retriever"
        #     self.retriever = ElasticsearchRetriever(self.document_store, top_k)

        # self.document_store = document_store
        # self.retriever_pretrained = retriever_pretrained
        self.retriever_type = retriever_type
        self.top_k = top_k

        try:
            if kwargs['config_path'] is not None:
                self.load_config_params(kwargs['config_path'])
        except:
            logger.warning(f'config_path is not found. Initialising with default parameters')

        self.pipeline = VenusPipeline(retriever=self.retriever)

    @classmethod
    def load_config_params(cls, config_path):
        """
        Parses YAML files into Python objects.
        Fails if the file does not exist.
        """
        try:
            logger.info(f"Load config file: {config_path}")
            with open(os.path.abspath(config_path), 'r') as stream:
                try:
                    config = yaml.safe_load(stream)
                    config = dict(zip(config, config['upload_services']))['upload_services']
                    return cls(
                        retriever_type=config['retriever_type'],
                        document_store_type=config['document_store_type'],
                        similarity=config['similarity']
                    )
                except ScannerError as e:
                    logger.error(
                        'Error parsing yaml of configuration file '
                        '{}: {}'.format(
                            e.problem_mark,
                            e.problem,
                        )
                    )
                    sys.exit(1)
        except FileNotFoundError:
            logger.error(
                'Error opening configuration file {}'.format(config_path)
            )
            sys.exit(1)

    def _store_retriever_model_with_index(self, index_attachment: List[Union[List, str]]):
        """
        Store a file that contains index with pretrained retriever model correspondingly.
        :param index_attachment: list of index with pretrained retriever models.
        :return:
        """
        header = ['index', 'pretrained_retriever_model']
        # check output log exists or not
        output_path = "./logs"
        index_path = os.path.join(output_path, "index_with_model_reference.csv")
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        if not os.path.exists(index_path):
            with open(index_path, "w", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(index_attachment)
                logger.info(f'store index with pretrained retriever model in `{index_path}`')
        else:
            with open(index_path, "r", encoding="utf-8") as buffer:
                stream = list(csv.reader(buffer))
                index_attachment_without_duplicate = []
                for item in index_attachment:
                    if item not in stream:
                        index_attachment_without_duplicate.append(item)
            if len(index_attachment_without_duplicate) > 0:
                with open(index_path, "a", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerows(index_attachment_without_duplicate)
                    logger.info(f'append new index into file at {index_path}')
            else:
                logger.info(f'No index to update into {self.document_store.__class__.__name__}')

    @timeit
    def run(self, file_paths):
        try:
            index_attachment = []
            documents = self.group_index(file_paths=file_paths)
            if documents is None:
                return HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No qualified document found. Please check input documents"
                )
            if documents is not None:
                for index_master in documents.keys():
                    for doc in documents[index_master]:
                        new_doc = copy.deepcopy(doc)
                        index = str(new_doc["meta"]["index"])
                        if self._check_duplicate_document(index=index, query=new_doc['text']):
                            self.pipeline.upload(
                                documents=[new_doc],
                                document_store=self.document_store,
                                index=index
                            )
                        else:
                            logger.log(
                                f"{new_doc['text']} is available in {self.document_store.__class__.__name__} Document Store")
                        logger.info(f'Your documents has been uploaded to {self.document_store.__class__.__name__}')

            return HTTPException(
                status_code=status.HTTP_200_OK,
                detail="Documents uploaded successfully"
            )
        except:
            return
            # return HTTPException(
            #     status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            #     detail="Documents has been uploaded unsuccessfully"
            # )

    def _check_duplicate_document(self, index: str, query: str) -> bool:
        """
        check if new document exists in document store or not
        :param index: index of document you want to update into document store
        :param query: text in document
        :return: true if new document has been stored and false if not found
        """
        # load all documents in document store
        try:
            _documents = self.document_store.get_all_documents(index=index)
            if _documents is None:
                return False
            queries_hashed = [deep_container_fingerprint(doc.text) for doc in _documents]
            if deep_container_fingerprint(query) not in queries_hashed:
                return True
            return False
        except:
            logger.info(f"{index} not found in Document Store")
            return True

    @staticmethod
    def is_index_available(index):
        pass

    def group_index(self, file_paths: Union[
        Path, List[Path], str, List[str], List[Union[Path, str]]]):
        """
        This function is to group data by index for model selection
        :param file_paths: path to route
        :return: list of data were grouped
        """
        grouped_documents = {}
        try:
            documents = self._load_data_from_files(file_paths=file_paths)
            if len(documents) == 0 or not isinstance(documents, list):
                logger.warning("Document not found")
                return
            for doc in documents:
                # if validate_document(data=doc):
                index_master = str(doc.get("meta").get("index")).split('_')[0]
                if index_master not in grouped_documents.keys():
                    grouped_documents[index_master] = list()
                grouped_documents[index_master].append(doc)
                # else:
                #     continue
            return grouped_documents
        except:
            logger.error(
                f'Some issues should be removed in group documents. {file_paths} must be a list of file paths.')
            return ""

    def _load_data_from_files(self, file_paths: Union[
        Path, List[Path], str, List[str], List[Union[Path, str]]]) -> List[Union[dict, Any]]:  # type: ignore
        """
        Send out files to temporary store for grouping
        :param file_paths: path to route
        :return: list of documents
        """
        file_data: list = []
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        for path in file_paths:
            if self._estimate_extension(Path(path)) in DEFAULT_FILE_TYPES:
                try:
                    with open(path, "r") as buffer:
                        data = json.load(buffer)
                        if validate_document(data=data):
                            file_data.append(data)
                        else:
                            continue
                except:
                    logger.error(f"Unexpected lines found in {str(path).split('/')[-1]} document")
                    pass

        if len(file_data) > 0:
            return list(chain.from_iterable(file_data))
        else:
            return []

    def _estimate_extension(self, file_path: Path) -> str:
        """
        Return the extension found based on the contents of the given file

        :param file_path: the path to extract the extension from
        """
        try:
            extension = pathlib.Path(file_path).suffix.lower()
            return extension.lstrip('.')
        except NameError as ne:
            logger.error(
                f"The type of '{file_path}' could not be guessed, probably because 'python-magic' is not installed. The supported types are {DEFAULT_FILE_TYPES}"
            )
            return ""

    @timeit
    def get_all_indices(self):  # type: ignore
        all_indices = self.document_store.get_all_indices()
        try:
            if len(all_indices) > 0:
                logger.info(f"Load all indices from {self.document_store.__class__.__name__}")
                return all_indices
        except:
            logger.info(f"No indices found in {self.document_store.__class__.__name__}")
            return []

    @timeit
    def delete_all_documents(self, index: str):
        self.document_store.delete_all_documents(index=index)
        return {"message": f'Document with {index} deleted'}

    def delete_index(self, index: str):
        self.document_store.delete_index(index=index)
        return HTTPException(
            status_code=status.HTTP_200_OK,
            detail="Successful"
        )

    @timeit
    @measure_memory
    def get_all_documents_by_index(self, index):
        return self.document_store.get_all_documents(index=index)

    # def __call__(self, *args, **kwargs):
    #     for key, value in kwargs.items():
    #         super(VenusServices, self).__setattr__(self, key, value)
    #     instance = VenusServices()
    #     return instance

    @classmethod
    def init_instance(cls, document_store_type: str):
        assert document_store_type is not None, f"{cls.__name__} got an unexpected keyword argument 'document_store_type'"
        return cls(document_store_type=document_store_type)

    @timeit
    @measure_memory
    def search(self, query, top_k, **kwargs):
        return self.pipeline.run(query=query, top_k_retriever=top_k)


