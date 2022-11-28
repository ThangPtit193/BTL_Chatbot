import json
import os
import time
from pathlib import Path
from typing import Union, List, Dict, Any

from loguru import logger

from meteor import Document
from meteor.document_stores import InMemoryDocumentStore, BaseDocumentStore
from meteor.nodes import TfidfRetriever, EmbeddingRetriever
from meteor.utils.io import http_get, load_json

doc_index = "eval_document"
label_index = "label"


def get_document_store(
        document_store_type,
        embedding_dim=768,
        embedding_field="embedding",
        index="meteor_test",
        similarity: str = "cosine",
        recreate_index: bool = True,
):  # cosine is default similarity as dot product is not supported by Weaviate
    document_store: BaseDocumentStore
    if document_store_type == "memory":
        document_store = InMemoryDocumentStore(
            return_embedding=True,
            embedding_dim=embedding_dim,
            embedding_field=embedding_field,
            index=index,
            similarity=similarity,
        )
    else:
        raise Exception(f"No document store fixture for '{document_store_type}'")
    return document_store


def get_retriever(retriever_type, document_store, model_name_or_path):
    if retriever_type == "tfidf":
        retriever = TfidfRetriever(document_store=document_store)
    elif retriever_type == "embedding":
        retriever = EmbeddingRetriever(
            document_store=document_store, embedding_model=model_name_or_path, use_gpu=False
        )
    else:
        raise Exception(f"No retriever fixture for '{retriever_type}'")

    return retriever


def load_from_config(config_filename, ci):
    conf = json.load(open(config_filename))
    if ci:
        params = conf["params"]["ci"]
    else:
        params = conf["params"]["full"]
    bucket = conf["data_bucket"]
    option = bucket["option"]
    if option == "local":
        conf_bucket = bucket["local"]
    else:
        conf_bucket = bucket["axiom"]

    if len(conf["retriever_models"]) > 0:
        retriever_models = conf["retriever_models"]
    else:
        raise ValueError(f"Retriever_models includes at least one model got 0 instead")

    return params, conf_bucket, retriever_models


def download_from_url(url: str, filepath: Union[str, Path]):
    """
    Download from an url to a local file. Skip already existing files.

    :param url: Url
    :param filepath: local path where the url content shall be stored
    :return: local path of the downloaded file
    """

    logger.info("Downloading %s", url)
    # Create local folder
    folder, filename = os.path.split(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Download file if not present locally
    if os.path.exists(filepath):
        logger.info("Skipping %s (exists locally)", url)
    else:
        logger.info("Downloading %s to %s", filepath)
        with open(filepath, "wb") as file:
            http_get(url=url, path=file)
    return filepath


def prepare_docs(corpus_dir) -> List[Union[Document, Dict[str, Any]]]:
    docs: list = []
    if os.path.exists(corpus_dir):
        dataset = load_json(corpus_dir)
    for key in dataset.keys():
        dataset[key] = [{"content": content} for content in dataset[key]]
        docs.extend(dataset[key])
    return docs
