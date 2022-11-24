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
        tmp_path=None,
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
            index=doc_index,
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
        # retriever.update_embeddings()
    else:
        raise Exception(f"No retriever fixture for '{retriever_type}'")

    return retriever


def load_config(config_filename, ci):
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


if __name__ == "__main__":
    print(load_config("/Users/phongnt/FTECH/knowledge-retrieval/meteor/nodes/evaluator/config.json", True))
    docs = prepare_docs("/Users/phongnt/FTECH/knowledge-retrieval/assets/corpus.json")
    # doc_store = get_document_store("memory", similarity="cosine")
    # retriever = get_retriever(retriever_name="embedding", doc_store=doc_store, model_name_or_path="ms-viquad-bi-encoder-phobert-base")
    # index_to_doc_store(doc_store=doc_store, docs=docs, retriever=retriever)
    # print(doc_store.get_all_documents())
