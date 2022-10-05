from typing import Any, Dict

import os
import logging
from pathlib import Path

# Since each instance of FAISSDocumentStore creates an in-memory FAISS index, the Indexing & Query Pipelines would
# end up with different indices. The same applies for InMemoryDocumentStore.
from loguru import logger
from venus.document_store import InMemoryDocumentStore
from venus.pipelines import Pipeline
from venus.rest_api.controller.utils import RequestLimiter

UNSUPPORTED_DOC_STORES = InMemoryDocumentStore
indexing_pipeline = dict()


def setup_pipelines() -> Dict[str, Any]:
    # Re-import the configuration variables
    global indexing_pipeline
    from rest_api import config  # pylint: disable=reimported

    pipelines = {}

    # Load query pipeline
    query_pipeline = Pipeline.load_from_yaml(Path(config.PIPELINE_YAML_PATH), pipeline_name=config.QUERY_PIPELINE_NAME)
    logging.info(f"Loaded pipeline nodes: {query_pipeline.graph.nodes.keys()}")
    pipelines["query_pipeline"] = query_pipeline

    # Find document store
    document_store = query_pipeline.get_document_store()
    logging.info(f"Loaded doc_store: {document_store}")
    pipelines["document_store"] = document_store

    # Setup concurrency limiter
    concurrency_limiter = RequestLimiter(config.CONCURRENT_REQUEST_PER_WORKER)
    logging.info("Concurrent requests per worker: {CONCURRENT_REQUEST_PER_WORKER}")
    pipelines["concurrency_limiter"] = concurrency_limiter

    # Load indexing pipeline (if available)
    try:
        indexing_pipeline = Pipeline.load_from_yaml(
            Path(config.PIPELINE_YAML_PATH), pipeline_name=config.INDEXING_PIPELINE_NAME
        )
        doc_store = indexing_pipeline.get_document_store()
        if isinstance(doc_store, UNSUPPORTED_DOC_STORES):
            indexing_pipeline = None
            raise ValueError(
                "Indexing pipelines with InMemoryDocumentStore are not supported by the REST APIs."
            )

    except ValueError as e:
        indexing_pipeline = None
        logger.error(f"{e}\nFile Upload API will not be available.")

    finally:
        pipelines["indexing_pipeline"] = indexing_pipeline

    # Create directory for uploaded files
    # os.makedirs(config.FILE_UPLOAD_PATH, exist_ok=True)

    return pipelines


if __name__ == "__main__":
    setup_pipelines()
