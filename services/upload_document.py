import ast
import json
import logging
from typing import List

from iteration_utilities import unique_everseen
from fastapi import HTTPException, status
from loguru import logger

from venus.document_store.elasticsearch_store import ElasticsearchDocumentStore

from config import DEFAULT_ENCODING
from utils.helper import validate_json
from utils.io import deep_container_fingerprint
from utils.handler import Handler

CONTENT_TYPE = "application/json"

es = ElasticsearchDocumentStore(update_existing_documents=True)
handler = Handler()


async def upload_document(files):
    allowed_documents = {}
    doc_counter = 0

    for file in files:
        bytes_str = await file.read()
        if file.content_type == CONTENT_TYPE:
            contents = json.loads(ast.literal_eval(str(bytes_str)).decode(DEFAULT_ENCODING))
            if validate_json(contents):
                for content in contents:
                    index = content["meta"]["index"]
                    id_hash_key = deep_container_fingerprint(f"{content['text']}_{index}")
                    content["id"] = id_hash_key
                    if index not in allowed_documents.keys():
                        allowed_documents[index] = list()
                    allowed_documents[index].append(content)

                    # Drop duplicates documents based on the same hash ID
                    allowed_documents[index] = list(unique_everseen(allowed_documents[index]))

    # register Document objects to document store

    if not allowed_documents:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="There was an error to upload file. Qualified document is empty"
        )

    for index, documents in allowed_documents.items():
        logger.info(f"Load {index} from data")
        index_live = Handler.is_index_available(index=index)

        if not index_live:
            es.write_documents(index=index, documents=documents)
            doc_counter += len(documents)
        else:
            documents_by_index = es.get_all_documents(index=index)
            if not documents_by_index:
                es.write_documents(index=index, documents=documents)
                doc_counter += len(documents)
            else:
                unique_ids: List[str] = handler.handle_duplicate_documents(index=index, documents=documents)
                _documents: List[dict] = [document for document in documents if document["id"] not in unique_ids]
                if _documents:
                    es.write_documents(index=index, documents=_documents)
                    doc_counter += len(_documents)
                    logger.info(f'{len(_documents)} will be registered to DocumentStore')

    return HTTPException(
        status_code=status.HTTP_201_CREATED,
        detail=f'{doc_counter} document(s) were registered to DocumentStore'
    )
