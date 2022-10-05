import ast
import json
from typing import List
from loguru import logger

from iteration_utilities import unique_everseen
from fastapi import HTTPException, status

from venus.document_store.elasticsearch_store import ElasticsearchDocumentStore
from venus.pipelines.pipeline import Pipeline
from venus.retriever import EmbeddingRetriever
from venus.sentence_embedding import SentenceEmbedding
from venus.document_store.in_memory_store import InMemoryDocumentStore

from config import DEFAULT_ENCODING
from utils.io import deep_container_fingerprint
from utils.handler import Handler

CONTENT_TYPE = "application/json"

handler = Handler()


async def upload_document(ds, retriever, files):
    allowed_documents = {}
    doc_counter = 0
    if ds == "ElasticsearchStore":
        document_store = ElasticsearchDocumentStore()
    else:
        document_store = InMemoryDocumentStore()

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
            document_store.write_documents(index=index, documents=documents)
            retriever = EmbeddingRetriever(document_store=document_store, model_name_or_path="ftech-bert-base")
            retriever.update_embeddings(index=index)
            doc_counter += len(documents)
        else:
            documents_by_index = document_store.get_all_documents(index=index)
            if not documents_by_index:
                document_store.write_documents(index=index, documents=documents)
                doc_counter += len(documents)
            else:
                unique_ids: List[str] = handler.handle_duplicate_documents(index=index, documents=documents_by_index)
                _documents: List[dict] = [document for document in documents if document["id"] not in unique_ids]
                if _documents:
                    document_store.write_documents(index=index, documents=_documents)
                    doc_counter += len(_documents)
                    logger.info(f'{len(_documents)} will be registered to DocumentStore')

    return HTTPException(
        status_code=status.HTTP_201_CREATED,
        detail=f'{doc_counter} document(s) were registered to DocumentStore'
    )


if __name__ == "__main__":
    document_store = ElasticsearchDocumentStore()
    # print(document_store.get_all_documents(index="index_science", return_embedding=True))
    print(document_store.delete_all_documents(index="index_science"))