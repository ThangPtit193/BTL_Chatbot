from fastapi import APIRouter

from test.sample_pipelines import VenusServices
from rest_api.config import DocumentStoreOption

router = APIRouter()


@router.post("/retrieve_query/{query}")
async def retrieve(document_store_type: DocumentStoreOption, query: str, top_k: int = 2):
    return VenusServices.init_instance(document_store_type=document_store_type.name).search(query=query, top_k=top_k)
