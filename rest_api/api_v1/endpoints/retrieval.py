from fastapi import APIRouter, Query

from rest_api.controller.engine import VenusServices
from schemas.document import DocumentStoreOption

router = APIRouter()


@router.post("/retrieve_query/{query}")
async def retrieve(
        document_store_type: DocumentStoreOption,
        index: str = Query("document", enum=(
            VenusServices.init_instance(document_store_type="elasticsearch").get_all_indices()
        )),
        top_k: int = 2,
        query: str = None,

):
    vs = VenusServices.init_instance(document_store_type=document_store_type.name, index=index)
    return vs.search(query=query, top_k=top_k)
