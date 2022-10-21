from fastapi import APIRouter

from rest_api.api_v1.endpoints.document_store import router as upload_router
from rest_api.api_v1.endpoints.retrieval import router as retrieve_router
from rest_api.api_v1.endpoints.authentication import router as authenticator_router
api_router = APIRouter()
api_router.include_router(upload_router, prefix="/document_store", tags=["document_store"])
api_router.include_router(authenticator_router, prefix='/auth', tags=['auth'])
api_router.include_router(retrieve_router, prefix="/retrieval", tags=["retrieval"])
