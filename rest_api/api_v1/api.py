from fastapi import APIRouter

from rest_api.api_v1.endpoints.file_upload import router

api_router = APIRouter()
api_router.include_router(router, prefix="/file-upload", tags=["file-upload"])
