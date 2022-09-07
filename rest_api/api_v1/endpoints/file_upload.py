from typing import List

from fastapi import APIRouter, UploadFile, File
from fastapi.params import Query

from services.upload_document import upload_document

router = APIRouter()


@router.post("/")
async def upload_file(
        # option: str = Query("default", enum=("default", "skip", "overwrite")),
        files: List[UploadFile] = File(description="Multiple files as UploadFile")
):
    """
    This endpoint allows you upload multiple documents to document store
    """
    return await upload_document(files)
