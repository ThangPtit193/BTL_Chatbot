from typing import List

from fastapi import APIRouter, UploadFile, File
from fastapi.params import Query

from services.store import upload_document

router = APIRouter()


@router.post("/")
async def upload_file(
        option: str = Query("default", enum=("default", "skip", "overwrite")),
        files: List[UploadFile] = File(description="Multiple files as UploadFile")
):
    return await upload_document(option, files)
