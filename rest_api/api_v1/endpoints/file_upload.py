from typing import List

from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.security import HTTPBearer

from services.upload_document import upload_document

router = APIRouter()

reusable_oauth2 = HTTPBearer(
    scheme_name='Authorization'
)


@router.post("/", dependencies=[Depends(reusable_oauth2)])
async def upload_file(
        # option: str = Query("default", enum=("default", "skip", "overwrite")),
        files: List[UploadFile] = File(...)
):
    """
    This endpoint allows you upload multiple documents to document store
    """
    return await upload_document(files)
