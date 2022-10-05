import os
import shutil
import uuid
from pathlib import Path
from typing import List, ClassVar

from fastapi import APIRouter, UploadFile, File, Depends, Query
from fastapi.security import HTTPBearer
from fastapi import HTTPException, status

from test.sample_pipelines import VenusServices

router = APIRouter()
# upload_service.get_config_params("/Users/phongnt/FTECH/knowledge-retrieval/rest_api/pipelines/master-pipelines.yml")

reusable_oauth2 = HTTPBearer(
    scheme_name='Authorization'
)


@router.post("/upload_file", dependencies=[Depends(reusable_oauth2)])
async def upload_file(
        # pretrained_retriever_model: str = Query("fschool-distilbert-multilingual-faq-v2.0.0",
        #                                         enum=("fschool-distilbert-multilingual-faq-v2.0.0")),
        retriever_selector: str = Query("fschool-distilbert-multilingual-faq-v2.0.0",
                                        enum=("fschool-distilbert-multilingual-faq-v2.0.0",
                                              "distilbert-multilingual-faq-v3.2")),
        files: List[UploadFile] = File(...)
):
    """
    This endpoint allows you upload multiple documents to document store
    """
    venus_services = VenusServices(retriever_pretrained=retriever_selector, host="localhost", port=9200)
    parent_file_path = Path("rest_api/upload")
    file_paths = []
    # clear cache
    if os.path.exists(os.path.abspath(parent_file_path)):
        list_dir = os.listdir(os.path.abspath(parent_file_path))
        if len(list_dir) > 10:
            shutil.rmtree(os.path.abspath(parent_file_path))

    if not os.path.exists(parent_file_path):
        os.mkdir(parent_file_path)

    for file in files:
        try:
            file_path = parent_file_path / f"{uuid.uuid4().hex}_{file.filename}"
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_paths.append(os.path.abspath(file_path))
        finally:
            file.file.close()

    return venus_services.run(file_paths=file_paths)


@router.get("/get_indices")
async def get_indices(document_store_type: str = Query("elasticsearch", enum=("elasticsearch", "inmemory"))):
    """
    This endpoint allows you get all indices stored in document store
    """
    venus_services = VenusServices(document_store_type=document_store_type, host="localhost", port=9200)
    return venus_services.get_all_indices()
