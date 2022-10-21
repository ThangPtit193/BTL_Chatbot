import os
import shutil
import uuid
from pathlib import Path
from typing import List
import ast

from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.security import HTTPBearer

from rest_api.controller import VenusServices
from rest_api.config import PROJECT_DIR, DocumentStoreOption

router = APIRouter()

reusable_oauth2 = HTTPBearer(
    scheme_name='Authorization'
)


@router.post("/upload_file", dependencies=[Depends(reusable_oauth2)])
async def upload_file(
        document_store_type: DocumentStoreOption,
        files: List[UploadFile] = File(...)
):
    """
    This endpoint allows you upload multiple documents to document store
    """

    parent_file_path = Path(os.path.join(PROJECT_DIR, "rest_api/upload"))
    file_paths = []
    # clear cache
    if os.path.exists(parent_file_path):
        list_dir = os.listdir(parent_file_path)
        if len(list_dir) > 10:
            shutil.rmtree(parent_file_path)

    if not os.path.exists(parent_file_path):
        os.mkdir(parent_file_path)

    for file in files:
        try:
            file_path = parent_file_path / f"{uuid.uuid4().hex}_{file.filename}"
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_paths.append(file_path)
        finally:
            file.file.close()

    return VenusServices.init_instance(document_store_type=document_store_type.name).run(file_paths=file_paths)


@router.get("/get_indices")
async def get_indices(document_store_type: DocumentStoreOption):
    """
    This endpoint allows you get all indices stored in document store
    """
    # if document_store_type == "inmemory":
    #     return {"message": "Querying index does not support for InMemoryDocumentStore"}
    return VenusServices.init_instance(document_store_type=document_store_type.name).get_all_indices()


@router.delete("/delete_documents/{index}", dependencies=[Depends(reusable_oauth2)])
async def delete_documents_by_index(
        document_store_type: DocumentStoreOption,
        index: str
):
    venus_services = VenusServices.init_instance(document_store_type=document_store_type.name)
    indices = venus_services.get_all_indices()
    if len(indices) == 0:
        return {"message": "No index found to delete"}

    return venus_services.delete_all_documents(index=index)


@router.get("/get_all_documents_by_index/{index}")
async def get_all_documents_by_index(document_store_type: DocumentStoreOption, index: str):
    try:
        _documents = VenusServices.init_instance(
            document_store_type=document_store_type.name).get_all_documents_by_index(
            index=index)
        return ast.literal_eval(str(_documents))
    except:
        return {"message": f"{index} not found"}


@router.delete("/delete_index/{index}")
async def delete_index(document_store_type: DocumentStoreOption, index: str):
    if document_store_type.name == "inmemory":
        return {"message": "Deleting index does not support for InMemoryDocumentStore"}
    return VenusServices.init_instance(document_store_type=document_store_type.name).delete_index(index=index)


@router.get("/get_all_documents")
async def get_all_documents(document_store_type: DocumentStoreOption):
    return VenusServices.init_instance(document_store_type=document_store_type.name).get_all_documents()
