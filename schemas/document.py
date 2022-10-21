from typing import List, Optional
from pydantic import BaseModel

from enum import auto
from fastapi_utils.enums import StrEnum


class Meta(BaseModel):
    answer: str
    adjacency_pair: str
    domain: str
    index: str


class DocumentEmbedding(BaseModel):
    id: Optional[str] = None
    text: str
    meta: Meta


class ListDocumentEmbedding(BaseModel):
    __root__: List[DocumentEmbedding]


class DocumentStoreOption(StrEnum):
    elasticsearch = auto()
    inmemory = auto()
