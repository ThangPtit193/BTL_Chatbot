from typing import List

from pydantic import BaseModel


class Meta(BaseModel):
    answer: str
    adjacency_pair: str
    domain: str
    index: str


class DocumentEmbedding(BaseModel):
    text: str
    meta: Meta


class ListDocumentEmbedding(BaseModel):
    __root__: List[DocumentEmbedding]
