from typing import List

from pydantic import BaseModel


class Meta(BaseModel):
    answer: str
    adjacency_pair: str
    domain: str
    index: str


class ModelItem(BaseModel):
    text: str
    meta: Meta


class DocumentInput(BaseModel):
    __root__: List[ModelItem]
