from typing import List, Optional

import pandas as pd
from pydantic import BaseModel
from pydantic.dataclasses import dataclass


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
