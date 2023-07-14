from typing import List, Text, Dict
import uuid
from pydantic import BaseModel


class Document(BaseModel):
    content: Text
    id: Text
    score: float = None
    meta: dict = None

    @classmethod
    def from_text(cls, text: Text, id: Text = None, meta: Dict = None):
        return cls(
            content=text,
            id=id or uuid.uuid4().hex,
            meta=meta
        )


class EvalData(BaseModel):
    query: Text
    answer: Text = None
    relevant_docs: List[Document]
