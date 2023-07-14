from typing import List, Text

from pydantic import BaseModel


class Document(BaseModel):
    content: Text
    id: Text
    score: float = None
    meta: dict = None


class EvalData(BaseModel):
    query: Text
    answer: Text = None
    relevant_docs: List[Document]
