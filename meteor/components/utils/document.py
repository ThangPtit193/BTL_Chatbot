from typing import *
import numpy as np
from uuid import uuid4


class EvalResult:
    def __init__(
        self, query: Text, query_id: Text, rr_score: float = None, ap_score: float = None, top_k_relevant: int = 20,
        most_relevant_docs: List["Document"] = None
    ):
        self.query = query
        self.query_id = query_id
        self.rr_score = rr_score
        self.ap_score = ap_score
        self.top_k_relevant = top_k_relevant
        self.most_relevant_docs = most_relevant_docs

    @classmethod
    def from_dict(cls, data: Dict):
        cls(**data)

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "top_k_relevant": self.top_k_relevant,
            "query_id": self.top_k_relevant,
            "rr_score": self.rr_score,
            "ap_score": self.ap_score,
            "most_relevant_docs": self.most_relevant_docs,
        }

    def __str__(self):
        return f"EvalResult(query={self.query}, rr_score={self.rr_score}, ap_score={self.ap_score})"

    def __repr__(self):
        return self.__str__()


class Document:
    def __init__(
        self,
        text: str,
        id: Optional[str] = None,
        label: Optional[str] = None,
        num_relevant: Optional[int] = None,
        embedding: Optional[np.ndarray] = None,
        meta: Dict[str, Any] = None,
    ):
        """
        Object used to represent documents / passages

        Args:
            text: Text of the document
            id: ID used within the DocumentStore
            meta: Meta fields for a document like name, url, or author.
            embedding: Vector encoding of the text
        """

        self.text = text
        self.label = label
        self.num_relevant = num_relevant
        # Create a unique ID (either new one, or one from user input)
        if id:
            self.id = str(id)
        else:
            self.id = str(uuid4())

        self.meta = meta or {}
        self.embedding = embedding

    def to_dict(self, with_embedding: bool = False):
        _doc: Dict[str, str] = {}
        for k, v in self.__dict__.items():
            if not with_embedding and k == "embedding":
                continue
            _doc[k] = v
        return _doc

    @classmethod
    def from_dict(cls, data):
        _doc = data.copy()
        init_args = ["text", "id", "label", "meta"]
        if "meta" not in _doc.keys():
            _doc["meta"] = {}
        # copy additional fields into "meta"
        for k, v in _doc.items():
            if k not in init_args:
                _doc["meta"][k] = v
        # remove additional fields from top level
        _new_doc = {}
        for k, v in _doc.items():
            if k in init_args:
                _new_doc[k] = v

        return cls(**_new_doc)

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())
