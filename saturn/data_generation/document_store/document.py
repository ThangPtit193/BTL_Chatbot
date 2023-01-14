from typing import Optional, Dict, Any
from uuid import uuid4

import numpy as np


class Document:
    def __init__(
        self,
        text: str,
        id: Optional[str] = None,
        meta: Dict[str, Any] = None,
        embedding: Optional[np.ndarray] = None,
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
        # Create a unique ID (either new one, or one from user input)
        if id:
            self.id = str(id)
        else:
            self.id = str(uuid4())
        self.meta = meta or {}
        self.embedding = embedding
        self.positive_ids = []
        self.negatives_ids = {}

    def to_dict(self, field_map={}, with_embedding: bool = False):
        inv_field_map = {v: k for k, v in field_map.items()}
        _doc: Dict[str, str] = {}
        for k, v in self.__dict__.items():
            if not with_embedding and k == "embedding":
                continue
            k = k if k not in inv_field_map else inv_field_map[k]
            _doc[k] = v
        return _doc

    @classmethod
    def from_dict(cls, dict, field_map={}):
        _doc = dict.copy()
        init_args = ["text", "id", "question", "meta", "embedding", "type_doc"]
        if "meta" not in _doc.keys():
            _doc["meta"] = {}
        # copy additional fields into "meta"
        for k, v in _doc.items():
            if k not in init_args and k not in field_map:
                _doc["meta"][k] = v
        # remove additional fields from top level
        _new_doc = {}
        for k, v in _doc.items():
            if k in init_args:
                _new_doc[k] = v
            elif k in field_map:
                k = field_map[k]
                _new_doc[k] = v

        return cls(**_new_doc)

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())
