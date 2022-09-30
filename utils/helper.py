import json
from loguru import logger
from typing import Union

from jsonschema import validate, exceptions

from schemas.document import DocumentEmbedding, ListDocumentEmbedding


def validate_document(data: Union[list, dict]) -> bool:
    try:
        if isinstance(data, (list, bytes)):
            _schema = ListDocumentEmbedding.schema()
        elif isinstance(data, dict):
            _schema = DocumentEmbedding.schema()
        else:
            raise TypeError(
                f"`data` must be a list or dict. Got {type(data)} instead"
            )

        _instance = json.loads(json.dumps(data))
        validate(instance=_instance, schema=_schema)
    except exceptions.ValidationError as err:
        return False
    return True
