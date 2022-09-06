import json
import logging
from typing import Union

from jsonschema import validate, exceptions

from schemas.document import DocumentEmbedding, ListDocumentEmbedding

logger = logging.getLogger(__name__)


def validate_json(data: Union[list, dict]) -> bool:
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
        logger.info(err)
        return False
    return True
