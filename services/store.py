import ast
import json

from iteration_utilities import unique_everseen
from fastapi import HTTPException, status

from config import DEFAULT_ENCODING
from utils.helper import validate_json
from utils.io import deep_container_fingerprint

CONTENT_TYPE = "application/json"


async def upload_document(files):
    allowed_documents = {}

    for file in files:
        bytes_str = await file.read()
        if file.content_type == CONTENT_TYPE:
            contents = json.loads(ast.literal_eval(str(bytes_str)).decode(DEFAULT_ENCODING))
            if validate_json(contents):
                for content in contents:
                    index = content["meta"]["index"]
                    id_hash_key = deep_container_fingerprint(f"{content['text']}_{index}")
                    content["id"] = id_hash_key
                    if index not in allowed_documents.keys():
                        allowed_documents[index] = list()
                    allowed_documents[index].append(content)

                    # Drop duplicates documents based on same hash ID
                    allowed_documents[index] = list(unique_everseen(allowed_documents[index]))

    return allowed_documents
