import json

CONTENT_TYPE = "application/json"


async def upload_document(option, files):
    allowed_files = []
    for file in files:
        contents = await file.read()
        if file.content_type == CONTENT_TYPE:
            pass

