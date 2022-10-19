import os
from pathlib import Path
from enum import auto
from fastapi_utils.enums import StrEnum

PIPELINE_YAML_PATH = os.getenv(
    "PIPELINE_YAML_PATH", str((Path(__file__).parent / "pipelines" / "master-pipelines.yml").absolute())
)
QUERY_PIPELINE_NAME = os.getenv("QUERY_PIPELINE_NAME", "query")
INDEXING_PIPELINE_NAME = os.getenv("INDEXING_PIPELINE_NAME", "indexing")

FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", str((Path(__file__).parent / "file-upload").absolute()))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ROOT_PATH = os.getenv("ROOT_PATH", "/")

CONCURRENT_REQUEST_PER_WORKER = int(os.getenv("CONCURRENT_REQUEST_PER_WORKER", "4"))

PROJECT_DIR = os.path.abspath(os.curdir)


class DocumentStoreOption(StrEnum):
    elasticsearch = auto()
    inmemory = auto()
