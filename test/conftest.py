from datetime import timedelta
from typing import Any, List, Dict, Union

import subprocess
from uuid import UUID
import time
import gc
from pathlib import Path

import requests_cache
import responses

import numpy as np
import pytest
from elasticsearch import Elasticsearch

from meteor import BaseComponent
from meteor.document_stores.memory import InMemoryDocumentStore
from meteor.document_stores.base import BaseDocumentStore
from meteor.nodes.retriever.sparse import TfidfRetriever

from meteor.nodes.retriever.dense import EmbeddingRetriever
from meteor.schema import Document

# To manually run the tests with default PostgreSQL instead of SQLite, switch the lines below
SQL_TYPE = "sqlite"
SAMPLES_PATH = Path(__file__).parent / "samples"
DC_API_ENDPOINT = "https://DC_API/v1"
DC_TEST_INDEX = "document_retrieval_1"
DC_API_KEY = "NO_KEY"
MOCK_DC = True

# Set metadata fields used during testing for PineconeDocumentStore meta_config
META_FIELDS = [
    "meta_field",
    "name",
    "date_field",
    "numeric_field",
    "f1",
    "f3",
    "meta_id",
    "meta_field_for_count",
    "meta_key_1",
    "meta_key_2",
]

# Cache requests (e.g. huggingface model) to circumvent load protection
# See https://requests-cache.readthedocs.io/en/stable/user_guide/filtering.html
requests_cache.install_cache(urls_expire_after={"huggingface.co": timedelta(hours=1), "*": requests_cache.DO_NOT_CACHE})


#
# Empty mocks, as a base for unit tests.
#
# Monkeypatch the methods you need with either a mock implementation
# or a unittest.mock.MagicMock object (https://docs.python.org/3/library/unittest.mock.html)
#


@pytest.fixture
def docs_all_formats() -> List[Union[Document, Dict[str, Any]]]:
    return [
        {
            "content": "Cảnh báo được gửi đến các tư lệnh Lục quân và Hải quân Hoa Kỳ tại Hawaii nhưng tin tức này không được nhận đúng lúc vì lỗi của bộ máy hành chính."
        },
        {
            "content": "Có thể hiểu nguyên tử nào ạ"
        },
        {
            "content": "Người đồng sáng lập tập đoàn Microsoft Paul Allen ra đi ở tuổi 65"
        },
        {
            "content": "Paris nằm ở điểm gặp nhau của các hành trình thương mại đường bộ và đường sông, và là trung tâm của một vùng nông nghiệp giàu có"
        },
        {
            "content": "em ơi mình cần biết giải thích về nguyên tử"
        },
        {
            "content": "'Người đồng sáng lập tập đoàn Microsoft Paul Allen qua đời ở tuổi 65'",
        },
        {
            "content": "Sinh ra tại Seattle, tiểu bang Washington, Paul Allen trở thành người bạn thân thiết từ thuở niên thiếu của Bill Gates. Chính ông là người đã thuyết phục Bill Gates bỏ Đại học Harvard để cùng nhau thành lập Tập đoàn Microsoft vào năm 1975.",
        },
        {
            "content": "Chính ông là người đã thuyết phục Bill Gates bỏ Đại học Harvard để cùng nhau thành lập Tập đoàn Microsoft vào năm 1975.",
        },
        {
            "content": "Paris nằm ở điểm gặp nhau của các hành trình thương mại đường bộ và đường sông, và là trung tâm của một vùng nông nghiệp giàu có"
        },
        {
            "content": "Chultem phân biệt ba kiểu kiến trúc truyền thống Mông Cổ: Mông Cổ, Tây Tạng và Trung Quốc và kiểu kết hợp."
        },
        # metafield at the top level for backward compatibility
        # {
        #     "content": "My name is Paul and I live in New York",
        #     "meta_field": "test2",
        #     "name": "filename2",
        #     "date_field": "2019-10-01",
        #     "numeric_field": 5.0,
        # },
        # # "dict" format
        # {
        #     "content": "My name is Carla and I live in Berlin",
        #     "meta": {"meta_field": "test1", "name": "filename1", "date_field": "2020-03-01", "numeric_field": 5.5},
        # },
        # # Document object
        # Document(
        #     content="My name is Christelle and I live in Paris",
        #     meta={"meta_field": "test3", "name": "filename3", "date_field": "2018-10-01", "numeric_field": 4.5},
        # ),
        # Document(
        #     content="My name is Camila and I live in Madrid",
        #     meta={"meta_field": "test4", "name": "filename4", "date_field": "2021-02-01", "numeric_field": 3.0},
        # ),
        # Document(
        #     content="My name is Matteo and I live in Rome",
        #     meta={"meta_field": "test5", "name": "filename5", "date_field": "2019-01-01", "numeric_field": 0.0},
        # ),
    ]


@pytest.fixture
def docs(docs_all_formats) -> List[Document]:
    return [Document.from_dict(doc) if isinstance(doc, dict) else doc for doc in docs_all_formats]


@pytest.fixture
def docs_with_ids(docs) -> List[Document]:
    # Should be already sorted
    uuids = [
        UUID("190a2421-7e48-4a49-a639-35a86e202dfb"),
        UUID("20ff1706-cb55-4704-8ae8-a3459774c8dc"),
        UUID("5078722f-07ae-412d-8ccb-b77224c4bacb"),
        UUID("81d8ca45-fad1-4d1c-8028-d818ef33d755"),
        # UUID("f985789f-1673-4d8f-8d5f-2b8d3a9e8e23"),
    ]
    uuids.sort()
    for doc, uuid in zip(docs, uuids):
        doc.id = str(uuid)
    return docs


@pytest.fixture
def docs_with_random_emb(docs) -> List[Document]:
    for doc in docs:
        doc.embedding = np.random.random([768])
    return docs


@pytest.fixture(autouse=True)
def gc_cleanup(request):
    """
    Run garbage collector between tests in order to reduce memory footprint for CI.
    """
    yield
    gc.collect()


@pytest.fixture(scope="session")
def elasticsearch_fixture():
    # test if a ES cluster is already running. If not, download and start an ES instance locally.
    try:
        client = Elasticsearch(hosts=[{"host": "localhost", "port": "9200"}])
        client.info()
    except:
        print("Starting Elasticsearch ...")
        status = subprocess.run(["docker rm haystack_test_elastic"], shell=True)
        status = subprocess.run(
            [
                'docker run -d --name haystack_test_elastic -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2'
            ],
            shell=True,
        )
        if status.returncode:
            raise Exception("Failed to launch Elasticsearch. Please check docker container logs.")
        time.sleep(30)


@pytest.fixture(params=["es_filter_only", "elasticsearch", "dpr", "embedding", "tfidf", "table_text_retriever"])
def retriever(request, document_store):
    return get_retriever(request.param, document_store)


# @pytest.fixture(params=["es_filter_only", "elasticsearch", "dpr", "embedding", "tfidf"])
@pytest.fixture(params=["tfidf"])
def retriever_with_docs(request, document_store_with_docs):
    return get_retriever(request.param, document_store_with_docs)


def get_retriever(retriever_type, document_store):
    if retriever_type == "tfidf":
        retriever = TfidfRetriever(document_store=document_store)
    elif retriever_type == "embedding":
        retriever = EmbeddingRetriever(
            document_store=document_store, embedding_model="ms-viquad-bi-encoder-phobert-base", use_gpu=False
        )
        # retriever.update_embeddings()
    else:
        raise Exception(f"No retriever fixture for '{retriever_type}'")

    return retriever


@pytest.fixture(params=["memory"])
def document_store_with_docs(request, docs, tmp_path, monkeypatch):
    embedding_dim = request.node.get_closest_marker("embedding_dim", pytest.mark.embedding_dim(768))
    document_store = get_document_store(
        document_store_type=request.param, embedding_dim=embedding_dim.args[0], tmp_path=tmp_path
    )
    document_store.write_documents(docs)
    document_store.update_embedding(retriever=get_retriever("embedding", document_store=get_document_store("memory")))
    yield document_store
    document_store.delete_index(document_store.index)


@pytest.fixture
def document_store(request, tmp_path, monkeypatch: pytest.MonkeyPatch):
    embedding_dim = request.node.get_closest_marker("embedding_dim", pytest.mark.embedding_dim(768))
    document_store = get_document_store(
        document_store_type=request.param, embedding_dim=embedding_dim.args[0], tmp_path=tmp_path
    )
    yield document_store
    document_store.delete_index(document_store.index)


def get_document_store(
        document_store_type,
        tmp_path=None,
        embedding_dim=768,
        embedding_field="embedding",
        index="meteor_test",
        similarity: str = "cosine",
        recreate_index: bool = True,
):  # cosine is default similarity as dot product is not supported by Weaviate
    document_store: BaseDocumentStore
    if document_store_type == "memory":
        document_store = InMemoryDocumentStore(
            return_embedding=True,
            embedding_dim=embedding_dim,
            embedding_field=embedding_field,
            index=index,
            similarity=similarity,
        )
    else:
        raise Exception(f"No document store fixture for '{document_store_type}'")
    return document_store