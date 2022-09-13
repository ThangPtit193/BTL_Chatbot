from abc import abstractmethod
from typing import Optional, List, Dict, Union, Generator, Set, Any

import numpy as np
from loguru import logger

from pipelines.nodes.base import BaseComponent
from pipelines.schema import Document


def njit(f):
    return f


@njit  # (fastmath=True)
def expit(x: float) -> float:
    return 1 / (1 + np.exp(-x))


class BaseDocumentStore(BaseComponent):
    """
    Base class for implementing Document Stores.
    """

    outgoing_edges: int = 1

    index: Optional[str]
    label_index: Optional[str]
    similarity: Optional[str]
    duplicate_document_options: tuple = ("skip", "overwrite", "fail")

    @abstractmethod
    def write_documents(
            self,
            documents: Union[List[dict], List[Document]],
            index: Optional[str] = None,
            batch_size: int = 10_000,
            duplicate_documents: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
    ):
        """
        Indexes documents for later queries.

        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
        :param index: Optional name of index where the documents shall be written to.
                      If None, the DocumentStore's default index (self.index) will be used.
        :param batch_size: Number of documents that are passed to bulk function at a time.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to document store client if supported
                        (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

        :return: None
        """
        pass

    @abstractmethod
    def get_all_documents(
            self,
            index: Optional[str] = None,
            filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
            return_embedding: Optional[bool] = None,
            batch_size: int = 10_000,
            headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:
                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            ```

        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: Number of documents that are passed to bulk function at a time.
        :param headers: Custom HTTP headers to pass to document store client if supported
                        e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication
        """
        pass

    @abstractmethod
    def get_all_documents_generator(
            self,
            index: Optional[str] = None,
            filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
            return_embedding: Optional[bool] = None,
            batch_size: int = 10_000,
            headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        Get documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                        __Example__:
                        ```python
                        filters = {
                            "$and": {
                                "type": {"$eq": "article"},
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": {"$in": ["economy", "politics"]},
                                    "publisher": {"$eq": "nytimes"}
                                }
                            }
                        }
                        ```

        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param headers: Custom HTTP headers to pass to document store client if supported
                        e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication
        """
        pass

    def __iter__(self):
        if not self.ids_iterator:
            self.ids_iterator = [x.id for x in self.get_all_documents()]
        return self

    def __next__(self):
        if len(self.ids_iterator) == 0:
            raise StopIteration
        curr_id = self.ids_iterator[0]
        ret = self.get_document_by_id(curr_id)
        self.ids_iterator = self.ids_iterator[1:]
        return ret

    @abstractmethod
    def get_document_by_id(
            self, id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> Optional[Document]:
        pass

    @abstractmethod
    def get_document_count(
            self,
            filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
            index: Optional[str] = None,
            only_documents_without_embedding: bool = False,
            headers: Optional[Dict[str, str]] = None,
    ) -> int:
        pass

    def normalize_embedding(self, emb: np.ndarray) -> None:
        """
        Performs L2 normalization of embeddings vector inplace. Input can be a single vector (1D array) or a matrix
        (2D array).
        """
        # Might be extended to other normalizations in future

        # Single vec
        if len(emb.shape) == 1:
            self._normalize_embedding_1D(emb)
        # 2D matrix
        else:
            self._normalize_embedding_2D(emb)

    @staticmethod
    @njit  # (fastmath=True)
    def _normalize_embedding_1D(emb: np.ndarray) -> None:
        norm = np.sqrt(emb.dot(emb))  # faster than np.linalg.norm()
        if norm != 0.0:
            emb /= norm

    @staticmethod
    @njit  # (fastmath=True)
    def _normalize_embedding_2D(emb: np.ndarray) -> None:
        for vec in emb:
            vec = np.ascontiguousarray(vec)
            norm = np.sqrt(vec.dot(vec))
            if norm != 0.0:
                vec /= norm

    def scale_to_unit_interval(self, score: float, similarity: Optional[str]) -> float:
        if similarity == "cosine":
            return (score + 1) / 2
        else:
            return float(expit(score / 100))

    @abstractmethod
    def query_by_embedding(
            self,
            query_emb: np.ndarray,
            filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
            top_k: int = 10,
            index: Optional[str] = None,
            return_embedding: Optional[bool] = None,
            headers: Optional[Dict[str, str]] = None,
            scale_score: bool = True,
    ) -> List[Document]:
        pass

    def delete_all_documents(
            self,
            index: Optional[str] = None,
            filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
            headers: Optional[Dict[str, str]] = None,
    ):
        pass

    @abstractmethod
    def delete_documents(
            self,
            index: Optional[str] = None,
            ids: Optional[List[str]] = None,
            filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
            headers: Optional[Dict[str, str]] = None,
    ):
        pass

    @abstractmethod
    def delete_index(self, index: str):
        """
        Delete an existing index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        pass

    @abstractmethod
    def _create_document_field_map(self) -> Dict:
        pass

    @abstractmethod
    def get_documents_by_id(
            self,
            ids: List[str],
            index: Optional[str] = None,
            batch_size: int = 10_000,
            headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        pass

    @abstractmethod
    def update_document_meta(self, id: str, meta: Dict[str, Any], index: str = None):
        pass

    def _drop_duplicate_documents(self, documents: List[Document], index: Optional[str] = None) -> List[Document]:
        """
        Drop duplicates documents based on same hash ID

        :param documents: A list of Haystack Document objects.
        :param index: name of the index
        :return: A list of Haystack Document objects.
        """
        _hash_ids: Set = set([])
        _documents: List[Document] = []

        for document in documents:
            if document.id in _hash_ids:
                logger.info(
                    f"Duplicate Documents: Document with id '{document.id}' already exists in index "
                    f"'{index or self.index}'"
                )
                continue
            _documents.append(document)
            _hash_ids.add(document.id)

        return _documents

    def _handle_duplicate_documents(
            self,
            documents: List[Document],
            index: Optional[str] = None,
            duplicate_documents: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
    ):
        """
        Checks whether any of the passed documents is already existing in the chosen index and returns a list of
        documents that are not in the index yet.

        :param documents: A list of Haystack Document objects.
        :param index: name of the index
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip (default option): Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to document store client if supported
                        e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication
        :return: A list of Haystack Document objects.
        """

        index = index or self.index
        if duplicate_documents in ("skip", "fail"):
            documents = self._drop_duplicate_documents(documents, index)
            documents_found = self.get_documents_by_id(ids=[doc.id for doc in documents], index=index, headers=headers)
            ids_exist_in_db: List[str] = [doc.id for doc in documents_found]

            if len(ids_exist_in_db) > 0 and duplicate_documents == "fail":
                logger.warning(
                    f"Document with ids '{', '.join(ids_exist_in_db)} already exists" f" in index = '{index}'."
                )

            documents = list(filter(lambda doc: doc.id not in ids_exist_in_db, documents))

        return documents

    def run(
            self,
            documents: List[Union[dict, Document]],
            index: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
            id_hash_keys: Optional[List[str]] = None,
    ):
        """
        Run requests of document stores

        Comment: We will gradually introduce the primitives. The document stores also accept dicts
                                                            and parse them to documents.
        In the future, however, only documents themselves will be accepted. Parsing the dictionaries in the run function
        is therefore only an interim solution until the run function also accepts documents.

        :param documents: A list of dicts that are documents.
        :param headers: A list of headers.
        :param index: Optional name of index where the documents shall be written to.
                      If None, the DocumentStore's default index (self.index) will be used.
        :param id_hash_keys: List of the fields that the hashes of the ids are generated from.
        """

        field_map = self._create_document_field_map()
        doc_objects = [
            Document.from_dict(d, field_map=field_map, id_hash_keys=id_hash_keys) if isinstance(d, dict) else d
            for d in documents
        ]
        self.write_documents(documents=doc_objects, index=index, headers=headers)
        return {}, "output_1"

    def run_batch(  # type: ignore
            self,
            documents: List[Union[dict, Document]],
            index: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
            id_hash_keys: Optional[List[str]] = None,
    ):
        return self.run(documents=documents, index=index, headers=headers, id_hash_keys=id_hash_keys)
