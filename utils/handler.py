import logging
from abc import ABC
from typing import List, Set, Optional, Union, Any

from venus.document_store.elasticsearch_store import ElasticsearchDocumentStore

from schemas.document import DocumentEmbedding

logger = logging.getLogger(__name__)


class Handler(ElasticsearchDocumentStore, ABC):
    def __init__(
            self,
            index: Optional[str] = None,
            duplicate_documents: Optional[bool] = True
    ):
        super().__init__()
        self.index = index
        self.duplicate_documents = duplicate_documents

    def _drop_duplicate_documents(self, documents: List[DocumentEmbedding]) -> Union[List[DocumentEmbedding], Any]:
        """
        Drop duplicates documents based on same hash ID

        :param documents: A list of  DocumentEmbedding objects.
        :return: A list of DocumentEmbedding objects and optional hash ID list
        """
        _hash_ids: Set = set([])
        _documents: List[DocumentEmbedding] = []

        for document in documents:
            if document['id'] in _hash_ids:
                logger.info(
                    f"Duplicate Documents: Document with id '{document['id']}' already exists in index "
                )
                continue
            _documents.append(document)
            _hash_ids.add(document['id'])

        return _documents

    def handle_duplicate_documents(
            self,
            documents: Union[List[DocumentEmbedding], List],
            duplicate_documents: Optional[bool] = None,
            index: Optional[str] = None
    ) -> List:
        """
        Checks whether any of the passed documents is already existing in the chosen index and returns a list of
        documents that are not in the index yet.

        :param documents: A list of DocumentEmbedding objects.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : (True, False)
                                    False: (default option): Ignore the duplicates documents
                                    True: Update any existing documents with the same ID when adding documents.
        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :return: A list of unique ids
        """

        index = index or self.index
        duplicate_documents = duplicate_documents or self.duplicate_documents
        if duplicate_documents:
            documents = self._drop_duplicate_documents(documents)

        ids = [doc['id'] for doc in documents]

        documents_found = self.get_documents_by_id(ids, index=index)
        ids_exist_in_db: List[str] = [doc.id for doc in documents_found]  # noqa

        # only get document id in DocumentStore
        # documents = list(
        #     filter(lambda doc: doc.id not in ids_exist_in_db, documents))

        return ids_exist_in_db

    @staticmethod
    def is_index_available(index) -> bool:
        """
        Check whether index is existing in Document Store or not
        :param index: Name of the index to get the documents from
        :return: Status of index True or False
        """
        try:
            es = ElasticsearchDocumentStore()
            docs = es.get_all_documents(index=index)
            if docs:
                return True
        except:
            return False
