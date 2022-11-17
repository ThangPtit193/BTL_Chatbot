from collections import OrderedDict, namedtuple
from typing import Optional, Union, List, Dict
import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from meteor.document_stores.base import BaseDocumentStore
from meteor.errors import DocumentStoreError
from meteor.nodes.retriever.base import BaseRetriever
from meteor.schema import Document


logger = logging.getLogger(__name__)
# TODO make Paragraph generic for configurable units of text eg, pages, paragraphs, or split by a char_limit
Paragraph = namedtuple("Paragraph", ["paragraph_id", "document_id", "content", "meta"])


class TfidfRetriever(BaseRetriever):
    """
    Read all documents from a SQL backend.

    Split documents into smaller units (eg, paragraphs or pages) to reduce the
    computations when text is passed on to a Reader for QA.

    It uses sklearn's TfidfVectorizer to compute a tf-idf matrix.
    """

    def __init__(self, document_store: Optional[BaseDocumentStore] = None, top_k: int = 10, auto_fit=True):
        """
        :param document_store: an instance of a DocumentStore to retrieve documents from.
        :param top_k: How many documents to return per query.
        :param auto_fit: Whether to automatically update tf-idf matrix by calling fit() after new documents have been added
        """
        super().__init__()

        self.df = None
        self.tfidf_matrix = None
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words=None, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1)
        )
        self.document_store = document_store
        self.top_k = top_k
        self.auto_fit = auto_fit
        self.document_count = 0
        if document_store and document_store.get_document_count():
            self.fit(document_store=document_store)

    def _get_all_paragraphs(self, document_store: BaseDocumentStore) -> List[Paragraph]:
        """
        Split the list of documents in paragraphs
        """
        documents = document_store.get_all_documents()

        paragraphs = []
        p_id = 0
        for doc in documents:
            for p in doc.content.split(
                    "\n\n"
            ):  # TODO: this assumes paragraphs are separated by "\n\n". Can be switched to paragraph tokenizer.
                if not p.strip():  # skip empty paragraphs
                    continue
                paragraphs.append(Paragraph(document_id=doc.id, paragraph_id=p_id, content=(p,), meta=doc.meta))
                p_id += 1
        logger.info("Found %s candidate paragraphs from %s docs in DB", len(paragraphs), len(documents))
        return paragraphs

    def _calc_scores(self, queries: Union[str, List[str]]) -> List[Dict[int, float]]:
        if isinstance(queries, str):
            queries = [queries]
        question_vector = self.vectorizer.transform(queries)
        doc_scores_per_query = self.tfidf_matrix.dot(question_vector.T).T.toarray()
        doc_scores_per_query = [
            [(doc_idx, doc_score) for doc_idx, doc_score in enumerate(doc_scores)]
            for doc_scores in doc_scores_per_query
        ]
        indices_and_scores: List[Dict] = [
            OrderedDict(sorted(query_idx_scores, key=lambda tup: tup[1], reverse=True))
            for query_idx_scores in doc_scores_per_query
        ]
        return indices_and_scores

    def retrieve(
            self,
            query: str,
            filters: Optional[
                Union[
                    Dict[str, Union[Dict, List, str, int, float, bool]],
                    List[Dict[str, Union[Dict, List, str, int, float, bool]]],
                ]
            ] = None,
            top_k: Optional[int] = None,
            index: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
            scale_score: Optional[bool] = None,
            document_store: Optional[BaseDocumentStore] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number of documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                                           If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                                           Otherwise, raw similarity scores (e.g. cosine or dot_product) will be used.

        :param document_store: the document store to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        if document_store is None:
            document_store = self.document_store
            if document_store is None:
                raise ValueError(
                    "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
                )
        else:
            self.fit(document_store=document_store)

        if self.auto_fit:
            if document_store.get_document_count(headers=headers) != self.document_count:
                # run fit() to update self.df, self.tfidf_matrix and self.document_count
                logger.warning(
                    "Indexed documents have been updated and fit() method needs to be run before retrieval. Running it now."
                )
                self.fit(document_store=document_store)
        if self.df is None:
            raise DocumentStoreError(
                "Retrieval requires dataframe df and tf-idf matrix but fit() did not calculate them probably due to an empty document store."
            )

        if filters:
            raise NotImplementedError("Filters are not implemented in TfidfRetriever.")
        if index:
            raise NotImplementedError("Switching index is not supported in TfidfRetriever.")
        if scale_score:
            raise NotImplementedError("Scaling score to the unit interval is not supported in TfidfRetriever.")

        if top_k is None:
            top_k = self.top_k
        # get scores
        indices_and_scores = self._calc_scores(query)

        # rank paragraphs
        df_sliced = self.df.loc[indices_and_scores[0].keys()]
        df_sliced = df_sliced[:top_k]

        logger.debug(
            "Identified %s candidates via retriever:\n%s",
            df_sliced.shape[0],
            df_sliced.to_string(col_space=10, index=False),
        )

        # get actual content for the top candidates
        paragraphs = list(df_sliced.content.values)
        meta_data = [
            {"document_id": row["document_id"], "paragraph_id": row["paragraph_id"], "meta": row.get("meta", {})}
            for idx, row in df_sliced.iterrows()
        ]

        documents = []
        for para, meta in zip(paragraphs, meta_data):
            documents.append(Document(id=meta["document_id"], content=para, meta=meta.get("meta", {})))

        return documents

    def retrieve_batch(
            self,
            queries: Union[str, List[str]],
            filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
            top_k: Optional[int] = None,
            index: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
            batch_size: Optional[int] = None,
            scale_score: Optional[bool] = None,
            document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number of documents
        that are most relevant to the supplied queries.

        Returns a list of lists of Documents (one per query).

        :param queries: Single query string or list of queries.
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param batch_size: Not applicable.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true similarity scores (e.g. cosine or dot_product) which naturally have a different
                            value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise, raw similarity scores (e.g. cosine or dot_product) will be used.
        :param document_store: the document store to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        if document_store is None:
            document_store = self.document_store
            if document_store is None:
                raise ValueError(
                    "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
                )
        else:
            self.fit(document_store=document_store)

        if self.auto_fit:
            if document_store.get_document_count(headers=headers) != self.document_count:
                # run fit() to update self.df, self.tfidf_matrix and self.document_count
                logger.warning(
                    "Indexed documents have been updated and fit() method needs to be run before retrieval. Running it now."
                )
                self.fit(document_store=document_store)
        if self.df is None:
            raise DocumentStoreError(
                "Retrieval requires dataframe df and tf-idf matrix but fit() did not calculate them probably due to an empty document store."
            )

        if filters:
            raise NotImplementedError("Filters are not implemented in TfidfRetriever.")
        if index:
            raise NotImplementedError("Switching index is not supported in TfidfRetriever.")
        if scale_score:
            raise NotImplementedError("Scaling score to the unit interval is not supported in TfidfRetriever.")

        if top_k is None:
            top_k = self.top_k

        indices_and_scores = self._calc_scores(queries)
        all_documents = []
        for query_result in indices_and_scores:
            df_sliced = self.df.loc[query_result.keys()]
            df_sliced = df_sliced[:top_k]
            logger.debug(
                "Identified %s candidates via retriever:\n%s",
                df_sliced.shape[0],
                df_sliced.to_string(col_space=10, index=False),
            )

            # get actual content for the top candidates
            paragraphs = list(df_sliced.content.values)
            meta_data = [
                {"document_id": row["document_id"], "paragraph_id": row["paragraph_id"], "meta": row.get("meta", {})}
                for idx, row in df_sliced.iterrows()
            ]
            cur_documents = []
            for para, meta in zip(paragraphs, meta_data):
                cur_documents.append(Document(id=meta["document_id"], content=para, meta=meta.get("meta", {})))
            all_documents.append(cur_documents)

        return all_documents

    def fit(self, document_store: BaseDocumentStore):
        """
        Performing training on this class according to the TF-IDF algorithm.
        """
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the fit() method."
            )
        paragraphs = self._get_all_paragraphs(document_store=document_store)
        if not paragraphs or len(paragraphs) == 0:
            raise DocumentStoreError("Fit method called with empty document store")

        self.df = pd.DataFrame.from_dict(paragraphs)
        self.df["content"] = self.df["content"].apply(lambda x: " ".join(x))  # pylint: disable=unnecessary-lambda
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["content"])
        self.document_count = document_store.get_document_count()
