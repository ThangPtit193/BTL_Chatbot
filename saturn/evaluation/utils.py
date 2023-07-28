from typing import List, Text

from .schemas import Document


class BaseRetriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents

    def rank(self, query: Text) -> List[Document]:
        """
        Ranks a list of documents based on their relevance to a given query.

        Args:
            query: A string representing the query for which relevant documents are to be retrieved.

        Returns:
            A list of `Document` objects sorted in descending order of their relevance to the query.
            Must be included the score of the document in the `score` field o

        Notes:
            The ranking algorithm used in this function is specific to the application or use case, and is not described here.
        """
        raise NotImplementedError

    def rank_batch(self, query_batch: List[Text]) -> List[List[Document]]:
        """
        Ranks a list of documents based on their relevance to a given query.

        Args:
            query_batch:A list of strings representing the query for which relevant documents are to be retrieved.

        Returns:
            A list of `Document` objects sorted in descending order of their relevance to the query.
            Must be included the score of the document in the `score` field o

        Notes:
            The ranking algorithm used in this function is specific to the application or use case, and is not described here.
        """
        raise NotImplementedError

    def __call__(self, query: Text) -> List[Document]:
        return self.rank(query)
