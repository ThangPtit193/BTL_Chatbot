from typing import List, Text

from .schemas import Document


class BaseRetriever:
    def __init__(self):
        pass

    def rank(self, query: Text, documents: List[Document]) -> List[Document]:
        """
        Ranks a list of documents based on their relevance to a given query.

        Args:
            query: A string representing the query for which relevant documents are to be retrieved.
            documents: A list of `Document` objects containing the documents to be ranked.

        Returns:
            A list of `Document` objects sorted in descending order of their relevance to the query.
            Must be included the score of the document in the `score` field o

        Notes:
            The ranking algorithm used in this function is specific to the application or use case, and is not described here.
        """
        raise NotImplementedError

    def __call__(self, query: Text, documents: List[Document]) -> List[Document]:
        return self.rank(query, documents)
