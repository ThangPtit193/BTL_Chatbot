from typing import List, Text

import cohere

from saturn.evaluation.schemas import Document
from saturn.evaluation.utils import BaseRetriever
from comet.components.similarity import CohereEmbedder, OpenAIEmbedder

COHERE_KEYS = [
    "Z16anlQpSDEMDFpTavLPMHJCW4tpIq1q9QnexcNN",
    "Ri7sBvVB9rWVls47JtbADFUx2qGBA61xsEguincD",
    "evz5pLwKkgKuIpGtQQR2kSmsIOc6A1Uy7UxOyQfl",
    "XF2CeBbVgxB3x9biUStn5ITWJktcFZNDXcStAmgi",
    "xKRuMCqwaAD6ANi9cK4i2S2SSe8lLJwueQCohtBx",
    "pHry8VqInDlKefi68qyYfoYtWY4Cddw3a8n2qOhX",
    "JtYtjpFkGGR34IzEhQevT6iHFFLpvaabnGz4ya9G",
    "OXw52cZTKuOi5sAb5nwl0RbWhQU5WMkILw35Q6G3",
    "dpr0PhIipg7iAuuFhVPN3jpN0MIbdZ8QRjDxRT8B",
    "gmJdnf2CPSskeDmFJ3L8jjGC36NqQ81dQqoxatPb"
]


class CohereRerankRetriever(BaseRetriever):
    def __init__(self, documents: List[Document], **kwargs):
        super().__init__(documents)
        self.co = cohere.Client("Z16anlQpSDEMDFpTavLPMHJCW4tpIq1q9QnexcNN")
        self.model_name = kwargs.get("model_name", "rerank-multilingual-v2.0")
        self.top_k = kwargs.get("top_k", 3)
        self.keys = COHERE_KEYS

    def rank(self, query: Text) -> List[Document]:
        """

        Args:
            query:
            documents:

        Returns:

        """
        texts = [doc.content for doc in self.documents]
        results = None
        while not results:

            try:
                results = self.co.rerank(model=self.model_name, query=query, documents=texts, top_n=self.top_k)
            except Exception as e:
                print(e)
                self.keys = self.keys[1:] + self.keys[:1]
                self.co.api_key = self.keys[0]
                results = None
                # print(f"wait for valid key 2s: {self.co.api_key} ")
                # time.sleep(2)
        # Get index and score from docs
        indices_scores = [(doc.index, doc.relevance_score) for doc in results.results]

        # final document after ranking
        ranked_docs = []
        for index, score in indices_scores:
            doc = self.documents[index]
            doc.score = score
            ranked_docs.append(doc)
        return ranked_docs

    def rank_batch(self, query_batch: List[Text]) -> List[List[Document]]:
        for query in query_batch:
            yield self.rank(query)


class CohereEmbeddingRetriever(BaseRetriever):
    def __init__(self, documents: List[Document], model_name="embed-multilingual-v2.0", top_k: int = 100, **kwargs):
        super().__init__(documents)
        # calculate embeddings
        self.embeder = CohereEmbedder(
            api_keys=COHERE_KEYS,
            model_name=model_name,
        )
        self.top_k = top_k
        self.documents = documents

    def rank(self, query: Text) -> List[Document]:
        ranked_docs_batch = self.rank_batch([query])
        return ranked_docs_batch[0]

    def rank_batch(self, query_batch: List[Text]) -> List[List[Document]]:
        texts = [doc.content for doc in self.documents]
        responses = self.embeder.find_similarity(
            src_texts=query_batch,
            tgt_texts=texts,
            top_n=self.top_k,
        )
        ranked_docs_batch = []
        for response in responses:
            relevance_docs = []
            for result in response:
                doc = self.documents[result['index']]
                doc.score = result["score"]
                relevance_docs.append(doc)
            ranked_docs_batch.append(relevance_docs)

        return ranked_docs_batch


class OpenAIEmbeddingRetriever(BaseRetriever):
    def __init__(self, documents: List[Document], model_name="text-embedding-ada-002", top_k: int = 100, **kwargs):
        super().__init__(documents)
        # calculate embeddings
        self.embeder = OpenAIEmbedder(
            model_name=model_name,
        )
        self.top_k = top_k
        self.documents = documents

    def rank(self, query: Text) -> List[Document]:
        ranked_docs_batch = self.rank_batch([query])
        return ranked_docs_batch[0]

    def rank_batch(self, query_batch: List[Text]) -> List[List[Document]]:
        texts = [doc.content for doc in self.documents]
        responses = self.embeder.find_similarity(
            src_texts=query_batch,
            tgt_texts=texts,
            top_n=self.top_k,
        )
        ranked_docs_batch = []
        for response in responses:
            relevance_docs = []
            for result in response:
                doc = self.documents[result['index']]
                doc.score = result["score"]
                relevance_docs.append(doc)
            ranked_docs_batch.append(relevance_docs)

        return ranked_docs_batch


if __name__ == "__main__":
    texts = [
        'Hello from Cohere!', 'مرحبًا من كوهير!', 'Hallo von Cohere!',
        'Bonjour de Cohere!', '¡Hola desde Cohere!', 'Olá do Cohere!',
        'Ciao da Cohere!', '您好，来自 Cohere！', 'कोहेरे से नमस्ते!'
    ]
    documents = [Document.from_text(text=text) for text in texts]
    retriever = CohereEmbeddingRetriever(documents)
    query = "Hello from Cohere!"
    ranked_docs = retriever.rank(query)
    print(ranked_docs)
