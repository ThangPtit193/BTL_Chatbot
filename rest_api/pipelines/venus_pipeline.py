from typing import Text, List, Union, Optional, Dict, Any

from venus.pipelines.pipeline import Pipeline, BaseStandardPipeline
from venus.retriever import Retriever
from venus.utils.schema import Document


class VenusPipeline(BaseStandardPipeline):
    def __init__(self, retriever: Retriever):
        """
        Initialize a Pipeline for uploading and indexing document
        :param retriever:
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

        self.retriever = retriever

    def run(self, query: str, filters: Optional[Dict] = None,
            top_k_retriever: Optional[int] = None, **kwargs):
        output = self.pipeline.run(query=query, filters=filters, top_k_retriever=top_k_retriever,
                                   **kwargs)
        documents = output["documents"]

        results: Dict = {"query": query, "answers": []}
        for doc in documents:
            cur_answer = {
                "query": doc.text,
                "answer": doc.meta["answer"],
                "document_id": doc.id,
                "context": doc.meta["answer"],
                "score": float(doc.score),
                "probability": float(doc.probability),
                "offset_start": 0,
                "offset_end": len(doc.meta["answer"]),
                "meta": doc.meta,
            }

            results["answers"].append(cur_answer)
        return results

    def upload(
            self,
            documents: Union[Text, Union[List[dict], List[Document]]],
            document_store,
            index: Text,
    ):
        if isinstance(documents, list):
            assert len(documents) > 0, f"No qualified document was found in {document_store}"

        document_store.write_documents(documents=documents, index=index)
        document_store.update_embeddings(retriever=self.retriever, index=index)
        document_store.save("./models")
