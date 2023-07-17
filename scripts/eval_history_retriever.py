from typing import List, Text

import cohere

from saturn.evaluation.ir_eval import IREvaluator
from saturn.evaluation.schemas import Document, EvalData
from saturn.evaluation.utils import BaseRetriever
from saturn.utils import io


class CohereRetriever(BaseRetriever):
    def __init__(self, **kwargs):
        super().__init__()
        self.co = cohere.Client("Z16anlQpSDEMDFpTavLPMHJCW4tpIq1q9QnexcNN")
        self.model_name = kwargs.get("model_name", "rerank-multilingual-v2.0")
        self.top_k = kwargs.get("top_k", 3)
        self.keys = [
            "Z16anlQpSDEMDFpTavLPMHJCW4tpIq1q9QnexcNN",
            "Ri7sBvVB9rWVls47JtbADFUx2qGBA61xsEguincD",
            "evz5pLwKkgKuIpGtQQR2kSmsIOc6A1Uy7UxOyQfl",
            "XF2CeBbVgxB3x9biUStn5ITWJktcFZNDXcStAmgi",
            "xKRuMCqwaAD6ANi9cK4i2S2SSe8lLJwueQCohtBx"
        ]

    def rank(self, query: Text, documents: List[Document]) -> List[Document]:
        """

        Args:
            query:
            documents:

        Returns:

        """
        texts = [doc.content for doc in documents]
        results = None
        while not results:

            try:
                results = self.co.rerank(model=self.model_name, query=query, documents=texts, top_n=3)
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
            doc = documents[index]
            doc.score = score
            ranked_docs.append(doc)
        return ranked_docs


def read_eval_datasets(filename) -> List[EvalData]:
    eval_datasets = []
    rows = io.load_json(filename)
    for row in rows:
        eval_data = EvalData(
            query=row['question'],
            # answer=row['answer'],
            relevant_docs_id=row['relevant_docs_id']
        )
        eval_datasets.append(eval_data)
    return eval_datasets


def read_documents(filename: Text) -> List[Document]:
    docs = []
    raw_docs = io.load_json(filename)
    for row in raw_docs:
        doc = Document.from_text(text=row['context'], id=row['id'])
        docs.append(doc)
    return docs


def eval_ir_history_data():
    eval_datasets = read_eval_datasets("data/history/full_history_v4.0.0.json")
    documents = read_documents("data/history/document_history_all_v201.json")
    evaluator = IREvaluator(
        retriever=CohereRetriever(),
        eval_dataset=eval_datasets,
        documents=documents,
    )
    evaluator.build_records()
    evaluator.save_records("tmp")
    # evaluator.load_records("tmp/records.json")

    # Evaluate
    evaluator.run_eval()


if __name__ == '__main__':
    eval_ir_history_data()
    # read_addition_docs("data/history/document_history_all_v201.json")
