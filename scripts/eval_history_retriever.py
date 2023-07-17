import csv
import os
from typing import List, Text
import time
import cohere
import tqdm
from saturn.evaluation.ir_eval import ir_evaluation, IREvaluator
from saturn.evaluation.schemas import Document, EvalData
from saturn.evaluation.utils import BaseRetriever


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
                print(f"wait for valid key 2s: {self.co.api_key} ")
                time.sleep(2)
        # Get index and score from docs
        indices_scores = [(doc.index, doc.relevance_score) for doc in results.results]

        # final document after ranking
        ranked_docs = []
        for index, score in indices_scores:
            doc = documents[index]
            doc.score = score
            ranked_docs.append(doc)
        return ranked_docs


def read_csv_to_dict(filename) -> List[EvalData]:
    eval_datasets = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            eval_data = EvalData(
                query=row['question'],
                answer=row['answer'],
                relevant_docs=[Document.from_text(text=row['context'], id=row['context_id'])],
            )
            eval_datasets.append(eval_data)
    return eval_datasets[:10]


def eval_ir_history_data():
    eval_datasets = read_csv_to_dict("data/history/cttgt2_v201.csv")
    # ir_evaluation(
    #     retriever=CohereRetriever(),
    #     eval_dataset=eval_datasets,
    #     top_k=5,
    #     report_dir="reports",
    #     model_name="rerank-english-v2.0",
    # )
    evaluator = IREvaluator(
        retriever=CohereRetriever(),
        eval_dataset=eval_datasets,
    )
    evaluator.build_records()
    evaluator.save_records("tmp")
    # evaluator.load_records("tmp/records.json")

    # Evaluate
    evaluator.run_eval()


if __name__ == '__main__':
    # query = "What is the capital of the United States?"
    # docs = [
    #     "Carson City is the capital city of the American state of Nevada. At the 2010 United States Census, Carson City had a population of 55,274.",
    #     "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
    #     "Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
    #     "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district. The President of the USA and many major national government offices are in the territory. This makes it the political center of the United States of America.",
    #     "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states. The federal government (including the United States military) also uses capital punishment."]
    # docs = [Document.from_text(doc) for doc in docs]
    # co_retriever = CohereRetriever()
    # results = co_retriever.rank(query, docs)
    # pprint.pprint(results)
    eval_ir_history_data()
