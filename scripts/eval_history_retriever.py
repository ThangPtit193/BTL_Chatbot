import cohere
import pprint
from saturn.evaluation.utils import BaseRetriever
from saturn.evaluation.schemas import Document
from saturn.evaluation.ir_eval import ir_evaluation
from typing import Text, List
import csv


class CohereRetriever(BaseRetriever):
    def __init__(self, **kwargs):
        super().__init__()
        self.co = cohere.Client("Z16anlQpSDEMDFpTavLPMHJCW4tpIq1q9QnexcNN")
        self.model_name = kwargs.get("model_name", "rerank-english-v2.0")
        self.top_k = kwargs.get("top_k", 3)

    def rank(self, query: Text, documents: List[Document]) -> List[Document]:
        """

        Args:
            query:
            documents:

        Returns:

        """
        texts = [doc.content for doc in documents]
        results = self.co.rerank(model="rerank-english-v2.0", query=query, documents=texts, top_n=3)

        # Get index and score from docs
        indices_scores = [(doc.index, doc.relevance_score) for doc in results.results]

        # final document after ranking
        ranked_docs = []
        for index, score in indices_scores:
            doc = documents[index]
            doc.score = score
            ranked_docs.append(doc)
        return ranked_docs


def read_csv_to_dict(filename):
    result_dict = {}
    result_list = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            result_list.append(row)
    return result_list


def eval_history_data():
    # Load dara from csv file as json
    import csv
    import pandas as pd
    datasets = read_csv_to_dict("data/history/cttgt2_v201.csv")
    pprint.pprint(datasets)
    #


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
    eval_history_data()
