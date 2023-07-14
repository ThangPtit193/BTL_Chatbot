from .utils import BaseRetriever
from .schemas import EvalData, Document
from typing import List


def ir_evaluation(
        retriever: BaseRetriever, eval_dataset: List[EvalData], additional_docs: List[Document] = None,
        threshold: float = None, top_k: int = None, report_dir: str = None
):
    """
    Evaluates the performance of a retriever on a given dataset, optionally including additional documents.


    Args:
        retriever (BaseRetriever): An instance of a retriever that implements the `BaseRetriever` interface.
        eval_dataset (List[EvalData]): A list of `EvalData` objects representing the evaluation dataset.
        additional_docs (List[Document]): A list of additional `Document` objects to be included in the index.
            If None, only the documents in the evaluation dataset will be used.
        threshold (float): A threshold value for document scores. Only documents with scores greater than or equal to
            this threshold will be considered relevant.
        top_k (int): The maximum number of documents to retrieve for each query. If not None, only the top `k` documents
            will be considered for evaluation.
        report_dir (str): The directory where evaluation reports will be saved. If None, no reports will be saved.
    Returns:
        A dictionary containing the evaluation results.
    """
    # TODO: Implement this function
    full_docs = [] + additional_docs
    the_most_relevant_docs = retriever(eval_dataset[0].query, full_docs)

