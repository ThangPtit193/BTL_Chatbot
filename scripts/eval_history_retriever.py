from typing import List, Text
import os
from scripts.retriever import CohereRerankRetriever, CohereEmbeddingRetriever, OpenAIEmbeddingRetriever
from saturn.evaluation.ir_eval import IREvaluator
from saturn.evaluation.schemas import Document, EvalData
from saturn.utils import io


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


def eval_ir_history_data_rerank():
    data_file_path = "data/history/cttgt2_v201.json"
    eval_datasets = read_eval_datasets(data_file_path)
    documents = read_documents("data/history/document_history_all_v201.json")
    retriever = CohereRerankRetriever(
        model_name="rerank-multilingual-v2.0",
        top_k=30
    )
    evaluator = IREvaluator(
        retriever=retriever,
        eval_dataset=eval_datasets,
        documents=documents,
    )
    evaluator.build_records()
    evaluator.save_records(save_dir="tmp", file_name="records_" + os.path.basename(data_file_path) + ".json")
    # evaluator.load_records(f"tmp/records_{os.path.basename(data_file_path)}.json")

    # Evaluate
    evaluator.run_eval(recall_top_k=[3, 5, 10, 15])


def eval_ir_history_data_embedding():
    # data_file_path = "data/history/cttgt2_v201.json"
    data_file_path = "data/history/full_history_v4.0.0.json"
    eval_datasets = read_eval_datasets(data_file_path)
    documents = read_documents("data/history/document_history_all_v201.json")
    retriever = CohereEmbeddingRetriever(
        documents=documents,
        model_name="embed-multilingual-v2.0",
        top_k=30
    )
    evaluator = IREvaluator(
        retriever=retriever,
        eval_dataset=eval_datasets,
        documents=documents
    )
    evaluator.build_records(batch_size=128)
    evaluator.save_records(
        save_dir="tmp",
        file_name=f"records_{retriever.__class__.__name__}_" + os.path.basename(data_file_path) + ".json"
    )
    evaluator.load_records(f"tmp/records_{retriever.__class__.__name__}_{os.path.basename(data_file_path)}.json")

    # Evaluate
    evaluator.run_eval(recall_top_k=[3, 5, 10, 15])


def eval_ir_history_data_openai_embedding():
    # data_file_path = "data/history/cttgt2_v201.json"
    data_file_path = "data/history/full_history_v4.0.0.json"
    eval_datasets = read_eval_datasets(data_file_path)
    documents = read_documents("data/history/document_history_all_v201.json")
    retriever = OpenAIEmbeddingRetriever(
        documents=documents,
        top_k=30
    )
    evaluator = IREvaluator(
        retriever=retriever,
        eval_dataset=eval_datasets,
        documents=documents
    )
    evaluator.build_records(batch_size=128)
    evaluator.save_records(
        save_dir="tmp",
        file_name=f"records_{retriever.__class__.__name__}_" + os.path.basename(data_file_path) + ".json"
    )
    evaluator.load_records(f"tmp/records_{retriever.__class__.__name__}_{os.path.basename(data_file_path)}.json")

    # Evaluate
    evaluator.run_eval(recall_top_k=[3, 5, 10, 15])


if __name__ == '__main__':
    eval_ir_history_data_openai_embedding()
