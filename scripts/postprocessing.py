from saturn.utils import io
from typing import *
import csv


def read_addition_docs(filename: Text) -> List[Dict]:
    docs = []
    raw_docs = io.load_json(filename)
    return docs


def process_eval_full_history_data():
    raws = []
    full_docs = io.load_json("data/history/document_history_all_v201.json")
    with open("data/history/full_history_v4.0.0.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row_relevant_ids = []
            label = row['label']
            for doc in full_docs:
                doc_label = [str(doc['grade']), doc['unit'], doc['title'], doc['section']]
                doc_label = "/".join(doc_label)
                if label == doc_label:
                    row_relevant_ids.append(doc['id'])
            if not row_relevant_ids:
                print(f"not found {label}")

            row['relevant_docs_id'] = row_relevant_ids

            raws.append(row)

    # Write to file
    io.write_json(raws, "cttgt2_v201.json")
    return raws


def process_eval_cttgt2_data():
    raws = []
    full_docs = io.load_json("data/history/document_history_all_v201.json")
    with open("data/history/cttgt2_v201.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row_relevant_ids = []
            label = row['label']
            for doc in full_docs:
                doc_id = doc['id']
                if row["context_id"] == doc_id:
                    row_relevant_ids.append(doc['id'])
            if not row_relevant_ids:
                print(f"not found {label}")

            row['relevant_docs_id'] = row_relevant_ids

            raws.append(row)

    # Write to file
    io.write_json(raws, "cttgt2_v201.json")
    return raws


if __name__ == '__main__':
    raws = process_eval_full_history_data()
