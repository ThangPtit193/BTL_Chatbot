import os.path

import pandas as pd

from torch.utils.data import IterableDataset
from sentence_transformers import InputExample
from venus.utils.utils import load_json

class QuadrupletDataset(IterableDataset):
    def __init__(
            self,
            quadruplet_examples=None,
            query_key: str = "query",
            pos_key: str = "pos",
            neg_key_1: str = "neg_1",
            neg_key_2: str = "neg_2"

    ):
        self.quadruplet_examples = quadruplet_examples if quadruplet_examples else []
        self.query_key = query_key
        self.pos_key = pos_key
        self.neg_key_1 = neg_key_1
        self.neg_key_2 = neg_key_2

    def __iter__(self):
        for example in self.quadruplet_examples:
            query_text = example[self.query_key]
            pos_text = example[self.pos_key]
            neg_text_1 = example[self.neg_key_1]
            neg_text_2 = example[self.neg_key_2]
            yield InputExample(texts=[query_text, pos_text, neg_text_1, neg_text_2])

    def __len__(self):
        return len(self.quadruplet_examples)

    @classmethod
    def from_quadruples_ids_df(
            cls,
            queries,
            corpus,
            quadruples,
            query_col: str = 'query_id',
            pos_col: str = 'pos_id',
            neg_1_col: str = 'neg_1_id',
            neg_2_col: str = 'neg_2_id'
    ):
        quadruplet_examples = []

        for _, row in quadruples.iterrows():
            query = queries[str(row[query_col])]
            pos = corpus[str(row[pos_col])]
            neg_1 = corpus[str(row[neg_1_col])]
            neg_2 = corpus[str(row[neg_2_col])]
            quadruplet_examples.append({'query': query, 'pos': pos, 'neg_1': neg_1, 'neg_2': neg_2})
        return cls(quadruplet_examples, 'query', 'pos', 'neg_1', 'neg_2')

    @classmethod
    def from_quadruples_ids_csv(
            cls,
            query_file,
            corpus_file,
            quadruples_file,
            query_col: str = 'query_id',
            pos_col: str = 'pos_id',
            neg_1_col: str = 'neg_1_id',
            neg_2_col: str = 'neg_2_id'
    ):
        queries = load_json(query_file)
        corpus = load_json(corpus_file)
        quadruples = pd.read_csv(quadruples_file)
        return cls.from_quadruples_ids_df(queries, corpus, quadruples, query_col, pos_col, neg_1_col, neg_2_col)

    def load_from_file(self, file_path):
        if os.path.exists(file_path):
            self.quadruple_examples = load_json(file_path)['data']



