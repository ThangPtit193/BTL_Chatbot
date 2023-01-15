import os.path
import random

import pandas as pd

from torch.utils.data import IterableDataset
from sentence_transformers import InputExample
from venus.utils.utils import load_json


class TripletsDataset(IterableDataset):
    def __init__(
        self,
        triplet_examples=None,
        query_key: str = "query",
        pos_key: str = "pos",
        neg_key: str = "neg",
        shuffle: bool = False,
    ):
        self.triplet_examples = triplet_examples if triplet_examples else []
        self.query_key = query_key
        self.pos_key = pos_key
        self.neg_key = neg_key
        if shuffle:
            random.shuffle(self.triplet_examples)

    def __iter__(self):
        for example in self.triplet_examples:
            print(example)
            query_text = example[self.query_key]
            pos_text = example[self.pos_key]
            neg_text = example[self.neg_key]
            yield InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.triplet_examples)

    @classmethod
    def from_triplets_ids_df(
        cls,
        queries,
        corpus,
        triplets,
        query_col: str = 'query_id',
        pos_col: str = 'pos_id',
        neg_col: str = 'neg_id'
    ):
        triple_examples = []

        for _, row in triplets.iterrows():
            query = queries[str(row[query_col])]
            pos = corpus[str(row[pos_col])]
            neg = corpus[str(row[neg_col])]
            triple_examples.append({'query': query, 'pos': pos, 'neg': neg})
        return cls(triple_examples, 'query', 'pos', 'neg')

    @classmethod
    def from_triplets_ids_csv(
        cls,
        query_file,
        corpus_file,
        triplets_file,
        query_col: str = 'query_id',
        pos_col: str = 'pos_id',
        neg_col: str = 'neg_id'
    ):
        queries = load_json(query_file)
        corpus = load_json(corpus_file)
        triplets = pd.read_csv(triplets_file)
        return cls.from_triplets_ids_df(queries, corpus, triplets, query_col, pos_col, neg_col)

    def load_from_file(self, file_path):
        if os.path.exists(file_path):
            self.triplet_examples = load_json(file_path)['data']
