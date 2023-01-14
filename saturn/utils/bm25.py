from collections import Counter, defaultdict
import numpy as np


def default_token_data():
    return {
        'index': [],
        'score': []
    }


class BM25:
    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.token_score = defaultdict(default_token_data)  # token -> score contribution of token for each doc
        self.doc_len = []
        self.idf = {}

        nd = self._initialize(corpus)
        self._calc_idf(nd)
        self._convert_token_score()

    def _initialize(self, corpus):
        nd = Counter()  # word -> number of documents with word
        num_doc = 0
        for i, document in enumerate(corpus):
            self.doc_len.append(len(document))
            num_doc += len(document)

            # Count frequencies of each token in document
            frequencies = Counter(document)

            # Append token_freq index and frequency for each token
            for token, freq in frequencies.items():
                self.token_score[token]['index'].append(i)
                self.token_score[token]['score'].append(freq)

            nd.update(frequencies.keys())

        self.doc_len = np.array(self.doc_len)
        self.token_score = dict(self.token_score)
        for token_data in self.token_score.values():
            token_data['index'] = np.array(token_data['index'], dtype=np.int32)
            token_data['score'] = np.array(token_data['score'], dtype=np.float64)

        self.avgdl = num_doc / self.corpus_size
        return dict(nd)

    def _calc_idf(self, nd):
        raise NotImplementedError

    def _convert_token_score(self):
        raise NotImplementedError

    def get_scores(self, query):
        raise NotImplementedError


class BM25Plus(BM25):
    def __init__(self, corpus, k1=1.5, b=0.75, delta=1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus)

    def _calc_idf(self, nd):
        for token, freq in nd.items():
            self.idf[token] = np.log((self.corpus_size + 1) / freq)

    def _convert_token_score(self):
        for data in self.token_score.values():
            freq = data['score']
            doc_len = self.doc_len[data['index']]
            score = freq * (self.k1 + 1) / (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + freq)
            data['score'] = score

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        for q in query:
            q_data = self.token_score.get(q)
            if q_data is None:
                continue
            idf = self.idf[q]
            score[q_data['index']] += idf * q_data['score']
            score += idf * self.delta
        return score

    # def is_same_intent_of_query(self, query,  full_doc: str):

