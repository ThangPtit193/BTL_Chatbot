import math
import random
from functools import lru_cache
from typing import *
from saturn.abstract_method.staturn_abstract import SaturnAbstract
import numpy as np
import pydash as ps
from comet.components.embeddings.embedding_models import BertEmbedder
from comet.lib import file_util, decorator, logger
from comet.utilities.utility import convert_unicode
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from saturn.data_generation import constants
from saturn.utils.config_parser import ConfigParser
from saturn.data_generation.document_store.document import Document
from saturn.data_generation.document_store.utils import fast_argsort, fast_argsort_bottleneck
from saturn.utils.bm25 import BM25Plus

_logger = logger.get_logger(__name__)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def cosin_similarity_multi(query, vectors):
    _scores = vectors.dot(query) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
    return _scores


class InmemoryDocumentStore(SaturnAbstract):
    max_anchor_positive_pairs_per_intent = -1
    embedding_batch_size = 1000
    neg_sim_batch_size = 10000
    search_mode = "random"

    def __init__(self, config_parser: ConfigParser, use_embedding=True):
        """
        In-memory document store for testing purposes.
        Args:
            use_embedding: If True, the document store will store the embeddings of the documents.
            config_parser: The ConfigParser object.
        """
        super(InmemoryDocumentStore, self).__init__(config_parser)
        self.documents: Dict[Text, Document] = {}
        self.use_embedding = use_embedding
        self._embedder = None
        self._positive_intents: List[Text] = []
        self.full_docs: List[Document] = []
        self._bm25 = None
        self.start_index_pos: int = None
        self.end_index_post: int = None
        self.initialize()
        # self._vectorize_bm25()

    def initialize(self):
        general_cfg = self.config_parser.data_generation_config()
        for key, val in general_cfg.items():
            if hasattr(self, key):
                setattr(self, key, val)

    @property
    def bm25(self):
        if not self._bm25:
            corpus = [doc.text.split(' ') for doc in self.full_docs]
            self._bm25 = BM25Plus(corpus)
        return self._bm25

    @property
    def embedder(self):
        if not self._embedder and self.use_embedding:
            data_generation_cfg = self.config_parser.data_generation_config()
            embedder_config = data_generation_cfg.get("EMBEDDER")
            if not embedder_config:
                raise ValueError("Can not find embedder config")
            self._embedder = BertEmbedder(
                **embedder_config
            )
            self._embedder.load_cache()
        return self._embedder

    @property
    def positive_intents(self):
        if not self._positive_intents and self.documents:
            positive_docs = self.get_documents(type=constants.KEY_POSITIVE)
            positive_documents = [doc.meta for doc in positive_docs]
            self._positive_intents = ps.uniq(ps.map_(positive_documents, constants.KEY_INTENT))
        return self._positive_intents

    @decorator.performance
    def build_documents(self):
        """

        Returns:

        """
        desc = "Building positives and negatives ids"
        self.full_docs = self.get_documents()

        for i, positive_intent in enumerate(tqdm(self.positive_intents, desc=desc)):
            main_intent, sub_intent = self._get_main_sub_intent(positive_intent)
            docs = self.get_documents(type=constants.KEY_POSITIVE, main_intent=main_intent, sub_intent=sub_intent)
            self.start_index_pos = self.full_docs.index(docs[0])
            self.end_index_post = self.full_docs.index(docs[-1])
            # Build positives ids
            for idx, doc in enumerate(docs):
                self.documents[doc.id].positive_ids = [_doc.id for _doc in docs[idx:]]

            # Build negatives ids
            iterators = batch(docs, self.neg_sim_batch_size)
            for idx, _batch in enumerate(iterators):
                # Build the negative docs
                self._build_negative_ids(_batch, positive_intent)

        self._post_processing_documents()

    @decorator.performance
    def _post_processing_documents(self):
        for doc_id, doc in self.documents.items():
            num_positives_ids = len(doc.positive_ids)
            for search_mode, negative_ids in doc.negatives_ids.items():
                if len(negative_ids) < num_positives_ids:
                    rest_num = num_positives_ids - len(negative_ids)
                    tgt_docs = self.get_documents(ignored_intents=[doc.meta[constants.KEY_INTENT]])
                    random_most_docs = random.choices(tgt_docs, k=rest_num)
                    doc.negatives_ids[search_mode].extend([doc.id for doc in random_most_docs])
                else:
                    doc.negatives_ids[search_mode] = doc.negatives_ids[search_mode][:num_positives_ids]

    def _build_negative_ids(
        self, _batch: Union[Document, List[Document]], positive_intent: Text
    ) -> Dict[Text, List[List[Document]]]:
        negative_combine_cfg = self.config_parser.data_generation_config().get("COMBINER", {})
        ratio_info = negative_combine_cfg.get(positive_intent)
        if ratio_info and isinstance(ratio_info, List):
            # Convert list of dict to dict
            ratio_info = {k: v for d in ratio_info for k, v in d.items()}

        if not ratio_info:
            # If no ratio info, use the random mode to select negative
            main_intent, _ = self._get_main_sub_intent(positive_intent)
            tgt_docs = self.get_documents(ignored_intents=main_intent)
            negatives_map = self.get_the_most_similarity_docs(
                _batch, tgt_docs, mode=self.search_mode
            )
        else:
            sum_weight = self._cal_sum_weight(ratio_info)
            multi_negatives_map = []
            weights = []
            for negative_intent in ratio_info.keys():
                weights.append(ratio_info[negative_intent]["weight"] / sum_weight)
                tgt_docs = self.get_documents(intent=negative_intent)
                if not tgt_docs:
                    raise ValueError(f"Can not find any negative docs for {negative_intent}")

                _negatives_map = self.get_the_most_similarity_docs(
                    _batch, tgt_docs, mode=self.search_mode
                )
                multi_negatives_map.append(_negatives_map)
            negatives_map = self._resolve_negatives_map(_batch, multi_negatives_map, weights)
        for idx, doc in enumerate(_batch):
            for search_mode, batch_negative_docs in negatives_map.items():
                doc.negatives_ids[search_mode] = [negative_doc.id for negative_doc in batch_negative_docs[idx]]

        return negatives_map

    @staticmethod
    def _resolve_negatives_map(
        _batch: List[Document], multi_negatives_map: List[Dict[Text, List[List[Document]]]], weights: List[int]
    ) -> Dict[Text, List[List[Document]]]:
        # Resolve the negatives map
        resolved_negatives_map = {}
        for ratio, _negatives_map in zip(weights, multi_negatives_map):
            if not resolved_negatives_map:
                modes = _negatives_map.keys()
                resolved_negatives_map = {mode: [[] for _ in range(len(_negatives_map[mode]))] for mode in modes}
            for mode, batch_negatives_docs in _negatives_map.items():
                for idx, negatives_docs in enumerate(batch_negatives_docs):
                    num_samples = math.ceil(len(negatives_docs) * ratio)
                    negatives_docs = negatives_docs[:num_samples]
                    resolved_negatives_map[mode][idx].extend(negatives_docs)
        return resolved_negatives_map

    @staticmethod
    def _cal_sum_weight(ratio_info: Dict):
        sum_weight = 0
        for _, ratio in ratio_info.items():
            sum_weight += ratio[constants.KEY_WEIGHT]
        return sum_weight

    # @decorator.performance
    def get_the_most_similarity_docs(
        self, src_docs: Union[Document, List[Document]], tgt_docs: List[Document], mode: Text = "bm25"
    ) -> Dict[Text, List[List[Document]]]:
        """
        Find the most similar documents in the document store.
        Args:
            src_docs:
            tgt_docs: The text to search for.
            mode: The mode to use. Can be "BM25" or "cosine".

        Returns: The most similar documents.

        """
        negatives_map = {}
        if all([constants.KEY_EMBEDDER not in mode, constants.KEY_RANDOM not in mode, constants.KEY_BM25 not in mode]):
            raise ValueError(f"Can not find mode {mode}")
        top_k = max([len(doc.positive_ids) for doc in src_docs])
        if constants.KEY_EMBEDDER in mode:
            embedder_most_docs = self._most_similarity_embedder(src_docs, tgt_docs, top_k=top_k)
            negatives_map.update({constants.KEY_EMBEDDER: embedder_most_docs})
        if constants.KEY_BM25 in mode:
            bm_25_most_docs = self._most_similarity_bm25(src_docs, tgt_docs, self.full_docs, top_k=top_k)
            negatives_map.update({constants.KEY_BM25: bm_25_most_docs})
        if constants.KEY_RANDOM in mode:
            random_most_docs = self._most_similarity_random(src_docs, tgt_docs, top_k=top_k)
            negatives_map.update({constants.KEY_RANDOM: random_most_docs})
        return negatives_map

    @lru_cache(maxsize=50)
    def get_documents(self, ignored_intents: Union[Text, List[Text]] = None, **kwargs) -> List[Document]:
        """
        Get the best negative documents.
        Args:
            ignored_intents: The intents to ignore.
            **kwargs: The filter here

        Returns:

        """
        if ignored_intents and isinstance(ignored_intents, Text):
            ignored_intents = [ignored_intents]

        documents = []
        if not kwargs and not ignored_intents:
            return list(self.documents.values())

        for doc in self.documents.values():
            if ignored_intents and doc.meta[constants.KEY_MAIN_INTENT] in ignored_intents:
                continue

            is_matched = True
            for key, val in kwargs.items():
                if doc.meta.get(key) != val:
                    is_matched = False
                    break
            if is_matched:
                documents.append(doc)
        return documents

    @staticmethod
    def _most_similarity_embedder(
        src_docs: Union[Document, List[Document]], tgt_docs: List[Document], top_k: int = 20
    ) -> List[List[Document]]:
        """
        Find the most similar documents in the document store using the embedder.
        Args:
            src_docs: The text to search for.
            top_k: The number of documents to return.
        """
        if not isinstance(src_docs, List):
            src_docs = [src_docs]
        # Method 1: Naive cosine similarity
        # t0 = time.time()
        # vectors_store = np.vstack([doc.embedding for doc in tgt_docs])
        # old_scores = []
        # old_score_indices = []
        # old_sorted_docs = []
        # for src_doc in src_docs:
        #     scores = cosin_similarity_multi(src_doc.embedding, vectors_store)
        #     old_scores.append(scores)
        #     _score_indices = np.argsort(scores)[::-1][:top_k]
        #     old_score_indices.append(_score_indices)
        #     old_sorted_docs.append([tgt_docs[idx] for idx in _score_indices])

        # Method 2: using numpy and batching
        # vectors_tgt = np.array([doc.embedding for doc in tgt_docs])
        # vector_srcs = np.array([src_doc.embedding for src_doc in src_docs])
        # t0 = time.time()
        # new_scores = cosine_similarity(vector_srcs, vectors_tgt)
        # t1 = time.time()
        # new_score_indices = np.argsort(new_scores, axis=1)[:, ::-1][:, :top_k]
        # print(new_scores.shape)
        # print(f"Time cosine_similarity cost: {t1 - t0}")
        # print(f"Time np.argsort cost: {time.time() - t1}")
        # tops = [len(doc.positive_ids) for doc in src_docs]
        # new_sorted_docs = [[tgt_docs[idx] for idx in _score_indices][:top_k]
        #                    for top_k, _score_indices in zip(tops, new_score_indices)]

        # Method 3: Using pytorch batching
        # scores = self.embedder.pytorch_cos_sim(vector_srcs, vectors_tgt, return_tensor=True)
        # print(f"pytorch_cos_sim cost {time.time() - t0} seconds")
        # score_indices = torch.argsort(scores, dim=1, descending=True)[:, :top_k]
        # Method 4: Using custom argsort
        vectors_tgt = np.array([doc.embedding for doc in tgt_docs])
        vector_srcs = np.array([src_doc.embedding for src_doc in src_docs])
        new_scores = cosine_similarity(vector_srcs, vectors_tgt)
        new_score_indices = fast_argsort_bottleneck(new_scores, axis=1, top_k=top_k)
        tops = [len(doc.positive_ids) for doc in src_docs]
        new_sorted_docs = [[tgt_docs[idx] for idx in _score_indices][:top_k]
                           for top_k, _score_indices in zip(tops, new_score_indices)]

        return new_sorted_docs

    def _most_similarity_bm25(self,
                              src_docs: Union[Document, List[Document]],
                              tgt_docs: List[Document],
                              full_docs: List[Document],
                              top_k: int = 20,
                              **kwargs) -> List[List]:
        """

        Args:
            src_docs: Document list
            tgt_docs: Document list
            top_k:  The number of documents to return

        Returns:  List with top_k docs relevant to each query

        """

        new_sorted_docs, idx, matrix_score = [], [], []

        # src index
        # start_index_src = full_docs.index(src_docs[0])
        # end_index_src = full_docs.index(src_docs[-1])

        # tgt index
        start_index_tgt = full_docs.index(tgt_docs[0])
        end_index_tgt = full_docs.index(tgt_docs[-1])

        # set origin score with zero array [0,0,0...,0]
        origin_score = np.zeros(len(full_docs))

        if start_index_tgt < self.start_index_pos < end_index_tgt:
            origin_score[self.start_index_pos:self.end_index_post + 1] = float('-inf')
            origin_score[start_index_tgt:self.start_index_pos] = 1
            origin_score[self.end_index_post + 1:end_index_tgt + 1] = 1
        else:
            origin_score[:start_index_tgt] = float('-inf')
            origin_score[end_index_tgt + 1:] = float('-inf')
            origin_score[start_index_tgt:end_index_tgt + 1] = 1

        for doc in src_docs:
            query = doc.text.split(' ')
            scores = self.bm25.get_scores(query)
            scores *= origin_score
            matrix_score.append(scores)

        new_scores = np.reshape(matrix_score, (len(src_docs), -1))
        new_score_indices = fast_argsort(new_scores, axis=1, top_k=top_k)
        new_sorted_docs = [[full_docs[idx] for idx in new_score_indices[i]]
                           for i in range(new_score_indices.shape[0])]

        return new_sorted_docs

    @staticmethod
    def _most_similarity_random(_batch: List[Document], tgt_docs: List[Document], top_k: int = 20):
        return [random.choices(tgt_docs, k=len(doc.positive_ids)) for doc in _batch]

    def load_document(self, data_path: Text):
        all_files = file_util.get_all_files_in_directory(data_path, extensions=(".yaml", ".yml"))
        for file in all_files:
            _logger.info(f"Loading file {file}")
            data = file_util.load_yaml_fast(file)
            self._load(data)

    @decorator.performance
    def _load(self, data: Dict):
        """
        Load positive or negative data
        Args:
            data:

        Returns:

        """
        if constants.KEY_POSITIVES in data:
            positives_data = data[constants.KEY_POSITIVES]
            _logger.info(f"Loading '{len(positives_data)}' positives data, be patient ....")

            for i, positive_data in enumerate(positives_data):
                examples = positive_data.pop("examples")
                examples = [convert_unicode(example) for example in examples]
                metadata = self._get_metadata(positive_data, constants.KEY_POSITIVE)

                # Get main intent and sub intent
                # if "/" in metadata["intent"]:
                #     main_intent, sub_intent = positive_data["intent"].split("/")
                # else:
                #     main_intent, sub_intent = positive_data["intent"], None
                # metadata = {
                #     "main_intent": main_intent,
                #     "sub_intent": sub_intent,
                #     "type": constants.KEY_POSITIVES,
                #     **positive_data
                # }
                # metadata.update({"type": constants.KEY_POSITIVE})
                iterators = batch(examples, n=self.embedding_batch_size)
                desc = f"{i}/{len(positives_data)} Loading positives {metadata[constants.KEY_INTENT]}"
                for _batch in tqdm(iterators, desc=desc, total=len(examples) / self.embedding_batch_size):

                    if constants.KEY_EMBEDDER in self.search_mode:
                        embeddings = self.embedder.get_encodings(_batch)
                        docs = [Document.from_dict({"text": example, "embedding": embedding, **metadata})
                                for example, embedding in zip(_batch, embeddings)]
                    else:
                        docs = [Document.from_dict({"text": example, **metadata}) for example in _batch]
                    docs_dict = {doc.id: doc for doc in docs}
                    self.documents.update(docs_dict)
        elif constants.KEY_NEGATIVES in data:
            negatives_data = data[constants.KEY_NEGATIVES]
            for i, negative_data in enumerate(negatives_data):
                examples = negative_data.pop("examples")
                examples = [convert_unicode(example) for example in examples]
                metadata = self._get_metadata(negative_data, constants.KEY_NEGATIVE)
                # metadata.update({"type": constants.KEY_NEGATIVE})
                iterators = batch(examples, n=self.embedding_batch_size)
                desc = f"{i}/{len(negatives_data)} Loading negatives {metadata[constants.KEY_INTENT]}"
                for _batch in tqdm(iterators, desc=desc, total=len(examples) / self.embedding_batch_size):
                    if constants.KEY_EMBEDDER in self.search_mode:
                        embeddings = self.embedder.get_encodings(_batch)
                        docs = [Document.from_dict({"text": example, "embedding": embedding, **metadata})
                                for example, embedding in zip(_batch, embeddings)]
                    else:
                        docs = [Document.from_dict({"text": example, **metadata}) for example in _batch]
                    docs_dict = {doc.id: doc for doc in docs}
                    self.documents.update(docs_dict)
        self.embedder.dump_cache()

    @staticmethod
    def _get_metadata(positive_data: Dict, data_type: Text):
        """
        Get metadata from positive data
        Args:
            positive_data:
            data_type: Maybe positive or negative

        Returns:

        """
        # Get main intent and sub intent
        if "/" in positive_data["intent"]:
            main_intent, sub_intent = positive_data["intent"].split("/")
        else:
            main_intent, sub_intent = positive_data["intent"], None
        metadata = {
            "main_intent": main_intent,
            "sub_intent": sub_intent,
            "type": data_type,
            **positive_data
        }
        return metadata

    @staticmethod
    def _get_main_sub_intent(intent: str):
        """
        Get main intent and sub intent
        Args:
            intent:

        Returns:

        """
        if "/" in intent:
            main_intent, sub_intent = intent.split("/")
        else:
            main_intent, sub_intent = intent, None
        return main_intent, sub_intent
