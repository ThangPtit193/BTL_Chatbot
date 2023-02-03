import math
import pprint
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
    # Remove duplicate mode: maybe: "EM", "fuzzy"
    remove_duplicate_doc = None
    # All positive intent in this list, will be used to generate as negative samples
    skipped_positives_of_intent = None
    # Split main intent and sub intent separated by "/"
    split_intent_by_slash = False

    # The intent name mapping using for generate from e2e result
    intent_field_map = {}

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
        # Filter positive intents
        if self.skipped_positives_of_intent:
            self._positive_intents = [positive_intent for positive_intent in self._positive_intents
                                      if positive_intent not in self.skipped_positives_of_intent]
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

        assert len(src_docs) > 0, "The src_docs can not be empty."
        assert len(tgt_docs) > 0, "The tgt_docs can not be empty."
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
        # raise bug if shape of embedding is not the same
        all_shape_tgt = set([doc.embedding.shape for doc in tgt_docs])
        all_shape_src = set([doc.embedding.shape for doc in src_docs])
        if len(all_shape_tgt) != 1 or len(all_shape_src) != 1:
            raise ValueError("You have some documents with different embedding shape. Please check your caching.")
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

        # Load document from e2e result
        self._load_from_e2e_result()
        self._load_from_ir_result()

        # Remove duplicate
        if self.remove_duplicate_doc:
            self._remove_duplicate()

        # Update the embeddings
        self._update_embeddings()

    def _remove_duplicate(self):
        """
        Remove duplicate documents
        Returns:

        """
        _logger.info("Removing duplicate documents")
        unique_docs = []
        duplicate_texts = []
        for doc in tqdm(self.documents.values()):
            if doc not in unique_docs:
                unique_docs.append(doc)
            else:
                duplicate_texts.append(doc.text)
        _logger.warning(f"Duplicate texts detected: {len(duplicate_texts)}")
        pprint.pprint(duplicate_texts)
        self.documents = {doc.id: doc for doc in unique_docs}

    def _load_from_e2e_result(self):
        """
        Load data from e2e result
        Returns:

        """
        _logger.info("Loading data from e2e result")
        from_e2e_config = self.config_parser.data_generation_config().get("FROM_E2E", {})

        if "e2e_eval_result_path" not in from_e2e_config:
            return

        e2e_result_paths = from_e2e_config.pop("e2e_eval_result_path")

        if not isinstance(e2e_result_paths, List):
            e2e_result_paths = [e2e_result_paths]
        for e2e_result_path in e2e_result_paths:
            rows = self._read_e2e_result(e2e_result_path, **from_e2e_config)
            for row in rows:
                query = convert_unicode(row["query"])
                label = row["label"]
                label = self.intent_field_map.get(label, label)
                # Get main intent and sub intent
                if self.split_intent_by_slash and "/" in label:
                    main_intent, sub_intent = label.split("/")
                else:
                    main_intent, sub_intent = label, None
                metadata = {
                    "main_intent": main_intent,
                    "sub_intent": sub_intent,
                    "type": row["doc_type"],
                    "intent": label,
                }

                doc = Document.from_dict({"text": query, **metadata})
                self.documents.update({doc.id: doc})
            _logger.info(f"Loaded {len(rows)} documents from {e2e_result_path}")

    def _load_from_ir_result(self):
        """
        Load data from e2e result
        Returns:

        """
        _logger.info("Loading data from e2e result")
        from_ir_eval_config = self.config_parser.data_generation_config().get("FROM_IR_EVAL", {})

        if "ir_eval_result_path" not in from_ir_eval_config:
            return
        ir_eval_result_path = from_ir_eval_config.pop("ir_eval_result_path")

        if not isinstance(ir_eval_result_path, List):
            ir_eval_result_path = [ir_eval_result_path]
        for e2e_result_path in ir_eval_result_path:
            data_save = {
                constants.KEY_POSITIVE: {},
                constants.KEY_NEGATIVE: {},
            }
            rows = self._read_ir_result(e2e_result_path, **from_ir_eval_config)
            for row in rows:
                text = convert_unicode(row["query"])
                label = row["label"]
                doc_type = row["doc_type"]
                # Get main intent and sub intent
                if self.split_intent_by_slash and "/" in label:
                    main_intent, sub_intent = label.split("/")
                else:
                    main_intent, sub_intent = label, None
                metadata = {
                    "main_intent": main_intent,
                    "sub_intent": sub_intent,
                    "type": doc_type,
                    "intent": label,
                }

                doc = Document.from_dict({"text": text, **metadata})
                self.documents.update({doc.id: doc})
                if label not in data_save[doc_type]:
                    data_save[doc_type][label] = []
                data_save[doc_type][label].append(text)
            _logger.info(f"Loaded {len(rows)} documents from {e2e_result_path}")

            # Save to file
            out_path = e2e_result_path[:-4] + "_converted.json"
            file_util.dump_obj_as_json_to_file(out_path, data_save)
            _logger.info(f"Saved to {out_path}")

    def _update_embeddings(self):
        if constants.KEY_EMBEDDER not in self.search_mode:
            _logger.info("Skip update embeddings because search mode is not 'embedder'")
            return
        _logger.info("Updating embeddings")
        # get all examples from documents
        examples = [doc.text for doc in self.documents.values()]
        doc_ids = self.documents.keys()
        examples_embedding = []
        # get all embeddings
        _logger.info(f"Get embeddings for '{len(examples)}' documents")
        iterators = batch(examples, n=self.embedding_batch_size)
        for _batch in tqdm(iterators, desc="Update embeddings", total=len(examples) / self.embedding_batch_size):
            embeddings = self.embedder.get_encodings(_batch)
            examples_embedding.extend(embeddings)
        # update embeddings
        for doc_id, embedding in zip(doc_ids, examples_embedding):
            self.documents[doc_id].embedding = embedding
        # dump cache
        self.embedder.dump_cache()

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
                # desc = f"{i}/{len(positives_data)} Loading positives {metadata[constants.KEY_INTENT]}"
                for _batch in iterators:
                    # if constants.KEY_EMBEDDER in self.search_mode:
                    #     embeddings = self.embedder.get_encodings(_batch)
                    #     docs = [Document.from_dict({"text": example, "embedding": embedding, **metadata})
                    #             for example, embedding in zip(_batch, embeddings)]
                    # else:
                    #     docs = [Document.from_dict({"text": example, **metadata}) for example in _batch]
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
                # desc = f"{i}/{len(negatives_data)} Loading negatives {metadata[constants.KEY_INTENT]}"
                for _batch in iterators:
                    # if constants.KEY_EMBEDDER in self.search_mode:
                    #     embeddings = self.embedder.get_encodings(_batch)
                    #     docs = [Document.from_dict({"text": example, "embedding": embedding, **metadata})
                    #             for example, embedding in zip(_batch, embeddings)]
                    # else:
                    #     docs = [Document.from_dict({"text": example, **metadata}) for example in _batch]
                    docs = [Document.from_dict({"text": example, **metadata}) for example in _batch]
                    docs_dict = {doc.id: doc for doc in docs}
                    self.documents.update(docs_dict)

    @staticmethod
    def _read_e2e_result(file_path: Text, **kwargs) -> List[Dict]:
        import csv

        with open(file_path, 'r') as f:
            # Read the csv file as a dictionary only field names: ['text', 'label]
            reader = csv.DictReader(f)
            rows = list(reader)
        assert "target_faq" in rows[0], "target_faq is not in the csv file"
        assert "predict_faq" in rows[0], "predict_faq is not in the csv file"
        assert "text" in rows[0], "text is not in the csv file"

        oos_intents = kwargs.get("oos_intents", [])
        eval_data = []
        for row in rows:
            target_faq = row['target_faq']
            if target_faq != row['predict_faq']:
                if target_faq in oos_intents:
                    doc_type = constants.KEY_NEGATIVE
                else:
                    doc_type = constants.KEY_POSITIVE
                eval_data.append({
                    "query": row['text'],
                    "label": target_faq,
                    "doc_type": doc_type,
                })
        return eval_data

    def _read_ir_result(self, file_path: Text, **kwargs) -> List[Dict]:
        """
        Read IR result
        Args:
            file_path:
            **kwargs:
                oos_intents: list of out of scope intents

        Returns: a list of row
            {
                "query": "query text",
                "label": "label",
            }

        """
        oos_intents = kwargs.get("oos_intents", [])
        rows = []
        # ir_data = file_util.load_json(file_path)
        ir_data = self._load_and_render_ir_data(file_path)
        query_data, relevant_data, predicted_labels, scores = \
            ir_data["query"], ir_data['most_relevant_docs'], ir_data['predicted_labels'], \
            ir_data['relevant_doc_scores']
        query_data = ir_data["query"]
        true_label_data = ir_data['label']
        predicted_label_data = ir_data['predicted_labels']
        relevant_data = ir_data['most_relevant_docs']
        relevant_score_data = ir_data['relevant_doc_scores']

        for index, queries in query_data.items():
            first_query = queries[0]
            true_label = true_label_data[index][0]
            score = relevant_score_data[index][0]

            # If top 1 is not the true label, then it is a false negative
            top_1_predicted_label = predicted_label_data[index][0]
            if top_1_predicted_label != true_label:
                if true_label not in oos_intents:
                    doc_type = constants.KEY_POSITIVE
                else:
                    doc_type = constants.KEY_NEGATIVE
                rows.append({
                    "query": first_query,
                    "label": true_label,
                    "doc_type": doc_type,
                })
            elif all([
                top_1_predicted_label == true_label,
                true_label not in oos_intents,
                score <= 0.95
            ]):
                rows.append({
                    "query": first_query,
                    "label": true_label,
                    "doc_type": constants.KEY_POSITIVE,
                })
            # Add relevant data to rows
            # for r_idx, re_text in enumerate(relevant_data[index]):
            #     rel_label = predicted_labels[index][r_idx]
            #     # If the label is not the true label, and the score is greater than 0.85, then it is a false positive
            #     rel_score = relevant_score_data[index][r_idx]
            #     if rel_label != true_label and rel_score > 0.85:
            #         if rel_label not in oos_intents:
            #             doc_type = constants.KEY_POSITIVE
            #         else:
            #             doc_type = constants.KEY_NEGATIVE
            #         rows.append({
            #             "query": re_text,
            #             "label": rel_label,
            #             "doc_type": doc_type,
            #         })
            #     elif rel_label == true_label and rel_score <= 0.9 and rel_label not in oos_intents:
            #         rows.append({
            #             "query": re_text,
            #             "label": rel_label,
            #             "doc_type": constants.KEY_POSITIVE,
            #         })

        return rows

    @staticmethod
    def _load_and_render_ir_data(file_path: Text):
        ir_data = file_util.load_json(file_path)

        for name, data in ir_data.items():
            for index, meta in data.items():
                data[index] = list(meta.values())
        return ir_data

    def _get_metadata(self, positive_data: Dict, data_type: Text):
        """
        Get metadata from positive data
        Args:
            positive_data:
            data_type: Maybe positive or negative

        Returns:

        """
        # Get main intent and sub intent
        if self.split_intent_by_slash and "/" in positive_data["intent"]:
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

    def _get_main_sub_intent(self, intent: str):
        """
        Get main intent and sub intent
        Args:
            intent:

        Returns:

        """
        if self.split_intent_by_slash and "/" in intent:
            main_intent, sub_intent = intent.split("/")
        else:
            main_intent, sub_intent = intent, None
        return main_intent, sub_intent
