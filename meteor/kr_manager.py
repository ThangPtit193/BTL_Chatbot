from meteor.utils.config_parser import ConfigParser
from comet.lib.helpers import get_module_or_attr
from comet.lib import file_util
from typing import *
from comet.utilities.utility import convert_unicode
from meteor.components.utils.document import Document, EvalResult
import os

if TYPE_CHECKING:
    from meteor.components.embeddings.embedding_models import SentenceEmbedder


class KRManager:
    def __init__(self, config_path: str):
        self.config_parser = ConfigParser(config_path)
        self._embedder: Optional[SentenceEmbedder] = None
        self._corpus_docs: Optional[List[Document]] = None
        self._query_docs: Optional[List[Document]] = None

    @property
    def embedder(self):
        if not self._embedder:
            embedder_config = self.config_parser.embedder_config()
            class_name = embedder_config.pop("class")
            module_name = embedder_config.pop("package")
            self._embedder = get_module_or_attr(module_name, class_name)(**embedder_config)
        return self._embedder

    @property
    def corpus_docs(self):
        if not self._corpus_docs:
            eval_config = self.config_parser.eval_config()

            if "corpus_name_or_path" not in eval_config:
                raise FileNotFoundError("No corpus path provided in config file")
            self._corpus_docs = self._load_docs(eval_config['corpus_name_or_path'])
        return self._corpus_docs

    @property
    def query_docs(self):
        if not self._query_docs:
            eval_config = self.config_parser.eval_config()
            if "query_name_or_path" not in eval_config:
                raise FileNotFoundError("No query path provided in config file")
            self._query_docs = self._load_docs(eval_config['query_name_or_path'])
        return self._query_docs

    def train_embedder(self):
        trainer_config = self.config_parser.trainer_config()

        self.embedder.train(trainer_config)

        # TODO save the model and upload it to axiom
        # Your code here

    def evaluate_embedder(self):
        model_name_or_paths = self.config_parser.eval_config()['model_name_or_path']
        if isinstance(model_name_or_paths, str):
            model_name_or_paths = [model_name_or_paths]
        evaluation_results: Dict[str, List[EvalResult]] = {}
        for model_name_or_path in model_name_or_paths:
            name = os.path.basename(model_name_or_path)
            evaluation_results[name] = []
            self.embedder.load_model(cache_path=name, pretrained_name_or_abspath=model_name_or_path)
            # TODO evaluate
            tgt_docs = [doc.text for doc in self.corpus_docs]
            src_docs = [doc.text for doc in self.query_docs]
            similarity_data = self.embedder.find_similarity(src_docs, tgt_docs, _no_sort=True)
            evaluation_results[name].extend(
                self._extract_eval_result(self.query_docs, self.corpus_docs, similarity_data))

        # TODO Something for calculating the mean of the results

        # TODO Export the results
        return evaluation_results

    def _extract_eval_result(
        self, src_docs: List[Document], tgt_docs, similarity_data_2d: List[List[Tuple[Text, float]]]
    ) -> List[EvalResult]:
        eval_results = []
        for src_doc, similarities in zip(src_docs, similarity_data_2d):
            indices = self._arg_sort(similarities)
            # TODO Compute the rr score
            rr_score = None
            # TODO Compute the app score
            ap_score = None

            # Get top_k relevant docs from indices
            top_k_relevant_docs = []
            num_srcs = len(src_docs)
            for i in indices[:num_srcs]:
                top_k_relevant_docs.append(tgt_docs[i])

            eval_result = EvalResult(
                query=src_doc.text,
                query_id=src_doc.id,
                rr_score=rr_score,
                ap_score=ap_score,
                top_k_relevant=num_srcs,
                most_relevant_docs=top_k_relevant_docs

            )
            eval_results.append(eval_result)
        return eval_results

    @staticmethod
    def _arg_sort(similarities: List[Tuple[Text, float]]) -> List[int]:
        """

        :param similarities:
        :return:
        """
        import numpy as np
        scores = [score for _, score in similarities]
        indices = np.argsort(scores)[::-1].tolist()
        return indices

    @staticmethod
    def _load_docs(path: str) -> List[Document]:
        """
        Load documents from a file or a directory
        :param path:
        :return:
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File {path} does not exist")

        data_docs = file_util.load_json(path)
        docs = []
        for unique_intent, query_list in data_docs.items():
            num_relevant = len(query_list)
            for query in query_list:
                docs.append(Document(
                    text=convert_unicode(query),
                    label=unique_intent,
                    num_relevant=num_relevant,
                ))
        return docs
