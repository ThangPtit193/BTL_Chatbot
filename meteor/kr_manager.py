import datetime
from pathlib import Path
import time
from time import perf_counter

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from meteor.constants import INDEX_RESULT_FILES
from meteor.utils.config_parser import ConfigParser
from comet.lib.helpers import get_module_or_attr
from comet.lib import file_util, logger
from typing import *
from comet.utilities.utility import convert_unicode
from meteor.components.utils.document import Document, EvalResult
import os

if TYPE_CHECKING:
    from meteor.components.embeddings.embedding_models import SentenceEmbedder

_logger = logger.get_logger(__name__)


class KRManager:
    def __init__(self, config_path: str):
        self.config_parser = ConfigParser(config_path)
        self._embedder: Optional[SentenceEmbedder] = None
        self._corpus_docs: Optional[List[Document]] = None
        self._query_docs: Optional[List[Document]] = None
        self._model_name_or_path: Optional[Union[str, List[str]]] = None
        self.retriever_type: Optional[Union[str, List]] = "embedding"
        self.document_store_type: Optional[Union[str, List]] = "memory"

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
            self._query_docs = self._load_docs(eval_config['query_name_or_path'], eval_config['corpus_name_or_path'])
        return self._query_docs

    @property
    def model_name_or_path(self):
        if not self._model_name_or_path:
            eval_config = self.config_parser.eval_config()
            if "model_name_or_path" not in eval_config:
                raise FileNotFoundError("No model name or path provided in config file")
            self._model_name_or_path = eval_config.get("model_name_or_path")

        if isinstance(self._model_name_or_path, str):
            self._model_name_or_path = [self._model_name_or_path]
            return self._model_name_or_path
        elif isinstance(self._model_name_or_path, list):
            return self._model_name_or_path
        else:
            raise ValueError("model_name_or_path should be a string or a list of string")

    def train_embedder(self):
        trainer_config = self.config_parser.trainer_config()

        self.embedder.train(trainer_config)

        # TODO save the model and upload it to axiom
        # Your code here

    def evaluate_embedder(self, save_markdown: Optional[bool] = True):
        retriever_results = []
        model_name_or_paths = self.config_parser.eval_config()['model_name_or_path']
        if isinstance(model_name_or_paths, str):
            model_name_or_paths = [model_name_or_paths]
        evaluation_results: Dict[str, List[EvalResult]] = {}
        for model_name_or_path in model_name_or_paths:
            name = os.path.basename(model_name_or_path)
            evaluation_results[name] = []
            self.embedder.load_model(cache_path=name, pretrained_name_or_abspath=model_name_or_path)

            tic = perf_counter()
            tgt_docs = [doc.text for doc in self.corpus_docs]
            src_docs = [doc.text for doc in self.query_docs]
            similarity_data = self.embedder.find_similarity(src_docs, tgt_docs, _no_sort=True)
            toc = perf_counter()
            retriever_time = toc - tic

            evaluation_results[name].extend(
                self._extract_eval_result(self.query_docs, self.corpus_docs, similarity_data))

            df = pd.DataFrame(evaluation_results[name])
            self.save_detail_report(out_dir='reports', model_name=name, df=df)

            records = {
                "model_name": name,
                "query_numbers": len(src_docs),
                "retriever_time": retriever_time,
                "query_per_second": retriever_time / len(src_docs),
                "map": df["ap_score"].mean(),
                "mrr": df["rr_score"].mean(),
                "date_time": datetime.datetime.now()
            }
            retriever_results.append(records)

        retriever_df = pd.DataFrame.from_records(retriever_results)
        retriever_df = retriever_df.sort_values(by="map")
        retriever_df.to_csv(INDEX_RESULT_FILES)

        if save_markdown:
            md_file = INDEX_RESULT_FILES.replace(".csv", ".md")
            with open(md_file, "w") as f:
                f.write(str(retriever_df.to_markdown()))
        time.sleep(10)

        return evaluation_results

    def save_detail_report(self, out_dir: Union[str, Path], model_name: str, df: DataFrame):
        """
        Saves the evaluation result.
        The result of each node is saved in a separate csv with file name {node_name}.csv to the out_dir folder.

        :param out_dir: Path to the target folder the csvs will be saved.
        :param model_name: Model name to benchmark
        :param df: Evaluation resulr
        """
        out_dir = out_dir if isinstance(out_dir, Path) else Path(out_dir)
        _logger.info("Saving evaluation results to %s", out_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        target_path = out_dir / f"{model_name}.csv"
        df.to_csv(target_path)

    def _extract_eval_result(
            self, src_docs: List[Document], tgt_docs, similarity_data_2d: List[List[Tuple[Text, float]]]
    ) -> List[dict]:
        eval_results = []

        for src_doc, similarities in zip(src_docs, similarity_data_2d):
            indices = self._arg_sort(similarities)
            rr_score = 0
            ap_score = 0

            # Get top_k relevant docs from indices
            top_k_relevant_docs = []
            num_srcs = len(src_docs)
            for i in indices[:num_srcs]:
                top_k_relevant_docs.append(tgt_docs[i].text)

            ground_truth = [doc.text for doc in self.corpus_docs if doc.label == src_doc.label]
            ap = 0
            for idx, relevant_doc in enumerate(top_k_relevant_docs):
                if relevant_doc in ground_truth:
                    ap += 1
                    if ap == 1:
                        rr_score = 1
                    ap_score += ap / (int(idx) + 1)

            eval_result = EvalResult(
                query=src_doc.text,
                query_id=src_doc.id,
                rr_score=rr_score,
                ap_score=round(ap_score / (len(ground_truth)), 2),
                top_k_relevant=len(top_k_relevant_docs),
                golden_docs=ground_truth,
                most_relevant_docs=top_k_relevant_docs

            )
            eval_results.append(eval_result.to_dict())
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
    def _load_docs(path: str, corpus_path: Optional[str] = None) -> List[Document]:
        """
        Load documents from a file or a directory
        :param path:
        :return:
        """
        corpus: dict = {}
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File {path} does not exist")

        data_docs = file_util.load_json(path)
        if corpus_path is not None:
            if not os.path.isfile(corpus_path):
                raise FileNotFoundError(f"File {corpus_path} does not exist")
            corpus = file_util.load_json(corpus_path)

        docs = []
        for unique_intent, query_list in data_docs.items():
            if corpus_path:
                num_relevant = len(corpus[unique_intent])
            else:
                num_relevant = None
            for query in query_list:
                docs.append(Document(
                    text=convert_unicode(query),
                    label=unique_intent,
                    num_relevant=num_relevant,
                ))
        return docs
