from meteor.utils.config_parser import ConfigParser
from comet.lib.helpers import get_module_or_attr
from comet.lib import file_util
from typing import *
from meteor.components.utils.document import Document
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

        if "triplets_data_path" not in trainer_config:
            raise FileNotFoundError("No triplets data path provided in config file")

        triplets_data = []
        triplets_data_path = trainer_config.pop("triplets_data_path")

        if isinstance(triplets_data_path, str):
            triplets_data_path = [triplets_data_path]
        for path in triplets_data_path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"triplets data path {path} does not exist")
            triplets_data.extend(file_util.load_json(path)['data'])
        trainer_config = dict(trainer_config, triplets_data=triplets_data)
        self.embedder.train(**trainer_config)

    def evaluate_embedder(self):
        model_name_or_paths = self.config_parser.eval_config()['model_name_or_path']
        if isinstance(model_name_or_paths, str):
            model_name_or_paths = [model_name_or_paths]
        for model_name_or_path in model_name_or_paths:
            name = os.path.basename(model_name_or_path)
            self.embedder.load_model(cache_path=name, pretrained_name_or_abspath=model_name_or_path)
            # TODO evaluate
            tgt_docs = [doc.text for doc in self.corpus_docs]
            src_docs = [doc.text for doc in self.query_docs]
            scores = self.embedder.find_similarity(src_docs, tgt_docs, _return_as_dict=True)
            # ====== Continue here======
            #
            # ==========================

            # TODO Save to eval store and export the results
            # ======= Continue here =======
            #
            # =============================

        raise NotImplementedError

    def _load_docs(self, path: str) -> List[Document]:
        from comet.utilities.utility import convert_unicode
        # TODO load data from path Remember to convert_unicode
        raise NotImplementedError
