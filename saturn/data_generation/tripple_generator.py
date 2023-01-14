import json
import os
import random
from typing import *
import shutil
from tqdm import tqdm
import questionary
from comet.lib import file_util, logger
from saturn.abstract_method.staturn_abstract import SaturnAbstract
from saturn.data_generation import constants
from saturn.data_generation.document_store.in_mem_store import InmemoryDocumentStore
from saturn.utils.config_parser import ConfigParser

_logger = logger.get_logger(__name__)


class TripleGenerator(SaturnAbstract):
    max_triple_per_file = 100000

    def __init__(self, config: Union[Text, ConfigParser]):
        super(TripleGenerator, self).__init__(config)
        self.document_store: InmemoryDocumentStore = InmemoryDocumentStore(
            self.config_parser
        )
        self.initialize()

    def initialize(self):
        general_cfg = self.config_parser.data_generation_config()
        for key, val in general_cfg.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def load(self):
        """
        Load document to store
        Returns:

        """
        if self.skipped:
            return
        # Check if data is already generated
        output_data_dir = self.get_data_dir()
        # Check if folder is not empty
        if self.is_warning_action and os.path.exists(output_data_dir) and len(os.listdir(output_data_dir)) > 0:
            is_regen = questionary.confirm(
                "Data is already generated. Do you want to regenerate?"
            ).ask()
            if not is_regen:
                self.skipped = True
                return
            else:
                # Clear old data in output folder
                is_cleaned = questionary.confirm(
                    "Do you want to clean old data in output folder?"
                ).ask()
                if is_cleaned:
                    shutil.rmtree(output_data_dir)

        data_dir = self.config_parser.data_generation_config()['data_dir']
        self.document_store.load_document(data_dir)
        self.ready = True
        _logger.info("Load data successfully")

    def generate_triples(self):
        if self.skipped_gen_data:
            return

        # Build documents
        self.document_store.build_documents()

        _logger.info("Starting generating triples")
        documents = self.document_store.get_documents(type=constants.KEY_POSITIVE)
        triples = []
        counter = 0
        check_point = 0
        for document in tqdm(documents):
            negatives_data = [{k: val} for k, val in document.negatives_ids.items()]
            for idx, positive_id in enumerate(document.positive_ids):
                anchor = document.text
                positive = self.document_store.documents[positive_id].text

                # Get ids randomly
                doc_ids = list(random.choice(negatives_data).values())[0]
                negative = self.document_store.documents[doc_ids[idx]].text
                triples.append((anchor, positive, negative))
                counter += 1
                if 0 < self.max_triple_per_file <= counter:
                    check_point += len(triples)
                    self.save(triples, f"triples_{check_point}.json", mode="triple")
                    counter = 0
                    triples = []

        check_point += len(triples)
        self.save(triples, f"triples_{check_point}.json", mode='triple')

    def generate_quadruplet(self):
        if not self.ready:
            raise ValueError("Please load data first")

        # Build documents
        self.document_store.build_documents()

        _logger.info("Starting generating quadruplet")
        documents = self.document_store.get_documents(type=constants.KEY_POSITIVE)
        quadruplet = []
        counter = 0
        check_point = 0
        for document in tqdm(documents):
            negatives_data = [{k: val} for k, val in document.negatives_ids.items()]
            for idx, positive_id in enumerate(document.positive_ids):
                anchor = document.text
                positive = self.document_store.documents[positive_id].text

                # Get ids randomly
                doc_ids = list(random.choice(negatives_data).values())[0]
                idx_random = random.randint(0, len(doc_ids) - 1)
                if idx_random == idx:
                    idx_random = len(doc_ids) - (idx + 1)
                negative1 = self.document_store.documents[doc_ids[idx]].text
                negative2 = self.document_store.documents[doc_ids[idx_random]].text
                quadruplet.append((anchor, positive, negative1, negative2))
                counter += 1
                if 0 < self.max_triple_per_file <= counter:
                    check_point += len(quadruplet)
                    self.save(quadruplet, f"quadruplet_{check_point}.json", mode="quadruplet")
                    counter = 0
                    quadruplet = []

        check_point += len(quadruplet)
        self.save(quadruplet, f"quadruplet_{check_point}.json", mode="quadruplet")

    def save(self, data: List[Tuple[str, str, str]], path, mode):
        rendered_data = self._rendered(data, mode=mode)
        output_dir = self.get_data_dir()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_path = os.path.join(output_dir, path)
        # file_util.dump_obj_as_json_to_file(file_path, rendered_data)
        file_util.write_text_file(json.dumps(rendered_data, indent=2, ensure_ascii=False), file_path)
        _logger.info(f"Saved data to {output_dir}")

    @staticmethod
    def _rendered(sentence_pairs: List[Tuple[str, str, str]], mode="triple"):
        rendered_data = {
            "data": [],
        }
        if mode in "triple":
            for pair in sentence_pairs:
                dict_triple = {
                    "query": pair[0],
                    "pos": pair[1],
                    "neg": pair[2],
                }
                rendered_data["data"].append(dict_triple)
        else:
            for pair in sentence_pairs:
                dict_quadruplet = {
                    "query": pair[0],
                    "pos": pair[1],
                    "neg1": pair[2],
                    "neg2": pair[3],
                }
                rendered_data["data"].append(dict_quadruplet)
        return rendered_data
