import copy
import itertools
import os
import time
import logging
import datetime
import random
import traceback
import json
from typing import List, Optional, Union, Dict, Any

import mmh3
import pandas as pd
from time import perf_counter
from iteration_utilities import unique_everseen

from loguru import logger

from meteor import Document, MultiLabel, Label, EvaluationResult
from meteor.document_stores import BaseDocumentStore, InMemoryDocumentStore
from meteor.modelling.evaluation.utils import load_config
from meteor.nodes import TfidfRetriever, EmbeddingRetriever
from meteor.nodes.retriever.sentence_embedding import SentenceEmbedding
from meteor.pipelines import DocumentSearchPipeline
from meteor.utils.io import load_json

doc_index = "eval_document"
label_index = "label"

index_results_file = "./reports/retriever_index_results.csv"
query_results_file = "./reports/retriever_query_results.csv"

seed = 42
random.seed(42)


class BenchMarker:
    def __init__(
            self,
            corpus_name_or_path: Optional[Union[str, List]] = None,
            dataset_name_or_path: Optional[Union[str, List]] = None,
            model_name_or_path: Optional[Union[str, List]] = None,

    ):
        self.corpus_name_or_path = corpus_name_or_path
        self.dataset_name_or_path = dataset_name_or_path
        self.model_name_or_path = model_name_or_path

    def indexing(
            self,
            retriever_type,
            doc_store_name,
            model_name_or_path: str = None,
            update_json: bool = False,
            save_markdown: bool = True
    ):
        retriever_results = []

        model_name_or_path = self.model_name_or_path if model_name_or_path is None else model_name_or_path
        doc_store = self.get_document_store(document_store_type=doc_store_name)
        docs, _ = self.prepare_data()
        retriever = self.get_retriever(retriever_type=retriever_type, document_store=doc_store,
                                       model_name_or_path=model_name_or_path)

        n_docs = len(docs)
        tic = perf_counter()
        self.index_to_doc_store(doc_store, docs, retriever)
        toc = perf_counter()
        indexing_time = toc - tic

        retriever_results.append(
            {
                "retriever": retriever_type,
                "retriever_model": model_name_or_path,
                "doc_store": doc_store_name,
                "n_docs": n_docs,
                "indexing_time": indexing_time,
                "docs_per_second": n_docs / indexing_time,
                "date_time": datetime.datetime.now(),
                "error": None,
            }
        )
        retriever_df = pd.DataFrame.from_records(retriever_results)
        retriever_df = retriever_df.sort_values(by="retriever").sort_values(by="doc_store")
        retriever_df.to_csv(index_results_file)
        logger.info("Deleting all docs from this run ...")

        doc_store.delete_documents(index=doc_index)

        if save_markdown:
            md_file = index_results_file.replace(".csv", ".md")
            with open(md_file, "w") as f:
                f.write(str(retriever_df.to_markdown()))
        time.sleep(10)

        if update_json:
            self.populate_retriever_json()

    def querying(
            self,
            retriever_type: Union[str, List] = None,
            doc_store_name: Union[str, List] = None,
            model_name_or_path: Union[str, List] = None,
            update_json: bool = False,
            save_markdown: bool = True
    ):
        """Benchmark the time it takes to perform querying. Doc embeddings are loaded from file."""
        retriever_results = []

        if model_name_or_path is None:
            model_name_or_path = self.model_name_or_path
        else:
            raise ValueError("Model is required to embed queries and documents")

        for kwarg in self.populate_kwargs(retriever_type, doc_store_name, model_name_or_path):
            results = self._querying(**kwarg)
            retriever_results.append(results)

        retriever_results = list(unique_everseen(retriever_results))
        retriever_df = pd.DataFrame.from_records(retriever_results)
        retriever_df = retriever_df.sort_values(by="retriever").sort_values(by="doc_store")
        retriever_df.to_csv(query_results_file)
        if save_markdown:
            md_file = query_results_file.replace(".csv", ".md")
            with open(md_file, "w") as f:
                f.write(str(retriever_df.to_markdown()))
        if update_json:
            self.populate_retriever_json()

    def populate_kwargs(self, retriever_types, doc_stores, models) -> List:
        kwargs = []
        retriever_types = [retriever_types] if isinstance(retriever_types, str) else retriever_types
        doc_stores = [doc_stores] if isinstance(doc_stores, str) else doc_stores
        models = [models] if isinstance(models, str) else models

        anchor_len = max(len(retriever_types), len(doc_stores), len(models))
        retriever_types = self._populate_arg(retriever_types, anchor_len)
        doc_stores = self._populate_arg(doc_stores, anchor_len)
        models = self._populate_arg(models, anchor_len)

        for kwarg in itertools.product(retriever_types, doc_stores, models):
            if kwarg[0] == "tfidf":
                model_name = "ms-viquad-bi-encoder-phobert-base"
            else:
                model_name = kwarg[2]
            kwargs.append(
                {
                    "retriever_type": kwarg[0],
                    "doc_store_name": kwarg[1],
                    "model_name_or_path": model_name
                }
            )
        return list(unique_everseen(kwargs))

    def _populate_arg(self, args: list, max_len: int) -> List:
        if len(args) < max_len:
            args.extend([args[0]] * (max_len - len(args)))
        return args

    def _querying(
            self,
            retriever_type,
            doc_store_name,
            model_name_or_path: str = None
    ):
        """Benchmark the time it takes to perform querying. Doc embeddings are loaded from file."""
        retriever_results = []

        docs, labels = self.prepare_data()

        n_docs = len(docs)
        n_queries = len(labels)

        try:
            logger.info("##### Start querying run: %s, %s, %s docs ##### ", retriever_type, doc_store_name,
                        n_docs)
            if retriever_type in ["elastic", "embedding"]:
                similarity = "cosine"
            else:
                similarity = "dot_product"

            embedding_dim = SentenceEmbedding.get_model_embedding_dim(model_name_or_path=model_name_or_path)
            doc_store = self.get_document_store(document_store_type=doc_store_name, embedding_dim=embedding_dim)
            retriever = self.get_retriever(retriever_type=retriever_type, document_store=doc_store,
                                           model_name_or_path=model_name_or_path)

            # For DPR, precomputed embeddings are loaded from file

            logger.info("Start indexing...")

            itic = perf_counter()
            self.index_to_doc_store(doc_store=doc_store, docs=docs, retriever=retriever)
            itoc = perf_counter()
            indexing_time = itoc - itic

            logger.info("Start queries...")

            rtic = perf_counter()
            pipeline = DocumentSearchPipeline(retriever=retriever)
            eval_result: EvaluationResult = pipeline.eval(labels=labels, params={"Retriever": {"top_k": 5}},
                                                          dynamic_top_k=True)
            rtoc = perf_counter()
            retrieve_time = rtoc - rtic

            raw_results = eval_result.calculate_metrics(document_scope="document_id")["Retriever"]
            if retriever_type == "tfidf":
                model_name_or_path = None

            results = {
                "retriever": retriever_type,
                "doc_store": doc_store_name,
                "retriever_model": model_name_or_path,
                "n_docs": n_docs,
                "indexing_time": indexing_time,
                "docs_per_second": n_docs / indexing_time,
                "n_queries": n_queries,
                "retrieve_time": retrieve_time,
                "queries_per_second": n_queries / retrieve_time,
                "seconds_per_query": retrieve_time / n_queries,
                "precision": round(raw_results["precision"] * 100, 2),
                "map": round(raw_results["map"] * 100, 2),
                "mrr": round(raw_results["mrr"], 2),
                "date_time": datetime.datetime.now(),
                "error": None,
            }

            logger.info("Deleting all docs from this run ...")
            doc_store.delete_documents(index=doc_index)
            time.sleep(5)
            del doc_store
            del retriever
        except Exception:
            tb = traceback.format_exc()
            logging.error(
                f"##### The following Error was raised while running querying run: {retriever_type}, {doc_store_name}, {n_docs} docs #####"
            )
            logging.error(tb)
            results = {
                "retriever": retriever_type,
                "doc_store": doc_store_name,
                "n_docs": n_docs,
                "n_queries": 0,
                "retrieve_time": 0.0,
                "queries_per_second": 0.0,
                "seconds_per_query": 0.0,
                "recall": 0.0,
                "map": 0.0,
                "top_k": 0,
                "date_time": datetime.datetime.now(),
                "error": str(tb),
            }
            logger.info("Deleting all docs from this run ...")
            # doc_store.delete_documents(index=doc_index)
            time.sleep(5)
            # del doc_store
        logger.info(results)
        return results

    def index_to_doc_store(self, doc_store, docs, retriever, labels=None):
        doc_store.write_documents(docs, doc_index)
        if labels:
            doc_store.write_labels(labels, index=label_index)
        # these lines are not run if the docs.embedding field is already populated with precomputed embeddings
        # See the prepare_data() fn in the retriever benchmark script
        if callable(getattr(retriever, "embed_documents", None)) and docs[0].embedding is None:
            doc_store.write_documents(docs)
            doc_store.update_embeddings(retriever=retriever, index=doc_index, batch_size=200)

    def prepare_data(self):
        corpus = load_json(self.corpus_name_or_path)
        dataset = load_json(self.dataset_name_or_path)
        labels = []

        docs = [Document.from_dict(doc) if isinstance(doc, dict) else doc for doc in self._prepare_data(corpus)]
        for data in dataset.keys():
            _dataset = dataset[data]
            _corpus = corpus[data]
            labels.extend([self.convert_labels(d, _corpus) for d in _dataset])
        return docs, labels

    def _prepare_data(self, data) -> List[Union[Document, Dict[str, Any]]]:
        docs: list = []
        contents = copy.deepcopy(data)
        for key in contents.keys():
            contents[key] = [{"content": content} for content in contents[key]]
            docs.extend(contents[key])
        return docs

    def convert_labels(self, query, contents) -> MultiLabel:
        labels = [Label(
            query=query,
            document=Document(
                id=self._get_id(content),
                content_type="text",
                content=content
            )
        ) for content in contents]
        return MultiLabel(labels=labels)

    def _get_id(self, content):
        return "{:02x}".format(mmh3.hash128(str(content), signed=False))

    def get_retriever(self, retriever_type, document_store, model_name_or_path):
        if retriever_type == "tfidf":
            retriever = TfidfRetriever(document_store=document_store)
        elif retriever_type == "embedding":
            retriever = EmbeddingRetriever(
                document_store=document_store, embedding_model=model_name_or_path, use_gpu=False
            )
        else:
            raise Exception(f"No retriever fixture for '{retriever_type}'")

        return retriever

    @classmethod
    def load_config(cls):
        pass

    def download_from_axiom(self):
        pass

    def document_store_with_docs(self, retriever_type, document_store_type, docs, model_name_or_path):
        document_store = self.get_document_store(document_store_type=document_store_type)
        retriever = self.get_retriever(retriever_type=retriever_type, document_store=document_store,
                                       model_name_or_path=model_name_or_path)
        document_store.delete_documents(document_store.index)
        document_store.write_documents(docs)
        document_store.update_embedding(
            retriever=retriever, document_store=document_store)
        yield document_store
        document_store.delete_index(document_store.index)

    def get_document_store(
            self,
            document_store_type,
            embedding_dim=768,
            embedding_field="embedding",
            index=doc_index,
            similarity: str = "cosine",
    ):
        document_store: BaseDocumentStore
        if document_store_type == "memory":
            document_store = InMemoryDocumentStore(
                return_embedding=True,
                embedding_dim=embedding_dim,
                embedding_field=embedding_field,
                index=index,
                similarity=similarity,
            )
        else:
            raise Exception(f"No document store fixture for '{document_store_type}'")
        return document_store


if __name__ == "__main__":
    params, conf_bucket, retriever_models = load_config(
        config_filename="/Users/phongnt/FTECH/knowledge-retrieval/meteor/nodes/evaluator/config.json", ci=True)
    ben = BenchMarker("assets/corpus.json", "assets/dataset.json",
                      ["ms-viquad-bi-encoder-phobert-base", "paraphrase-multilingual-MiniLM-L12-v2"])
    ben.querying(["embedding", "tfidf"], "memory")
