import datetime
import os
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Optional, List, Union, Tuple, Text, Dict
from comet.components.embeddings.embedding_models import BertEmbedder
import questionary
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from xlsxwriter.utility import xl_range_abs
from tabulate import tabulate
import shutil
from comet.lib import file_util, logger
from comet.lib.helpers import get_module_or_attr
from comet.utilities.utility import convert_unicode
from comet.lib.print_utils import print_title

from saturn.components.utils.document import Document, EvalResult
from saturn.utils.config_parser import ConfigParser
from saturn.abstract_method.staturn_abstract import SaturnAbstract

if TYPE_CHECKING:
    from saturn.components.embeddings.embedding_models import SBertSemanticSimilarity

_logger = logger.get_logger(__name__)


class KRManager(SaturnAbstract):
    def __init__(self, config: Union[str, ConfigParser]):
        super(KRManager, self).__init__(config)
        self._embedder: Optional[SBertSemanticSimilarity] = None
        self._corpus_docs: Optional[List[Document]] = None
        self._query_docs: Optional[List[Document]] = None
        self._pretrained_name_or_abspath: Optional[Union[str, List[str]]] = None
        self.retriever_type: Optional[Union[str, List]] = "embedding"
        self.document_store_type: Optional[Union[str, List]] = "memory"
        self._output_dir: Optional[Union[str, Path]] = None

    @property
    def embedder(self):
        if not self._embedder:
            embedder_config = self.config_parser.trainer_config()
            class_name = embedder_config.pop("class")
            module_name = embedder_config.pop("package")
            self._embedder = get_module_or_attr(module_name, class_name)(config=self.config_parser, **embedder_config)
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
    def pretrained_name_or_abspath(self):
        if not self._pretrained_name_or_abspath:
            eval_config = self.config_parser.eval_config()
            if "pretrained_name_or_abspath" not in eval_config:
                raise FileNotFoundError("No model name or path provided in config file")
            self._pretrained_name_or_abspath = eval_config.get("pretrained_name_or_abspath")

        if isinstance(self._pretrained_name_or_abspath, str):
            self._pretrained_name_or_abspath = [self._pretrained_name_or_abspath]
            return self._pretrained_name_or_abspath
        elif isinstance(self._pretrained_name_or_abspath, list):
            return self._pretrained_name_or_abspath
        else:
            raise ValueError("pretrained_name_or_abspath should be a string or a list of string")

    @property
    def output_dir(self):
        if not self._output_dir:
            eval_config = self.config_parser.eval_config()
            if "output_dir" not in eval_config:
                self._output_dir = "reports"
                return self._output_dir
            self._output_dir = eval_config.get("output_dir")
            return self._output_dir
        else:
            _logger.warning("No specific output direction so that the report will be saved at `./reports`")
            self._output_dir = "reports"
            return self._output_dir

    def train_embedder(self):
        if self.skipped_training:
            return

        output_model_dir = self.get_model_dir()
        if self.is_warning_action and os.path.exists(output_model_dir) and len(os.listdir(output_model_dir)) > 0:
            is_retrained = questionary.confirm("The model has been trained, do you want to retrain it?").ask()
            if not is_retrained:
                self.skipped = True
                return
            else:
                is_cleaned = questionary.confirm("Do you want to clean the model directory?").ask()
                if is_cleaned:
                    shutil.rmtree(output_model_dir)

        trainer_config = self.config_parser.trainer_config()
        self.embedder.train(trainer_config)

        # TODO save the model and upload it to axiom
        # Your code here

    def inference(self, input_query: str, input_corpus_list_or_path: Union[str, List[str]], top_k: int):
        """
        inference the input query with the input corpus
        :param input_query: the input query
        :param input_corpus_list_or_path: the input corpus list or the corpus path
        :param top_k: the top k results
        :return: the top k results
        """
        retriever_results = {'relevant doc': [], 'score': []}
        # pretrained_name_or_abspaths = self.pretrained_name_or_abspath

        input_query = [convert_unicode(input_query)]
        input_corpus = self._get_input_reference_corpus(input_corpus_list_or_path)
        # self.embedder.load_model(cache_path=name, pretrained_name_or_abspath=pretrained_name_or_abspath)
        similarity_data = self.embedder.find_similarity(input_query, input_corpus, _no_sort=False, top_n=top_k)
        similarity_data = similarity_data[0][:top_k]
        # reformat the results
        for text, score in similarity_data:
            retriever_results['relevant doc'].append(text)
            retriever_results['score'].append(score)
        return retriever_results

    def evaluate_embedder(self, top_k: int = None):
        retriever_results = []
        retriever_top_k_results = []
        pretrained_name_or_abspaths = self.pretrained_name_or_abspath
        if isinstance(pretrained_name_or_abspaths, str):
            pretrained_name_or_abspaths = [pretrained_name_or_abspaths]
        evaluation_results: Dict[str, List[EvalResult]] = {}
        # evaluation_top_k_results: Dict[str, List[EvalResult]] = {}
        for pretrained_name_or_abspath in tqdm(list(set(pretrained_name_or_abspaths))):
            name = os.path.basename(pretrained_name_or_abspath)
            evaluation_results[name] = []
            evaluation_top_k_results: Dict[str, List[EvalResult]] = {}
            try:
                embedder = BertEmbedder(
                    pretrained_name_or_abspath=pretrained_name_or_abspath, device=self.device
                )
                # self.embedder.load_model(cache_path=None, pretrained_name_or_abspath=pretrained_name_or_abspath)
            except Exception as e:
                _logger.error(f"Failed to load model {pretrained_name_or_abspath} due to {e}")
                continue
            tic = perf_counter()
            tgt_docs = [convert_unicode(doc.text) for doc in self.corpus_docs]
            src_docs = [convert_unicode(doc.text) for doc in self.query_docs]
            similarity_data = embedder.find_similarity(src_docs, tgt_docs, _no_sort=True)
            toc = perf_counter()
            retriever_time = toc - tic

            eval_results, eval_top_k_results = self._extract_eval_result(self.query_docs,
                                                                         tgt_docs,
                                                                         similarity_data,
                                                                         top_k=top_k)
            evaluation_results[name].extend(eval_results)
            evaluation_top_k_results[name] = eval_top_k_results
            retriever_top_k_results.append(evaluation_top_k_results)
            df = pd.DataFrame(evaluation_results[name])

            records = {
                "model_name": name,
                "query_numbers": len(src_docs),
                "retriever_time": round(retriever_time, 2),
                "query_per_second": round(retriever_time / len(src_docs), 2),
                "map": round(df["ap_score"].mean(), 2),
                "mrr": round(df["rr_score"].mean(), 2),
                "date_time": datetime.datetime.now()
            }
            retriever_results.append(records)

        return retriever_results, retriever_top_k_results

    def save(self, report_type: str = "all", top_k: int = None, save_markdown: bool = True):
        output_report_dir = self.get_report_dir()
        if self.skipped_eval:
            return

        if self.is_warning_action and os.path.exists(output_report_dir) and len(os.listdir(output_report_dir)) > 0:
            is_re_evaluated = questionary.confirm("The report has been generated, do you want to re-evaluate it?").ask()
            if not is_re_evaluated:
                return
            else:
                is_cleaned = questionary.confirm("Do you want to clean the report directory?").ask()
                if is_cleaned:
                    shutil.rmtree(output_report_dir)

        retriever_results, retriever_top_k_results = self.evaluate_embedder(top_k=top_k)
        if report_type == "overall":
            self.save_overall_report(
                output_dir=self.output_dir,
                df=pd.DataFrame(retriever_results),
                save_markdown=save_markdown
            )
        elif report_type == "detail":
            for models in retriever_top_k_results:
                for model, data in models.items():
                    self.save_detail_report(output_dir=self.output_dir, model_name=model, df=data)
        elif report_type == "all":
            self.save_overall_report(
                output_dir=self.output_dir,
                df=pd.DataFrame(retriever_results),
                save_markdown=save_markdown
            )
            for models in retriever_top_k_results:
                for model, data in models.items():
                    self.save_detail_report(output_dir=self.output_dir, model_name=model, df=data)
        else:
            raise NotImplemented(f"Sorry, this report type {report_type} is not found")

    def save_overall_report(
        self,
        output_dir: Union[str, Path],
        df: Union[DataFrame, List],
        save_markdown: bool = False
    ):
        output_dir = Path(self.get_report_dir())
        _logger.info("Saving evaluation results to %s", output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        target_path = os.path.join(output_dir, "knowledge_retrieval.csv")

        retriever_df = pd.DataFrame.from_records(df)
        retriever_df = retriever_df.sort_values(by="map")
        retriever_df.to_csv(target_path)

        if save_markdown:
            md_file = target_path.replace(".csv", ".md")
            with open(md_file, "w") as f:
                f.write(str(retriever_df.to_markdown()))

        print_title(text="Knowledge Retrieval Overall Results", scale=110, color='purple')
        print(tabulate(retriever_df, headers='keys', tablefmt='pretty'))

    def save_detail_report(self, output_dir: Union[str, Path], model_name: str, df: Union[DataFrame, List]):
        """
        Saves the evaluation result.
        The result of each node is saved in a separate csv with file name {node_name}.csv to the output_dir folder.

        :param output_dir: Path to the target folder the csvs will be saved.
        :param model_name: Model name to benchmark
        :param df: Evaluation result
        """
        output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
        _logger.info("Saving evaluation results to %s", output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        target_path = output_dir / f"{model_name}.xlsx"

        if isinstance(df, list):
            df = pd.DataFrame(df)
        # df = df.drop(columns=['query_id'])
        df.insert(loc=0, column='index', value=df.index)

        # explode lists of corpus to row
        df = df.apply(pd.Series.explode)
        df_merged = pd.DataFrame(df.to_dict('records'))

        writer = pd.ExcelWriter(f'{target_path}', engine='xlsxwriter')
        df_merged.to_excel(writer, sheet_name='Detail_Report', index=False)

        workbook = writer.book
        worksheet = writer.sheets['Detail_Report']
        header_format = workbook.add_format(
            {
                'align': 'center',
                'valign': 'vcenter',
                'border': 1,
                'bg_color': '#fff2cc'
            }
        )
        header_range = xl_range_abs(0, 0, 0, df_merged.shape[1] - 1)
        worksheet.conditional_format(header_range, {'type': 'no_blanks',
                                                    'format': header_format})

        merge_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})
        bg_format_odd = workbook.add_format({'bg_color': '#cfe2f3', 'border': 1})
        bg_format_even = workbook.add_format({'bg_color': '#FFFFFF', 'border': 1})
        bg_format_wrong = workbook.add_format({'font_color': 'red'})

        for idx in df_merged['index'].unique():
            # find indices and add one to account for header
            u = df_merged.loc[df_merged['index'] == idx].index.values + 1

            cell_range = xl_range_abs(u[0], 0, u[0] + len(u) - 1, df_merged.shape[1])

            if idx % 2 == 0:
                worksheet.conditional_format(cell_range, {'type': 'no_blanks',
                                                          'format': bg_format_odd})
            else:
                worksheet.conditional_format(cell_range, {'type': 'no_blanks',
                                                          'format': bg_format_even})
            for i in u:
                if df_merged['label'][i - 1] != df_merged['predicted_labels'][i - 1]:
                    wrong_label_range = xl_range_abs(i, df_merged.columns.get_loc('most_relevant_docs'),
                                                     i, df_merged.shape[1])
                    worksheet.conditional_format(wrong_label_range, {'type': 'no_blanks',
                                                                     'format': bg_format_wrong})

            if len(u) < 2:
                pass  # do not merge cells if there is only one row
            else:
                column_index = {
                    'index': 0,
                    'query': 1,
                    'label': 2,
                    'top_k_relevant': 3,
                    'rr_score': 4,
                    'ap_score': 5
                }
                # merge cells using the first and last indices
                for key, index in column_index.items():
                    worksheet.merge_range(u[0], index, u[-1], index, df_merged.loc[u[0], f'{key}'], merge_format)

        # auto-adjust column size
        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column))
            col_idx = df.columns.get_loc(column)
            writer.sheets['Detail_Report'].set_column(col_idx, col_idx, column_width)

        writer.save()
        _logger.info(f"Evaluation report with excel format is saved at {target_path}")

    def _extract_eval_result(
        self, src_docs: List[Document], tgt_docs, similarity_data_2d: List[List[Tuple[Text, float]]],
        top_k: int = None
    ):
        eval_results = []
        eval_top_k_results = []

        for src_doc, similarities in zip(src_docs, similarity_data_2d):
            indices = self._arg_sort(similarities)
            rr_score = 0
            ap_score = 0
            ap = 0

            # Get top_k relevant docs from indices
            top_k_relevant_docs = []
            relevant_doc_scores = []
            predicted_labels = []

            # Get score of each relevant doc

            for i in indices[:src_doc.num_relevant]:
                top_k_relevant_docs.append(tgt_docs[i])
                for id, answer in enumerate(similarities):
                    if tgt_docs[i] == answer[0]:
                        relevant_doc_scores.append(str(answer[1]))
                        predicted_labels.extend([doc.label for doc in self.corpus_docs if doc.text == answer[0]])

            ground_truths = [doc.text for doc in self.corpus_docs if doc.label == src_doc.label]
            for idx, relevant_doc in enumerate(top_k_relevant_docs):
                if relevant_doc not in ground_truths:
                    continue
                ap += 1
                if ap == 1 and rr_score == 0:
                    rr_score = 1 / int(idx + 1)
                else:
                    rr_score = rr_score
                ap_score += ap / (int(idx) + 1)

            eval_result = EvalResult(
                query=src_doc.text,
                label=src_doc.label,
                rr_score=rr_score,
                ap_score=round((ap_score / src_doc.num_relevant), 2),
                top_k_relevant=src_doc.num_relevant,
                most_relevant_docs=top_k_relevant_docs,
                relevant_doc_scores=relevant_doc_scores,
                predicted_labels=predicted_labels

            )
            top_k = top_k if top_k in range(1, src_doc.num_relevant) else src_doc.num_relevant
            if top_k > 20:
                _logger.warning(
                    f"Got {top_k} instead of being less than or equal to 20, so the default value {10} will be applied")
                top_k = 10

            try:
                tmp_df = pd.DataFrame(eval_result.to_dict()).head(top_k)
            except:
                tmp_df = pd.DataFrame()
            eval_top_k_results.append(tmp_df.to_dict())
            eval_results.append(eval_result.to_dict())
        return eval_results, eval_top_k_results

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
    def _get_input_reference_corpus(list_or_path: str) -> List[str]:
        """
        Load the corpus from the given path.
        :param list_or_path: Path to the corpus
        :return: list of corpus
        raise FileNotFoundError: If the given path is not found
        """
        # print(list_or_path)
        if isinstance(list_or_path, str):
            if not os.path.isfile(list_or_path):
                raise FileNotFoundError(f"File {list_or_path} does not exist")
            with open(list_or_path, "r") as f:
                input_corpus = f.readlines()
                for idx, item in enumerate(input_corpus):
                    input_corpus[idx] = convert_unicode(item.replace("\n", ""))
        elif isinstance(list_or_path, list):
            input_corpus = list_or_path
            for idx, item in enumerate(input_corpus):
                input_corpus[idx] = convert_unicode(item)
        else:
            raise ValueError(f"Invalid input type {type(list_or_path)}")
        return input_corpus

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
