import datetime
import os
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Optional, List, Union, Tuple, Text, Dict

import questionary
import pandas as pd
from pandas import DataFrame
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
from xlsxwriter.utility import xl_range_abs
from tabulate import tabulate
import shutil
import torch

from comet.lib import file_util, logger
from comet.lib.helpers import get_module_or_attr
from comet.utilities.utility import convert_unicode
from comet.lib.print_utils import print_title
from comet.components.embeddings.embedding_models import BertEmbedder

from saturn.components.utils.document import Document, EvalResult
from saturn.utils.config_parser import ConfigParser
from saturn.abstract_method.staturn_abstract import SaturnAbstract
from saturn.utils.io import write_csv, write_md, write_json
from saturn.data_generation.document_store.utils import fast_argsort_1d_bottleneck
from saturn.utils.reflection import Style

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
        self._top_k: Optional[int] = None
        self._retriever_threshold: Optional[float] = None
        self._default_faq_label: Optional[str] = "faq/out_of_scope"

        try:
            self.eval_config = self.config_parser.eval_config()
        except:
            _logger.error(f"Failed to load general config")

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
        """
        Path to save report with convention: reports/project/version/sub_folder
        """
        if not self._output_dir:
            general_config = self.config_parser.general_config()
            if "output_report" not in general_config:
                self._output_dir = "reports"
                return self._output_dir
            self._output_dir = os.path.join(general_config.get("output_report"), general_config.get("project"),
                                            general_config.get("version"))
            return self._output_dir
        return self._output_dir

    @property
    def top_k(self):
        if not self._top_k:
            if 'top_k' not in self.eval_config:
                _logger.warning('Not found to get top_k, so the default value will be applied')
                self._top_k = 5
                return self._top_k
            self._top_k = self.eval_config.get('top_k')
            return self._top_k
        return self._top_k

    @property
    def retriever_threshold(self):
        if not self._retriever_threshold:
            if 'retriever_threshold' not in self.eval_config:
                _logger.warning('Not found to get retriever_threshold, so the default value will be applied')
                self._retriever_threshold = 0.5
                return self._retriever_threshold
            self._retriever_threshold = self.eval_config.get('retriever_threshold')
            return self._retriever_threshold
        return self._retriever_threshold

    @property
    def default_faq_label(self):
        if not self._default_faq_label:
            if 'default_faq_label' not in self.eval_config:
                _logger.warning('Not found to get default_faq_label, so the default value will be applied')
                self._default_faq_label = 'faq/out_of_scope'
                return self._default_faq_label
            self._default_faq_label = self.eval_config.get('default_faq_label')
            return self._default_faq_label
        return self._default_faq_label

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

    def evaluate(self,
                 top_k: int = None,
                 retriever_threshold: float = None,
                 default_faq_label: str = None,
                 save_report: bool = True):

        top_k = top_k if top_k else self.top_k
        retriever_threshold = retriever_threshold if retriever_threshold else self.retriever_threshold
        default_faq_label = default_faq_label if default_faq_label else self.default_faq_label

        pretrained_name_or_abspaths = self.pretrained_name_or_abspath

        if isinstance(pretrained_name_or_abspaths, str):
            pretrained_name_or_abspaths = [pretrained_name_or_abspaths]
        evaluation_results: Dict[Dict[str, List[EvalResult]]] = {}

        for pretrained_name_or_abspath in tqdm(list(pretrained_name_or_abspaths)):
            name = os.path.basename(pretrained_name_or_abspath)
            self.print_line(name, Style.MAGENTA)
            evaluation_results[name] = {}
            try:
                batch_size = 256 if torch.cuda.is_available() else 8
                embedder = BertEmbedder(
                    pretrained_name_or_abspath=pretrained_name_or_abspath, device=self.device, batch_size=batch_size
                )
            except Exception as e:
                _logger.error(f"Failed to load model {pretrained_name_or_abspath} due to {e}")
                continue
            tic = perf_counter()
            tgt_docs = [convert_unicode(doc.text) for doc in self.corpus_docs]
            src_docs = [convert_unicode(doc.text) for doc in self.query_docs]

            similarity_data = embedder.find_similarity(src_docs, tgt_docs, _no_sort=True)

            toc = perf_counter()
            retriever_time = toc - tic

            eval_results = self._extract_eval_result(self.query_docs,
                                                     tgt_docs,
                                                     similarity_data,
                                                     top_k=top_k,
                                                     retriever_threshold=retriever_threshold,
                                                     default_faq_label=default_faq_label)

            evaluation_results[name]["retriever_docs"] = eval_results
            evaluation_results[name]["retriever_time"] = round(retriever_time, 2)
            evaluation_results[name]["query_numbers"] = len(src_docs)

            # log mrr and map metrics for tracing the evaluation
            df = pd.DataFrame(eval_results)
            df = df.apply(pd.Series.explode)
            df.insert(5, 'score', df.pop('relevant_doc_scores'))
            df.columns = ['query', 'gt_label', 'top_k', 'rr', 'ap', 'score', 'relevant_docs', 'predicted_label']
            _logger.info(f"Retriever results for {Style.BLUE}{Style.BOLD}'{name}': map: {round(df['ap'].mean(), 2)}, "
                         f"mrr: {round(df['rr'].mean(), 2)}")

            if save_report:
                self.save_detail_report(model_name=name, df=evaluation_results[name])
                write_json(
                    pd.DataFrame(eval_results).to_dict(), os.path.join(self.output_dir, 'details', f'{name}.json'))

        if save_report:
            # compute information retrieval metrics
            metrics = self.compute_ir_metrics(eval_results=evaluation_results)  # type: ignore

            # export information retrieval summaries
            write_csv(self.output_dir, 'model_ir_summary.csv', metrics)
            write_md(self.output_dir, 'model_ir_summary.md', metrics)

        return evaluation_results

    def compute_ir_metrics(self, eval_results: Dict[str, Dict[str, List[EvalResult]]] = None) -> List:
        """
        Compute information retrieval metrics such as mean average precision (mAP), mean reciprocal rank (mRR)
        To see detail, please visit here: https://amitness.com/2020/08/information-retrieval-evaluation/

        :param eval_results: evaluation results to compute metrics
        """
        metrics = []
        if not eval_results:
            raise ValueError(f"Failed to write results with empty data")

        for model_name, eval_result in eval_results.items():
            df = pd.DataFrame(eval_result["retriever_docs"])
            df = df.apply(pd.Series.explode)
            df.insert(5, 'score', df.pop('relevant_doc_scores'))
            df.columns = ['query', 'gt_label', 'top_k', 'rr', 'ap', 'score', 'relevant_docs', 'predicted_label']
            # _logger.info(f"Retriever results for '{model_name}': map: {round(df['ap'].mean(), 2)}, "
            #              f"mrr: {round(df['rr'].mean(), 2)}")
            records = {
                "model_name": model_name,
                "query_numbers": eval_result["query_numbers"],
                "retriever_time": eval_result["retriever_time"],
                "query_per_second": round(
                    float(eval_result["retriever_time"] / eval_result["query_numbers"]), 2),
                "map": round(df["ap"].mean(), 2),
                "mrr": round(df["rr"].mean(), 2),
                "date_time": datetime.datetime.now()
            }
            metrics.append(records)

        measurer = np.vectorize(len)
        scale = sum(measurer(pd.DataFrame(metrics).values.astype(str)).max(axis=0)) + 25
        print_title(text="Knowledge Retrieval Overall Results", scale=scale, color='purple')
        print(tabulate(pd.DataFrame(metrics), headers='keys', tablefmt='pretty'))
        return metrics

    def save_overall_report(
            self,
            output_dir: Union[str, Path],
            df: Union[DataFrame, List],
            report_type: Optional[str] = 'csv',
            report_name: Optional[str] = None,
    ):
        # output_dir = Path(self.get_report_dir())
        _logger.warn('This function will be deprecated in version 0.1.2 and above. Please use io.write_csv() instead',
                     DeprecationWarning, stacklevel=2)
        _logger.info("Saving evaluation results to %s", output_dir)
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True)

        retriever_df = pd.DataFrame.from_records(df)
        if report_type == 'csv':
            report_name = report_name if report_name else 'knowledge_retrieval'
            target_path = os.path.join(output_dir, f"{report_name}.csv")
            retriever_df.to_csv(target_path)

        elif report_type == 'markdown':
            md_file = os.path.join(output_dir, f"{report_name}.md")
            with open(md_file, "w") as f:
                f.write(str(retriever_df.to_markdown()))
        else:
            raise NotImplemented(f"This format {report_type} has not supported yet.")

    def save_detail_report(self,
                           model_name: str,
                           df: Union[DataFrame, List],
                           output_dir: Optional[Union[str, Path]] = None
                           ):
        """
        Saves the evaluation result.
        The result of each node is saved in a separate csv with file name {node_name}.csv to the output_dir folder.

        :param model_name: Model name to benchmark
        :param df: Evaluation results
        :param output_dir: Path to the target folder the csvs will be saved.
        """
        output_dir = output_dir if output_dir else self.output_dir
        output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
        output_dir = Path(os.path.join(output_dir, 'details'))
        _logger.info("Saving evaluation results to %s", output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        target_path = output_dir / f"{model_name}.xlsx"

        if isinstance(df, list):
            df = pd.DataFrame(df)

        df = pd.DataFrame(df["retriever_docs"])
        df.insert(loc=0, column='index', value=df.index)
        df.insert(6, 'score', df.pop('relevant_doc_scores'))
        df.columns = ['index', 'query', 'gt_label', 'top_k', 'rr', 'ap', 'score', 'relevant_docs', 'predicted_label']
        # explode lists of corpus to row
        df = df.apply(pd.Series.explode)

        df_merged = pd.DataFrame(df.to_dict('records'))
        df_merged.score = df_merged.score.round(decimals=2)

        df_merged_wrong_queries = pd.DataFrame(df.to_dict('records'))
        df_merged_wrong_queries = df_merged_wrong_queries[
            df_merged_wrong_queries['gt_label'] != df_merged_wrong_queries['predicted_label']]
        df_merged_wrong_queries = df_merged_wrong_queries.reset_index()
        df_merged_wrong_queries.pop('level_0')
        df_merged_wrong_queries.score = df_merged_wrong_queries.score.round(decimals=2)

        writer = pd.ExcelWriter(f'{target_path}', engine='xlsxwriter')
        df_merged.to_excel(writer, sheet_name='Detail_Report', index=False)
        df_merged_wrong_queries.to_excel(writer, sheet_name='Wrong_Query', index=False)

        workbook = writer.book

        header_format = workbook.add_format(
            {
                'align': 'center',
                'valign': 'vcenter',
                'border': 1,
                'bg_color': '#fff2cc'
            }
        )

        merge_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})
        bg_format_odd = workbook.add_format({'bg_color': '#cfe2f3', 'border': 1})
        bg_format_even = workbook.add_format({'bg_color': '#FFFFFF', 'border': 1})
        bg_format_wrong = workbook.add_format({'font_color': 'red'})
        bg_best_score = workbook.add_format({'bold': True})

        def formatter(dt, sheet_name: str):
            worksheet = writer.sheets[sheet_name]
            header_range = xl_range_abs(0, 0, 0, dt.shape[1] - 1)
            worksheet.conditional_format(header_range, {'type': 'no_blanks',
                                                        'format': header_format})
            for idx in dt['index'].unique():
                # find indices and add one to account for header
                u = dt.loc[dt['index'] == idx].index.values + 1

                cell_range = xl_range_abs(u[0], 0, u[0] + len(u) - 1, dt.shape[1])

                if idx % 2 == 0:
                    worksheet.conditional_format(cell_range, {'type': 'no_blanks',
                                                              'format': bg_format_odd})
                else:
                    worksheet.conditional_format(cell_range, {'type': 'no_blanks',
                                                              'format': bg_format_even})
                for i in u:
                    if dt['gt_label'][i - 1] != dt['predicted_label'][i - 1]:
                        # wrong_label_range = xl_range_abs(i, dt.columns.get_loc('most_relevant_docs'),
                        #                                  i, dt.shape[1])
                        worksheet.write(
                            i, dt.columns.get_loc('relevant_docs'),
                            dt['relevant_docs'][i - 1], bg_format_wrong)
                        worksheet.write(
                            i, dt.columns.get_loc('predicted_label'),
                            dt['predicted_label'][i - 1], bg_format_wrong)
                        worksheet.write(
                            i, dt.columns.get_loc('score'),
                            dt['score'][i - 1], bg_format_wrong)
                    else:
                        if float(dt['score'][i - 1]) >= 0.9:
                            worksheet.write(
                                i, dt.columns.get_loc('score'),
                                dt['score'][i - 1], bg_best_score)
                        else:
                            pass

                if len(u) < 2:
                    pass  # do not merge cells if there is only one row
                else:
                    column_index = {
                        'index': 0,
                        'query': 1,
                        'gt_label': 2,
                        'top_k': 3,
                        'rr': 4,
                        'ap': 5
                    }
                    # merge cells using the first and last indices
                    for key, index in column_index.items():
                        worksheet.merge_range(u[0], index, u[-1], index, dt.loc[u[0], f'{key}'], merge_format)

        # auto-adjust column size
        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column))
            col_idx = df.columns.get_loc(column)
            writer.sheets['Detail_Report'].set_column(col_idx, col_idx, column_width)
            writer.sheets['Wrong_Query'].set_column(col_idx, col_idx, column_width)

        formatter(df_merged, 'Detail_Report')
        formatter(df_merged_wrong_queries, 'Wrong_Query')
        writer.save()
        _logger.info(f"Evaluation report with excel format is saved at {target_path}")

    def _extract_eval_result(
            self,
            src_docs: List[Document],
            tgt_docs,
            similarity_data_2d: List[List[Tuple[Text, float]]],
            top_k: int = None,
            retriever_threshold: float = None,
            default_faq_label: str = None
    ) -> List:
        eval_results = []

        for src_doc, similarities in tzip(src_docs, similarity_data_2d):
            rr_score = 0
            ap_score = 0
            ap = 0

            # Get top_k relevant docs from indices
            top_k_relevant_docs = []
            relevant_doc_scores = []
            predicted_labels = []
            top_k = min(top_k, 20)

            scores = np.array([score for _, score in similarities])
            indices = fast_argsort_1d_bottleneck(scores, axis=0, top_k=src_doc.num_relevant).tolist()

            # Get score of each relevant doc
            for i in indices[:src_doc.num_relevant]:
                top_k_relevant_docs.append(tgt_docs[i])
                sim_score = similarities[i][1]
                relevant_doc_scores.append(sim_score)
                predicted_labels.append(self.corpus_docs[i].label)

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
                most_relevant_docs=top_k_relevant_docs[:top_k],
                relevant_doc_scores=relevant_doc_scores[:top_k],
                predicted_labels=predicted_labels[:top_k]
            )
            # eval_results.append(eval_result.to_dict())
            df = pd.DataFrame(eval_result.to_dict())
            df = df.apply(pd.Series.explode)
            df.loc[df['relevant_doc_scores'] <= retriever_threshold, 'predicted_labels'] = default_faq_label
            eval_results.append(df.to_dict())
        return eval_results

    @staticmethod
    def _get_input_reference_corpus(list_or_path: str) -> List[str]:
        """
        Load the corpus from the given path.
        :param list_or_path: Path to the corpus
        :return: list of corpus
        raise FileNotFoundError: If the given path is not found
        """
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

    @staticmethod
    def print_line(text, style):
        print("😹 😹 😹 😹 😹 😹 😹 😹 😹 😹 {} 😹 😹 😹 😹 😹 😹 😹 😹 😹 😹".format(f'{style}{text.upper()}'))
