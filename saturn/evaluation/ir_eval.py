import os
from pathlib import Path
from typing import *

import pandas as pd
from loguru import logger
from pandas import DataFrame
from xlsxwriter.utility import xl_range_abs
from saturn.utils.io import write_json, load_json
from saturn.evaluation.schemas import Document, EvalData
from saturn.evaluation.utils import BaseRetriever
from saturn.utils import print_utils
import tqdm


class IREvaluator:
    def __init__(self, retriever: BaseRetriever, eval_dataset: List[EvalData], documents: List[Document] = None):
        """
        Evaluates the performance of a retriever on a given dataset, optionally including additional documents.

        Args:
            retriever (BaseRetriever): An instance of a retriever that implements the `BaseRetriever` interface.
            eval_dataset (List[EvalData]): A list of `EvalData` objects representing the evaluation dataset.
            documents (List[Document]): A list of additional `Document` objects to be included in the index.
                If None, only the documents in the evaluation dataset will be used.
        """
        self.retriever = retriever
        self.eval_dataset = eval_dataset
        self.documents = documents
        self.id_to_doc = {doc.id: doc for doc in documents}
        # Other variables
        self.records = None

    def build_records(self, save_dir: Text = None, max_relevant: int = 100, batch_size: int = 32, ):
        self.records = []
        # Loop with batch
        for i in tqdm.tqdm(range(0, len(self.eval_dataset), batch_size)):
            batch = self.eval_dataset[i:i + batch_size]
            texts = [sample.query for sample in batch]
            batch_retrieve_documents = self.retriever.rank_batch(texts)
            for sample, top_k_retrieve_document in zip(batch, batch_retrieve_documents):
                record = {
                    'query': sample.query,
                    'answer': sample.answer or '',
                    'top_k_relevant': max_relevant,
                    'relevant_docs': [self.id_to_doc[doc_id].content for doc_id in sample.relevant_docs_id],
                    'relevant_scores': [doc.score for doc in top_k_retrieve_document],
                    'predicted_relevant_docs': [doc.dict() for doc in top_k_retrieve_document]
                }
                self.records.append(record)

        # for sample in tqdm.tqdm(self.eval_dataset, total=len(self.eval_dataset), ):
        #     top_k_retrieve_documents = [doc for doc in self.retriever(sample.query)][:max_relevant]
        #     record = {
        #         'query': sample.query,
        #         'answer': sample.answer or '',
        #         'top_k_relevant': max_relevant,
        #         'relevant_docs': [self.id_to_doc[doc_id].content for doc_id in sample.relevant_docs_id],
        #         'relevant_scores': [doc.score for doc in top_k_retrieve_documents],
        #         'predicted_relevant_docs': [doc.dict() for doc in top_k_retrieve_documents]
        #     }
        #     self.records.append(record)

        if save_dir:
            record_path = os.path.join(save_dir, 'records.json')
            write_json(self.records, record_path)

    def save_records(self, save_dir: Text = None, file_name: Text = "records.json"):
        if not self.records:
            raise ValueError(
                'Records not found. Please run `build_records` or `load_records` before running `save_records`.')
        if not file_name.endswith('.json'):
            file_name += '.json'
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=False)
            record_path = os.path.join(save_dir, file_name)
            write_json(self.records, record_path)

    def load_records(self, record_path: Text):
        self.records = load_json(record_path)

    def run_eval(
            self, report_dir: Text = "reports", threshold: float = 0.0, recall_top_k: List[int] = None,
            file_name: Text = None
    ):
        if not recall_top_k:
            recall_top_k = [5, 10]
        if not self.records:
            raise ValueError(
                'Records not found. Please run `build_records` or `load_records` before running `run_eval`.')
        reports = []
        mrr = 0
        map = 0
        for record, sample in zip(self.records, self.eval_dataset):
            if not sample.relevant_docs_id:
                print(f"Relevant for sample is None {sample.relevant_docs_id}")
                continue
            top_k_retrieve_documents = [Document(**record) for record in record.pop('predicted_relevant_docs')
                                        if record['score'] >= threshold]

            # Get predicts id
            predicted_ids = [doc.id for doc in top_k_retrieve_documents]
            true_ids = sample.relevant_docs_id

            rr_score = 0
            ap_score = 0
            ap = 0
            for id in predicted_ids:
                if id not in true_ids:
                    continue

                ap += 1
                if ap == 1 and rr_score == 0:
                    rr_score = 1 / int(predicted_ids.index(id) + 1)
                else:
                    rr_score = rr_score
                ap_score += ap / (int(predicted_ids.index(id)) + 1)

                mrr = mrr + (rr_score / len(predicted_ids))
                map = map + (ap_score / len(predicted_ids))

            # Update reports
            report = {
                'rr': round(rr_score / len(predicted_ids), 2),
                'ap': round(ap_score / len(predicted_ids), 2),
                "predicted_label": [doc.content for doc in top_k_retrieve_documents],
                **record,
            }
            # Calculate recall
            for k in recall_top_k:
                recall = len(set(predicted_ids[:k]).intersection(set(true_ids))) / len(set(true_ids))
                report.update({f'recall@{k}': round(recall, 2)})
            reports.append(report)
        print_utils.print_style_free(
            f"Mean Reciprocal Rank: {round(mrr / len(self.eval_dataset), 2)}"
        )
        print_utils.print_style_free(
            f"Mean Average Precision: {round(map / len(self.eval_dataset), 2)}"
        )
        for k in recall_top_k:
            print_utils.print_style_free(
                f"Recall@{k}: {round(sum([report[f'recall@{k}'] for report in reports]) / len(reports), 2)}"
            )
        if not reports:
            return reports

        # Save report as json
        if report_dir:
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            report_path = os.path.join(report_dir,
                                       f'{file_name or self.retriever.__class__.__name__}_ir_eval_results.json')
            write_json(reports, report_path)
        # save_detail_report(reports, report_dir, file_name=self.retriever.__call__.__name__)


def save_detail_report(
        df: Union[DataFrame, List],
        output_dir: Optional[Union[str, Path]] = None,
        file_name: Optional[str] = "ir_eval_results"
):
    """
    Saves the evaluation result.
    The result of each node is saved in a separate csv with file name {node_name}.csv to the output_dir folder.

    :param file_name: Model name to benchmark
    :param df: Evaluation results
    :param output_dir: Path to the target folder the csvs will be saved.
    """
    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    output_dir = Path(os.path.join(output_dir, 'details'))
    logger.info(f"Saving evaluation results to {output_dir}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    target_path = output_dir / f"{file_name or 'No Name'}.xlsx"

    if isinstance(df, list):
        df = pd.DataFrame(df)

    # df = pd.DataFrame(df["retriever_docs"])
    df.insert(loc=0, column='index', value=df.index)
    # df.insert(6, 'score', df.pop('relevant_doc_scores'))
    # df.columns = ['index', 'query', 'answer', 'top_k', 'rr', 'ap', 'score', 'relevant_docs', 'predicted_label']
    df.columns = ['index'] + list(df.columns[1:])
    # explode lists of corpus to row
    df = df.apply(pd.Series.explode)

    df_merged = pd.DataFrame(df.to_dict('records'))
    df_merged.score = df_merged.relevant_scores.round(decimals=5)

    df_merged_wrong_queries = pd.DataFrame(df.to_dict('records'))
    df_merged_wrong_queries = df_merged_wrong_queries[
        (df_merged_wrong_queries['relevant_docs'] != df_merged_wrong_queries['predicted_label'])]
    df_merged_wrong_queries = df_merged_wrong_queries.reset_index()
    df_merged_wrong_queries.pop('level_0')
    df_merged_wrong_queries.relevant_scores = df_merged_wrong_queries.relevant_scores.round(decimals=2)

    writer = pd.ExcelWriter(f'{target_path}', engine='xlsxwriter')
    df_merged.to_excel(writer, sheet_name='Detail_Report', index=False)
    # df_merged_wrong_queries.to_excel(writer, sheet_name='Wrong_Query', index=False)

    workbook = writer.book

    header_format = workbook.add_format(
        {
            'align': 'center',
            'valign': 'vcenter',
            'border': 1,
            'bg_color': '#fff2cc'
        }
    )

    cell_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
    merge_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})
    bg_format_odd = workbook.add_format({'bg_color': '#cfe2f3', 'border': 1, 'align': 'center'})
    bg_format_even = workbook.add_format({'bg_color': '#FFFFFF', 'border': 1, 'align': 'center'})
    bg_format_wrong = workbook.add_format({'font_color': 'red'})
    bg_format_correct_label = workbook.add_format({'align': 'center'})
    bg_format_incorrect_label = workbook.add_format({'font_color': 'red', 'align': 'center'})

    bg_best_score = workbook.add_format({'bold': True, 'align': 'center'})

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
                if dt['answer'][i - 1] != dt['predicted_label'][i - 1]:
                    worksheet.write(
                        i, dt.columns.get_loc('relevant_docs'),
                        dt['relevant_docs'][i - 1], bg_format_wrong)
                    worksheet.write(
                        i, dt.columns.get_loc('predicted_label'),
                        dt['predicted_label'][i - 1], bg_format_wrong)
                    worksheet.write(
                        i, dt.columns.get_loc('relevant_scores'),
                        dt['relevant_scores'][i - 1], bg_format_incorrect_label)
                else:
                    if float(dt['relevant_scores'][i - 1]) >= 0.9:
                        worksheet.write(
                            i, dt.columns.get_loc('score'),
                            dt['score'][i - 1], bg_best_score)
                    else:
                        worksheet.write(
                            i, dt.columns.get_loc('relevant_scores'),
                            dt['relevant_scores'][i - 1], bg_format_correct_label)

            # column to merge or reformat
            column_index = {
                'index': 0,
                'query': 1,
                'answer': 2,
                'top_k': 3,
                'rr': 4,
                'ap': 5
            }

            for key, index in column_index.items():
                if len(u) < 2:
                    # pass  # do not merge cells if there is only one row
                    crange = xl_range_abs(u[0], column_index['index'], u[-1], len(column_index))
                    worksheet.conditional_format(crange, {'type': 'no_blanks',
                                                          'format': cell_format})
                else:
                    # merge cells using the first and last indices
                    worksheet.merge_range(u[0], index, u[-1], index, dt.loc[u[0], f'{key}'],
                                          merge_format)

    # auto-adjust column size
    for column in df:
        column_width = max(df[column].astype(str).map(len).max(), len(column))
        col_idx = df.columns.get_loc(column)
        writer.sheets['Detail_Report'].set_column(col_idx, col_idx, column_width)
        # writer.sheets['Wrong_Query'].set_column(col_idx, col_idx, column_width)

    formatter(df_merged, 'Detail_Report')
    # formatter(df_merged_wrong_queries, 'Wrong_Query')
    writer.close()
    logger.info(f"Evaluation report with excel format is saved at {target_path}")
