import os
import shutil
import time
from pathlib import Path
from random import random

from typing import Union
import pandas as pd

import streamlit as st
from comet.lib.file_util import zip_folder
from streamlit_tags import st_tags
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode

from saturn.kr_manager import KRManager
from ui import MODELS
from ui.utils.io import get_json, load_docs

DEFAULT_CONFIG_AT_STARTUP = os.getenv(
    "DEFAULT_CONFIG_AT_STARTUP", "config/dummy/config_mini_sbert.yaml")
if 'overall_report' not in st.session_state:
    st.session_state['overall_report'] = None
if 'detail_report' not in st.session_state:
    st.session_state['detail_report'] = None

kr = KRManager(DEFAULT_CONFIG_AT_STARTUP)


def trigger_on_click(folder_path: Union[str, Path]):
    time.sleep(1)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


st.set_page_config(
    page_title="Evaluation",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Evaluation")

with st.form("eval model") as eval_form:
    eval_models = st_tags(
        label='Enter model:',
        text='Press enter to add more',
        value=['vinai/phobert-base'],
        suggestions=MODELS,
        maxtags=5,
        key='eval_model')
    col_1, col_2 = st.columns(2)
    with col_1:
        eval_query = st.file_uploader(
            label='Upload query file',
            type=['json'], key='2',
            accept_multiple_files=True
        )
    with col_2:
        eval_corpus = st.file_uploader(
            label='Upload corpus file',
            type=['json'],
            key='3',
            accept_multiple_files=True
        )
    summit_button = st.form_submit_button()


# @st.experimental_singleton
def lazy_init():
    models_name = list(set(eval_models))
    queries_json = get_json(eval_query)
    corpus_json = get_json(eval_corpus)
    queries = load_docs(queries_json, corpus_json)
    corpus = load_docs(corpus_json)

    kr._model_name_or_path = models_name
    kr._corpus_docs = None
    kr._corpus_docs = corpus
    kr._query_docs = None
    kr._query_docs = queries

    return kr


result_type = st.selectbox('Select result type', ['',
                                                  'Display detail report',
                                                  'Display overall result',
                                                  'Download overall report',
                                                  'Download detail report',
                                                  'Download all reports'], key='4')

if summit_button:
    kr = lazy_init()
    retriever_results, retriever_top_k_results = kr.evaluate_embedder()

    st.session_state['overall_report'] = retriever_results
    st.session_state['detail_report'] = retriever_top_k_results

if result_type == "Display overall result":
    if st.session_state['overall_report'] is not None:
        results = st.session_state['overall_report']
        df = pd.DataFrame(results)
        AgGrid(df, key=random())
        st.success('Compute metrics done!')
    else:
        st.error('No file to show', icon="🚨")

if result_type == "Display detail report":
    theme = st.sidebar.selectbox('Theme', ['light', 'dark', 'blue', 'fresh', 'material'], key='5')
    grid_options = {
        "columnDefs": [
            {
                "headerName": "index",
                "field": "query",
                "tooltipField": 'query'

            }
        ]
    }

    if st.session_state['detail_report'] is not None:
        results = st.session_state['detail_report']
        my_bar = st.progress(0)
        percentage = 0
        for idx, models in enumerate(results):
            percentage = (idx + 1) / len(results)
            my_bar.progress(percentage)
            for model, df in models.items():
                if isinstance(df, list):
                    df = pd.DataFrame(df)
                # df = df.drop(columns=['query_id'])
                df.insert(loc=0, column='index', value=df.index)

                # explode lists of corpus to row
                df = df.apply(pd.Series.explode)
                with st.container():
                    st.write(
                        f"<h2 style='text-align: center; color: red; font-size:20px;'>Results for {model}</h2>",
                        unsafe_allow_html=True)
                    df_merged = pd.DataFrame(df.to_dict('records'))
                    ob = GridOptionsBuilder.from_dataframe(df_merged)
                    ob.configure_column('query', header_name='query', rowGroup=True, enableRowGroup=True,
                                        dragAndDrop=True, tooltip=True)
                    ob.configure_side_bar()
                    ob.configure_selection("single")
                    ob.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
                    gridOptions = ob.build()
                    AgGrid(df_merged,
                           gridOptions=gridOptions,
                           data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                           allow_unsafe_jscode=True,
                           theme=theme,
                           key=random(),
                           enable_enterprise_modules=True)
        st.success('Compute metrics done!')
    else:
        st.error('No file to show', icon="🚨")

if result_type == 'Download overall report':
    if st.session_state['overall_report'] is not None:
        df = pd.DataFrame(st.session_state['overall_report']).to_csv()
        st.download_button(label=' 📥 Download overall report',
                           data=df,
                           file_name='knowledge_retrieval.csv')
    else:
        st.error('No file to download', icon="🚨")

if result_type == 'Download detail report':
    if st.session_state['detail_report'] is not None:

        results = st.session_state['detail_report']
        kr = lazy_init()

        # create tempt folder to compress all reports
        dt = str(time.time())
        temp_path = os.path.join('reports', dt)
        os.mkdir(temp_path)
        for idx, models in enumerate(results):
            for model, df in models.items():
                kr.save_detail_report(output_dir=temp_path, model_name=model, df=df)
        time.sleep(5)
        zip_path = zip_folder(folder=temp_path)
        with open(zip_path, "rb") as fp:
            btn = st.download_button(
                label="📥 Download detail reports",
                data=fp,
                file_name=f"{zip_path}",
                mime="application/zip",
                on_click=trigger_on_click(temp_path)
            )
    else:
        st.error('No file to download', icon="🚨")

if result_type == "Download all reports":
    kr = lazy_init()

    # create tempt folder to compress all reports
    dt = str(time.time())
    temp_path = os.path.join('reports', dt)
    os.mkdir(temp_path)
    if st.session_state['detail_report'] is not None:
        results = st.session_state['detail_report']
        for idx, models in enumerate(results):
            for model, df in models.items():
                kr.save_detail_report(output_dir=temp_path, model_name=model, df=df)
    if st.session_state['overall_report'] is not None:
        df = pd.DataFrame(st.session_state['overall_report'])
        kr.save_overall_report(output_dir=temp_path, df=df, save_markdown=True)

    time.sleep(5)
    zip_path = zip_folder(folder=temp_path)
    with open(zip_path, "rb") as fp:
        btn = st.download_button(
            label="📥 Download all reports",
            data=fp,
            file_name=f"{zip_path}",
            mime="application/zip",
            on_click=trigger_on_click(temp_path)
        )
