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
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, JsCode

from saturn.kr_manager import KRManager
from ui import MODELS
from ui.utils.io import get_json, load_docs

DEFAULT_CONFIG_AT_STARTUP = os.getenv(
    "DEFAULT_CONFIG_AT_STARTUP", "config/dummy/config_mini_sbert.yaml")
if 'overall_report' not in st.session_state:
    st.session_state['overall_report'] = None
if 'detail_report' not in st.session_state:
    st.session_state['detail_report'] = None

if 'Download all reports' not in st.session_state:
    st.session_state['Download_all_reports'] = None


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
with st.expander("ℹ️ Introduce", expanded=True):
    st.markdown("### Evaluation - Knowledge Retrieval ### ")
    st.markdown('INSTRUCTION: Select models, query and corpus to evaluate')
    st.markdown("""Query and corpus must be in json format""")

with st.expander("📂 Download sample files", expanded=False):
    st.markdown("You can download sample json file here")
    with open("data/eval-data/dummy/corpus_docs.json", "r") as f:
        st.download_button("Download corpus file", f.read(), file_name="corpus_docs.json", mime="application/json")
    with open("data/eval-data/dummy/query_docs.json", "r") as f:
        st.download_button("Download query file", f.read(), file_name="query_docs.json", mime="application/json")
    

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
    summit_button = st.form_submit_button(label = 'Evaluate')


# @st.experimental_singleton
def lazy_init():
    models_name = list(set(eval_models))
    queries_json = get_json(eval_query)
    corpus_json = get_json(eval_corpus)
    queries = load_docs(queries_json, corpus_json)
    corpus = load_docs(corpus_json)

    kr._pretrained_name_or_abspath = models_name
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
    if len(eval_query) == 0 or len(eval_corpus) == 0:
        st.error('Please upload query and corpus file', icon="🚨")
        st.stop()
    kr = lazy_init()
    retriever_results, retriever_top_k_results = kr.evaluate_embedder()
    st.success('Evaluate done! please check the result below')
    st.session_state['overall_report'] = retriever_results
    st.session_state['detail_report'] = retriever_top_k_results
    st.session_state['Download_all_reports'] = True


if result_type == "Display overall result":
    if st.session_state['overall_report'] is not None:
        results = st.session_state['overall_report']
        df = pd.DataFrame(results)
        AgGrid(df, key=random())
        st.success('Compute metrics done!')
    else:
        st.error('No file to show', icon="🚨")
def format_color_groups(values):
    if isinstance(values, float) and values > 0:
        return  'color:red;border-collapse: collapse; border: 1px solid black;'
    else:
        return 'border-collapse: collapse; border: 1px solid black;'
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
                    # df_merged = df_merged.style.highlight_max(axis=0)
                    # df_merged = df_merged.style.applymap(format_color_groups)
                    select_option = st.selectbox('Select option', ['Show group', 'Show all with wrong results'], key='6')
                    if select_option == 'Show group':
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
                    if select_option == 'Show all with wrong results':
                        ob = GridOptionsBuilder.from_dataframe(df_merged)
                        cellstyle_jscode = JsCode("""
                        function(params) {
                            console.log(params);
                            if (params.data.predicted_labels != params.data.label) {
                                return {
                                    'color': 'red',
                                }
                            }
                        }
                        """)
                        
                        ob.configure_grid_options(getRowStyle=cellstyle_jscode)
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
    if st.session_state['Download_all_reports'] is not None:
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
    else:
        st.error('No file to download', icon="🚨")