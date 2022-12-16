import streamlit as st
import pandas as pd
from ui import MODELS
from typing import List
import json
import os
from saturn.kr_manager import KRManager
from comet.lib import file_util
from saturn.components.utils.document import Document
from comet.utilities.utility import convert_unicode
from streamlit_tags import st_tags
from ui.utils import filter_dataframe


DEFAULT_CONFIG_AT_STARTUP = os.getenv(
    "DEFAULT_CONFIG_AT_STARTUP", "config/dummy/config_mini_sbert.yaml")
if 'retriever_results' not in st.session_state:
    st.session_state['retriever_results'] = None
if 'retriever_top_k_results' not in st.session_state:
    st.session_state['retriever_top_k_results'] = None


def load_docs(data_docs, corpus=None) -> List[Document]:
    """
    Load documents from a file or a directory
    """
    if not isinstance(data_docs, dict):
        raise FileNotFoundError(f"File not valid")

    docs = []
    for unique_intent, query_list in data_docs.items():
        if corpus:
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


st.set_page_config(
    page_title="Evaluation",
    page_icon="🤖",
    layout="wide",
)


def get_json(json_files):
    json_list = {}
    for json_file in json_files:
        temp_json = json.loads(json_file.read())
        json_list.update(temp_json)
    return json_list


st.title("🤖 Evaluation")

with st.form("eval model") as eval_form:
    eval_model = st_tags(
        label='Enter model:',
        text='Press enter to add more',
        value=['vinai/phobert-base'],
        suggestions=['vinai/phobert-base'],
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

if summit_button:
    model_list = eval_model
    queries_json = get_json(eval_query)
    corpus_json = get_json(eval_corpus)
    queries = load_docs(queries_json, corpus_json)
    corpus = load_docs(corpus_json)

    kr = KRManager(DEFAULT_CONFIG_AT_STARTUP)
    kr._model_name_or_path = model_list

    kr._corpus_docs = corpus
    kr._queries = queries
    retriever_results, retriever_top_k_results = kr.evaluate_embedder()

    st.session_state['retriever_results'] = retriever_results
    # print(retriever_top_k_results)
    df = pd.DataFrame(retriever_top_k_results)
    # df = pd.DataFrame(retriever_top_k_results[0]['all-MiniLM-L6-v2'])
    if isinstance(df, list):
        df = pd.DataFrame(df)
    # df = df.drop(columns=['query_id'])
    # df.insert(loc=0, column='index', value=df.index)

    # # explode lists of corpus to row
    df = df.apply(pd.Series.explode)
    print(df)
    # df_merged = pd.DataFrame(df.to_dict('records'))
    # print(df_merged)
    # st.dataframe(retriever_top_k_results[0]['phobert-base'])
    # st.session_state['retriever_top_k_results'] = retriever_top_k_results
    kr._save_overall_report(
        output_dir='reports',
        df=pd.DataFrame(retriever_results),
        save_markdown=False,
    )
    for models in retriever_top_k_results:
        for model, data in models.items():
            kr._save_detail_report(output_dir="reports",
                                   model_name=model, df=data)


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


result_type = st.selectbox('Select result type', [
                           'Retriever results', 'Detail retriever results'], key='4')
if result_type == 'Retriever results':
    # st.session_state['df_1']
    if st.session_state['retriever_results'] is not None:
        st.dataframe(st.session_state['retriever_results'])
        # st.download_button('Retriever results', convert_df(st.session_state['retriever_results'][0]), 'download.csv')
        with open('reports/knowledge_retrieval.csv') as f:
            st.download_button('Retriever results', f,
                               'knowledge_retrieval.csv')
if result_type == 'Detail retriever results':
    if st.session_state['retriever_top_k_results'] is not None:
        details_result = st.session_state['retriever_top_k_results']
        print(type(details_result[0]))
        model_list_details = list(details_result[0].keys())
        key_select = st.selectbox(
            'Select result type', model_list_details, key='5')
        details_result[0][key_select]
        df_key_select = pd.DataFrame(details_result[0][key_select])
        st.dataframe(df_key_select)

        with open(f'reports/{key_select}.xlsx') as f:
            st.download_button('Retriever results', f, f'{key_select}.xlsx')
