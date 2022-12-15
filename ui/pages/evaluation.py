import os

import streamlit as st
import seaborn as sns
import pandas as pd
from ui import MODELS

from saturn.kr_manager import KRManager

# from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder
from streamlit_tags import st_tags
from ui.utils import filter_dataframe
import json

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
        value=['Vinai/Phobert-base'],
        suggestions=['Vinai/Phobert-base'],
        maxtags = 5,
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

    result_type = st.selectbox('Select result type', ['All revelant', 'Result ranking'], key='4')

    summit_button = st.form_submit_button()
if not summit_button:
    st.stop()

if summit_button:
    model_list = eval_model
    queries_json = get_json(eval_query)
    corpus_json = get_json(eval_corpus)

df = pd.read_csv('ui/pages/retriever.csv') # dummy


if result_type == 'All revelant':
    # todo: Eval all revelant
    _, _, col_3 = st.columns(3)
    with col_3:
        st.download_button(
            label='Download all revelant',
            data=df.to_csv(index=False),
            file_name='all_revelant.csv',
            mime='text/csv',
        )
    df = pd.read_csv('ui/pages/retriever.csv') # dummy
    st.dataframe(df)
if result_type == 'Result ranking':
    st.dataframe(filter_dataframe(df))




