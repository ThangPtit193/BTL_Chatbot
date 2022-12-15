import os

import streamlit as st
import pandas as pd
import seaborn as sns

from saturn.kr_manager import KRManager

from ui import MODELS
from ui.utils import check_input, check_corpus
from st_aggrid import AgGrid


DEFAULT_MODEL_AT_STARTUP = os.getenv("DEFAULT_MODEL_AT_STARTUP", "vinai/phobert-base")
DEFAULT_TOP_K_AT_STARTUP = os.getenv("DEFAULT_TOP_K_AT_STARTUP", 10)
DEFAULT_CONFIG_AT_STARTUP = os.getenv("DEFAULT_CONFIG_AT_STARTUP", "config/dummy/config_mini_sbert.yaml")
DEFAULT_MAX_WORDS = os.getenv("DEFAULT_MAX_WORDS_AT_STARTUP", 50)
# INPUT query
DEFAULT_INPUT_QUERY = os.getenv("DEFAULT_INPUT_QUERY", "Hôm nay tôi đi học")
# INPUT corpus
DEFAULT_INPUT_CORPUS = os.getenv("DEFAULT_INPUT_CORPUS", "data/sample_corpus.txt")


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def main():
    set_state_if_absent("model_name", DEFAULT_MODEL_AT_STARTUP)
    set_state_if_absent("top_k", DEFAULT_TOP_K_AT_STARTUP)
    set_state_if_absent("config", DEFAULT_CONFIG_AT_STARTUP)
    set_state_if_absent("max_words", DEFAULT_MAX_WORDS)
    set_state_if_absent("input_query", DEFAULT_INPUT_QUERY)
    set_state_if_absent("input_corpus", DEFAULT_INPUT_CORPUS)

    st.set_page_config(
        page_title="Knowledge Retrieval",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get help': 'https://trello.com/b/8zAozWIB/knowledge-retriever',
            'Report a bug': 'https://ftech.ai/',
            'About': 'The services is applied for knowledge retrieval and infer models quickly developed by VA TEAM'
        }
    )
    st.title("🤖 Knowledge Retrieval ")

    with st.expander("ℹ️ Introduce", expanded=True):
        st.write(
            """     
    -   The Knowledge Retrieval will search the relevant sentences/paragraphs from query
    -   You can use it to experiment models from Huggingface or Axiom
        """
        )

        st.markdown("")

    st.markdown("")
    st.markdown("## 📌 **Query and Relevant docs** ##")
    st.markdown(
        """
        Enter model name, top_k retrieval, query and corpus to get relevant documents
        """,
        unsafe_allow_html=True,
    )

    model_options = [model for model in MODELS] + ["Your model ..."]
    model_name = st.selectbox("Choose model from Axiom Hub", options=model_options)

    if model_name == "Your model ...":
        model_name = st.text_input(
            label="Enter your model from local or HuggingFace",
            value=DEFAULT_MODEL_AT_STARTUP,
            help="Choose model from Huggingface or Axiom",
        )
    st.info(f":white_check_mark: The selected option is {model_name} ")
    with st.form(key="my_form"):

        top_N = st.slider(
            "Top k",
            min_value=1,
            max_value=50,
            value=DEFAULT_TOP_K_AT_STARTUP,
            help="You can choose the number of relevant results to display. Between 1 and 50, default number is 10",
        )
        # center the button
        doc = st.text_input(
            label="Paste your text below (max 50 words)",
            value=DEFAULT_INPUT_QUERY,
            help="Input your query here",
        )

        import re

        res = len(re.findall(r"\w+", doc))
        if res > DEFAULT_MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 50 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )
            doc = doc[:DEFAULT_MAX_WORDS]

        tab_1, tab_2 = st.tabs(["Input docs", "Input txt file"])
        with tab_1:
            df_template = pd.DataFrame(
                '',
                index=range(6),
                columns=['Samples text']
            )
            response_samples = AgGrid(df_template, editable=True, fit_columns_on_grid_load=True, key='sample', height=203)
        with tab_2:
            corpus_uploader = st.file_uploader(
                label="Choose a txt file",
                type=["txt"],
                help="You can upload a txt file to get the relevant sentences. The file should contain one sentence per line.",
                key="corpus_uploader",
                accept_multiple_files=False)

        submit_button = st.form_submit_button(label="✨ Get relevant sentences!")

    def get_corpus(response_samples, input_corpus):
        response_samples = response_samples['data']['Samples text'].tolist()
        response_samples = [x for x in response_samples if x != '']
        input_corpus = check_corpus(input_corpus)

        if input_corpus and response_samples:
            corpus = input_corpus + response_samples
        elif input_corpus:
            corpus = input_corpus
        elif response_samples:
            corpus = response_samples
        else:
            st.error("Please input corpus or samples")
            st.stop()
        return corpus

    def get_inference(input_doc, response_samples, input_corpus, input_top_k, input_model):
        input_doc = check_input(input_doc)
        # input_corpus = check_corpus(input_corpus)
        corpus = get_corpus(response_samples, input_corpus)
        return input_model.inference(input_doc, corpus, input_top_k)

    @st.experimental_memo(max_entries=3, ttl=60 * 3)
    def get_model(input_model_name):
        # check if cache is full
        kr_loader = KRManager(DEFAULT_CONFIG_AT_STARTUP)
        try:
            kr_loader.embedder.load_model(pretrained_name_or_abspath=input_model_name)
        except Exception as e:
            st.error("Model not found: Error: {}".format(e))
            st.stop()
        return kr_loader

    if submit_button:
        kr = get_model(model_name)
        inference_docs = get_inference(doc, response_samples, corpus_uploader, top_N, kr)

        st.markdown("## 🎈 **Check results**")
        df = (
            pd.DataFrame(inference_docs)
        )
        # styling
        df.index += 1
        cmGreen = sns.light_palette("green", as_cmap=True)

        df = df.style.background_gradient(cmap=cmGreen, subset=["score"])
        df = df.set_properties(**{"text-align": "left"})

        format_dictionary = {
            "score": "{:.3}",
        }

        df = df.format(format_dictionary)
        st.table(df)
        st.stop()

main()
