import streamlit as st
import numpy as np
from pandas import DataFrame
import seaborn as sns
import os
import json
from saturn.kr_manager import KRManager
from ui.utils import check_input, check_corpus


DEFAULT_MODEL_AT_STARTUP = os.getenv("DEFAULT_MODEL_AT_STARTUP", "vinai/phobert-base")
DEFAULT_TOP_K_AT_STARTUP = os.getenv("DEFAULT_TOP_K_AT_STARTUP", 10)
DEFAULT_CONFIG_AT_STARTUP = os.getenv("DEFAULT_CONFIG_AT_STARTUP", "config/dummy/config_mini_sbert.yaml")
DEFAULT_MAX_WORDS = os.getenv("DEFAULT_MAX_WORDS_AT_STARTUP", 50)
# INPUT query
DEFAULT_INPUT_QUERY = os.getenv("DEFAULT_INPUT_QUERY", "Hôm nay tôi đi học")
# INPUT corpus
DEFAULT_INPUT_CORPUS = os.getenv("DEFAULT_INPUT_CORPUS", "data/sample_corpus.txt")

# fschool-distilbert-multilingual-faq-v8.0.0

# kr.load_model(model_name_or_path="fschool-distilbert-multilingual-faq-v8.0.0"

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
        page_title="Knowledge retriever",
        page_icon="🎈",
    )
    st.title("🔑 Knowledge retriever ")

    with st.expander("ℹ️ Introduce", expanded=True):
        st.write(
            """     
    -   The Knowledge retriever will get the relevant sentences/paragraphs from query
    -   You can use it to experiment models from Huggingface or Axiom
        """
        )

        st.markdown("")

    st.markdown("")
    st.markdown("## 📌 **Query and Relevant docs** ##")
    st.markdown(
        """
        Input model name, top k, query and corpus to get the top k relevant docs
        """,
        unsafe_allow_html=True,
    )
    
    with st.form(key="my_form"):
        model_name = st.text_input(
            label="Input your model name",
            value=DEFAULT_MODEL_AT_STARTUP,
            help="Choose model from Huggingface or Axiom",
        )

        top_N = st.slider(
            "Top k",
            min_value=1,
            max_value=50,
            value=DEFAULT_TOP_K_AT_STARTUP,
            help="You can choose the number of relevant results to display. Between 1 and 50, default number is 10",
        )
        # center the button

        doc = st.text_input(
            label = "Paste your text below (max 50 words)",
            value = DEFAULT_INPUT_QUERY,
            help = "Input your query here",
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
        corpus_uploader = st.file_uploader(
            label="Choose a txt file",
            type=["txt"],
            help="You can upload a txt file to get the relevant sentences. The file should contain one sentence per line.",
            key="corpus_uploader",
            accept_multiple_files=False)

        submit_button = st.form_submit_button(label="✨ Get relevants sentences!")

    def get_inference(input_doc, input_corpus, input_top_k, input_model):
        input_doc = check_input(input_doc)
        input_corpus = check_corpus(input_corpus)
        return input_model.inference(input_doc, input_corpus, input_top_k)

    @st.experimental_memo(max_entries = 3, ttl = 60*3)
    def get_model(input_model_name):
        # check if cache is full
        kr_loader = KRManager(DEFAULT_CONFIG_AT_STARTUP)
        try:
            kr_loader.embedder.load_model(pretrained_name_or_abspath = input_model_name)
        except Exception as e:
            st.error("Model not found: Error: {}".format(e))
            st.stop()
        return kr_loader

    if submit_button:
        kr = get_model(model_name)
        inference_docs = get_inference(doc, corpus_uploader, top_N, kr)


        st.markdown("## 🎈 **Check results**")
        df = (
            DataFrame(inference_docs)
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