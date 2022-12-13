import streamlit as st
import numpy as np
from pandas import DataFrame
import seaborn as sns
import os
import json
from saturn.kr_manager import KRManager
from typing import List

CONFIG_DEFAUT = "config/dummy/config_mini_sbert.yaml"


# fschool-distilbert-multilingual-faq-v8.0.0

# kr.load_model(model_name_or_path="fschool-distilbert-multilingual-faq-v8.0.0"

def read_txt_streamlit(file) -> List:
    """
    Read txt file from streamlit

    Args:
        file: file from streamlit

    Returns: list of lines

    """
    lines = []
    for line in file:
        lines.append(line.decode("utf-8"))
    return lines


def check_input(input_doc):
    if input_doc is not None and input_doc != "":
        return input_doc
    else:
        st.error("Please paste your text")
        st.stop()


def check_corpus(input_corpus_loader):
    if input_corpus_loader is not None:
        return read_txt_streamlit(input_corpus_loader)
    else:
        st.error("Please upload a txt file")
        st.stop()


st.set_page_config(
    page_title="Knowledge retriever",
    page_icon="🎈",
)

# st.sidebar.title("🔑 Knowledge retriever")


st.title("🔑 Knowledge retriever ")

with st.expander("ℹ️ Introduce", expanded=True, ):
    st.write(
        """     
-   The Knowledge retriever will get the relevant sentences/paragraphs from query
-   You can use it to experiment models from Huggingface or Axiom
    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## 📌 **Paste document** ##")
with st.form(key="my_form"):
    model_name = st.text_input(
        "Paste your model name",
        help="Choose model from Huggingface or Axiom",
    )

    top_N = st.slider(
        "Top k",
        min_value=1,
        max_value=50,
        value=10,
        help="You can choose the number of relevant results to display. Between 1 and 50, default number is 10",

    )

    clear_memo = st.form_submit_button("Clear Cache")

    doc = st.text_input(
        "Paste your text below (max 50 words)",
    )

    MAX_WORDS = 50
    import re

    res = len(re.findall(r"\w+", doc))
    if res > MAX_WORDS:
        st.warning(
            "⚠️ Your text contains "
            + str(res)
            + " words."
            + " Only the first 50 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
        )
        doc = doc[:MAX_WORDS]
    corpus_uploader = st.file_uploader("Choose a txt file", accept_multiple_files=False)

    submit_button = st.form_submit_button(label="✨ Get relevants sentences!")


def get_inference(input_doc, input_corpus, input_top_k, input_model):
    input_doc = check_input(input_doc)
    input_corpus = check_corpus(input_corpus)
    return input_model.inference(input_doc, input_corpus, input_top_k)


@st.experimental_memo(max_entries=2, ttl=60 * 3)
def get_model(input_model_name):
    kr_loader = KRManager(CONFIG_DEFAUT)
    try:
        kr_loader.embedder.load_model(pretrained_name_or_abspath=input_model_name)
    except Exception as e:
        st.error("Model not found: Error: {}".format(e))
        st.stop()
    return kr_loader


if clear_memo:
    # Clear values from *all* memoized functions:
    st.experimental_memo.clear()
    st.experimental_rerun()

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
    cmRed = sns.light_palette("red", as_cmap=True)

    df = df.style.background_gradient(cmap=cmGreen, subset=["score"])

    format_dictionary = {
        "score": "{:.2}",
    }

    df = df.format(format_dictionary)
    st.table(df)
    st.stop()
