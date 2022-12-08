import streamlit as st
import numpy as np
from pandas import DataFrame
import seaborn as sns
import os
import json
from saturn.kr_manager import KRManager
from typing import List

CONFIG_DEFAUT = "config/dummy/config_mini_sbert.yaml"


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

@st.experimental_singleton
def get_model(input_model_name):
    print("abc", input_model_name)
    kr_loader = KRManager(CONFIG_DEFAUT)
    kr_loader.embedder.load_model(pretrained_name_or_abspath = input_model_name)
    print(kr_loader.model_name_or_path)
    return kr_loader


def check_input(input_doc):
    if input_doc is not None:
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


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    st.title("🔑 Knowledge retriever")

with st.expander("ℹ️ Introduce", expanded=True):
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
    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 2, 0.07])
    with c1:
        model_name = st.text_input(
            "Paste your model name",
            help="Choose model from Huggingface or Axiom",
        )

        top_N = st.slider(
            "# top_k",
            min_value=1,
            max_value=50,
            value=10,
            help="You can choose the number of relevant results to display. Between 1 and 50, default number is 10",
        )

    with c2:
        doc = st.text_input(
            "Paste your text below (max 200 words)",
        )

        MAX_WORDS = 200
        import re

        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 200 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )
            doc = doc[:MAX_WORDS]
        corpus_uploader = st.file_uploader("Choose a txt file", accept_multiple_files=False)

        submit_button = st.form_submit_button(label="✨ Get relevants sentences!")

# if load_model_button:
#     try:
#         kr = get_model(model_name)
#         if kr is None:
#             st.error("Model not found")
#             st.stop()
#         print((f"Model {model_name} loaded successfully"))
#     except Exception as e:
#         st.error("Model not found: Error: {}".format(e))
#         st.stop()
#     st.success("Model loaded successfully")

def get_inference(input_doc, input_corpus, top_N, kr):
    input_doc = check_input(input_doc)
    input_corpus = check_corpus(input_corpus)
    inference_docs = kr.inference(input_doc, input_corpus, top_N)
    return inference_docs

if submit_button:
    kr = get_model(model_name)
    inference_docs = get_inference(doc, corpus_uploader, top_N, kr)


    st.markdown("## 🎈 **Check & download results**")
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

# input_doc = check_input(doc)
# corpus = check_corpus(corpus_uploader)
# inference_docs = kr.inference(doc, corpus, top_N)

# merge os pat



