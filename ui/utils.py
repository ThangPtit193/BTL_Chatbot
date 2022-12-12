import streamlit as st
from typing import List


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
    """
    Check if input is not empty
    """
    if input_doc is not None and input_doc != "":
        return input_doc
    else:
        st.error("Please paste your text")
        st.stop()

def check_corpus(input_corpus_loader):
    """
    Check if input is not empty
    """
    if input_corpus_loader is not None:
        return read_txt_streamlit(input_corpus_loader)
    else:
        st.error("Please upload a txt file")
        st.stop()