import json

import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from typing import List
import streamlit as st

from comet.utilities.utility import convert_unicode

from saturn.components.utils.document import Document


def get_json(json_files):
    json_list = {}
    for json_file in json_files:
        temp_json = json.loads(json_file.read())
        json_list.update(temp_json)
    return json_list


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
        for idx, query in enumerate(query_list):
            docs.append(Document(
                text=convert_unicode(query),
                label=unique_intent,
                num_relevant=num_relevant,
            ))
    return docs


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
        return None


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]
    return df
