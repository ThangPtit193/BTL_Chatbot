import json
from typing import List

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
        for query in query_list:
            docs.append(Document(
                text=convert_unicode(query),
                label=unique_intent,
                num_relevant=num_relevant,
            ))
    return docs
