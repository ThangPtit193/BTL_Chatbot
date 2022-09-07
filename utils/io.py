import json
from hashlib import md5
from typing import List, Any, Text, Union, Dict, Set

from config import DEFAULT_ENCODING


def deep_container_fingerprint(
        obj: Union[List[Any], Dict[Any, Any], Any], encoding: Text = DEFAULT_ENCODING
) -> Text:
    """Calculate a hash which is stable, independent of a containers key order.
    Works for lists and dictionaries. For keys and values, we recursively call
    `hash(...)` on them. Keep in mind that a list with keys in a different order
    will create the same hash!
    Args:
        obj: dictionary or list to be hashed.
        encoding: encoding used for dumping objects as strings
    Returns:
        hash of the container.
    """
    if isinstance(obj, dict):
        return get_dictionary_fingerprint(obj, encoding)
    elif isinstance(obj, list):
        return get_list_fingerprint(obj, encoding)
    elif hasattr(obj, "fingerprint") and callable(obj.fingerprint):
        return obj.fingerprint()
    else:
        return get_text_hash(str(obj), encoding)


def get_dictionary_fingerprint(
        dictionary: Dict[Any, Any], encoding: Text = DEFAULT_ENCODING
) -> Text:
    """Calculate the fingerprint for a dictionary.
    The dictionary can contain any keys and values which are either a dict,
    a list or elements which can be dumped as a string.
    Args:
        dictionary: dictionary to be hashed
        encoding: encoding used for dumping objects as strings
    Returns:
        The hash of the dictionary
    """
    string_field = json.dumps(
        {
            deep_container_fingerprint(k, encoding): deep_container_fingerprint(
                v, encoding
            )
            for k, v in dictionary.items()
        },
        sort_keys=True,
    )
    return get_text_hash(string_field, encoding)


def get_list_fingerprint(
        elements: List[Any], encoding: Text = DEFAULT_ENCODING
) -> Text:
    """Calculate a fingerprint for an unordered list.
    Args:
        elements: unordered list
        encoding: encoding used for dumping objects as strings
    Returns:
        the fingerprint of the list
    """
    string_field = json.dumps(
        [deep_container_fingerprint(element, encoding) for element in elements]
    )
    return get_text_hash(string_field, encoding)


def get_text_hash(text: Text, encoding: Text = DEFAULT_ENCODING) -> Text:
    """Calculate the md5 hash for a text."""
    return md5(text.encode(encoding)).hexdigest()  # noqa


if __name__ == "__main__":
    doc_test = {
        "text": "Test duplicate",
        "meta": {
            "answer": "utter_science_define {'trigger_slot': 'TE_BAO'}",
            "adjacency_pair": "ask_define/utter_science_define {'trigger_slot': 'TE_BAO'}",
            "domain": "science",
            "index": "science"
        }
    }
