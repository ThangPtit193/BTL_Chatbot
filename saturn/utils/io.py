import codecs
import json
import os
import sys
from typing import Dict, Text

import jsbeautifier
import requests
from tqdm import tqdm


def load_json(file_path: Text) -> Dict:
    """
    Load content from json file

    Args:
        file_path (Text): json path

    Returns: a dictionary

    """
    with open(file_path, 'r') as f:
        config = json.load(f)

    return config


def write_json(data, file_path, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as pf:
        json.dump(data, pf, ensure_ascii=False, indent=4)


def write_json_beautifier(file_path: Text, dict_info: Dict) -> None:
    """
    Write the content from dictionary into file with a beautiful format

    Args:
        file_path (Text): The file path
        dict_info (Dict): Dict will be dumped

    Returns:

    """
    opts = jsbeautifier.default_options()
    opts.indent_size = 4
    dict_ = jsbeautifier.beautify(json.dumps(dict_info, ensure_ascii=False), opts)
    with codecs.open(file_path, 'w', 'utf-8') as f:
        f.write(dict_)


def http_get(url, path):
    """
    Downloads a URL to a given path on disc
    """
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code),
              file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path + "_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()
