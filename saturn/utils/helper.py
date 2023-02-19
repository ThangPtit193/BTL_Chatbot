import os
import sys

import requests
from axiom_client import AXIOM_CONFIG_FILE
from axiom_client.client import Axiom
from axiom_client.utils.file_utils import get_config
from tqdm import tqdm

from comet.constants import env
from comet.lib import logger

_logger = logger.get_logger(__name__)
AXIOM_EMAIL = env.str('AXIOM_EMAIL', '')
AXIOM_PASSWORD = env.str('AXIOM_PASSWORD', '')

MODELS_ID = os.getenv('MODELS_ID', 187)
MODELS_VERSION = os.getenv('MODELS_VERSION', 'v0.1')


def get_model_from_axiom():
    base_url = os.environ.get('AXIOM_ENDPOINT', 'https://axiom.dev.ftech.ai')
    token = get_config(AXIOM_CONFIG_FILE, "token")
    client = Axiom(base_url=base_url, token=token)
    # table item
    result = client.resource_detail(MODELS_ID)
    page = 1
    is_continue = True
    item_list = []
    while is_continue:
        data = client.resource_list_item(result["name"], "model", MODELS_VERSION, page=page)
        item_list.extend(data["results"])
        next_url = data["next"]
        if not next_url:
            break
        page += 1
    item_list = [item['key'].strip('.zip') for item in item_list if item["key"].endswith(".zip")]

    return item_list


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
