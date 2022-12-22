import os

from comet.lib import logger
from axiom_client import AXIOM_CONFIG_FILE
from axiom_client.client import Axiom
from axiom_client.utils.file_utils import get_config
from comet.constants import env

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
