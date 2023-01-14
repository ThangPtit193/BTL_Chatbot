import os
from logging import Logger
from typing import *
from vaxiom_client.client import Vaxiom
from vaxiom_client import constants
from vaxiom_client.utils.file_utils import get_config, create_config
import questionary
_logger = Logger(__name__)

REPOSITORY_NAME = "saturn-model-hub"
REPOSITORY_VERSION = "v1.0.0"


class ModelHub:
    def __init__(self, base_url: Text = None, token: Text = None):

        if not base_url:
            base_url = os.environ.get('VAXIOM_ENDPOINT', constants.DEFAULT_VAXIOM_ENDPOINT)
        if not token:
            token = get_config(constants.VAXIOM_CONFIG_FILE, "token")
        self.client = Vaxiom(base_url=base_url, token=token)

    def upload_model(self, model_name: Text, model_path: Text):
        # Get all file in repository
        response = self.client.get_resource_items(REPOSITORY_NAME, REPOSITORY_VERSION)
        all_name_files = []
        if response[0]:
            all_paths = [get_absolute_path('', file['name']) for file in response[1]['file_children']]
            for folder in response[1]['folder_children']:
                all_paths.extend(get_path_recursive(folder, ''))
            all_paths.sort()
            # files = self.client.get_your_repositories()
            # self.client.upload_resource()
            all_name_files = [os.path.basename(path) for path in all_paths]


def get_path_recursive(folder, prefix):
    paths = [get_absolute_path(prefix, folder['name'])]
    prefix = get_absolute_path(prefix, folder['name'])
    paths.extend([get_absolute_path(prefix, file['name']) for file in folder['file_children']])
    for sub_folder in folder['folder_children']:
        paths.extend(get_path_recursive(sub_folder, prefix))
    return paths


def get_absolute_path(prefix, name):
    return prefix + '/' + name


if __name__ == "__main__":
    model_hub = ModelHub()
    model_hub.upload_model("test", "test")
