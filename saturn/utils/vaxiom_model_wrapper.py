import json
import tempfile
import os
import shutil
from logging import Logger
from typing import *
# from venus.utils.constants import VENUS_DIR, VENUS_DIR_MODEL
# from venus.utils.utils import load_json, write_json
from vaxiom_client.utils.file_utils import get_config, create_config, create_register_file
from vaxiom_client.client import Vaxiom
from vaxiom_client import constants
import os
from typing import TYPE_CHECKING

import click

from vaxiom_client.utils.file_system import local_file_system
from vaxiom_client.utils.exceptions import print_error
from vaxiom_client.utils.tree import Tree
from vaxiom_client import constants
from vaxiom_client.utils.file_utils import get_config
from vaxiom_client.utils.table_utils import create_simple_table, create_public_links_table
from vaxiom_client.utils.tree_display import TreeDisplay
from vaxiom_client.utils.client import get_versions, get_repository_info
from saturn.utils.helper import http_get

_logger = Logger(__name__)

REPOSITORY_NAME = "saturn-model-hub"
REPOSITORY_ID = 10
REPOSITORY_VERSION = "v1.0.0"


class ModelHub:
    def __init__(self, base_url: Text = None, token: Text = None):

        if not base_url:
            base_url = os.environ.get('VAXIOM_ENDPOINT', constants.DEFAULT_VAXIOM_ENDPOINT)
        if not token:
            token = get_config(constants.VAXIOM_CONFIG_FILE, "token")
        self.client = Vaxiom(base_url=base_url, token=token)

    def upload_model(self, model_name, model_path, replace=False):
        """
        Upload model to vaxiom model hub
        Args:
            model_name:
            model_path:
            replace:

        Returns:

        """
        if os.path.exists(model_path) is False:
            raise FileNotFoundError(f"Model path {model_path} is not existed")
        # Get all file in repository
        if not replace:
            all_models_info = self._get_models_info()
            model_info = [model for model in all_models_info if model['model_name'] == model_name]
            if model_info:
                raise ValueError(f"Model name {model_name} is existed in model hub")

        temp_dir_path = tempfile.mkdtemp()
        if os.path.isdir(model_path):
            model_path = shutil.make_archive(base_name=model_name,
                                             format='zip',
                                             root_dir=model_path)
            shutil.move(model_path, temp_dir_path)
        elif os.path.isfile(model_path):
            model_path = model_path
            shutil.copy2(model_path, temp_dir_path)
        # Upload model to axiom using vaxiom client
        os.system(
            f"vaxiom resource upload --repo {REPOSITORY_NAME} --version {REPOSITORY_VERSION} "
            f"--path {temp_dir_path} --force"
        )

    def download_model(self, model_name: Text, model_path: Text):
        # Get all file in repository
        all_models_info = self._get_models_info()

        model_info = [model for model in all_models_info if model['model_name'] == model_name]
        if not all_models_info:
            raise ValueError(f"Model name {model_name} is not existed in model hub")
        model_info = model_info[0]

        _logger.info(f"Downloading model '{model_name}' from {model_info['public_link']}")
        self._download_model_from_url(
            model_url=model_info['public_link'],
            model_dir=model_path,
            model_file=os.path.join(model_path, model_info['model_file'])
        )
        return model_path

    def _get_models_info(self):
        # Get all file in repository
        response = self.client.get_resource_items(repository_name=REPOSITORY_NAME, version=REPOSITORY_VERSION)
        if not response[0]:
            print_error(response[1])
            return

        # get path information and public links
        tree = Tree()
        tree.from_response(response[1])
        tree_dict = tree.as_dict()
        all_paths = list(tree_dict.keys())
        all_paths.sort()
        public_links = []
        for path in all_paths:
            public_link = tree_dict[path][0]['public_link']
            if public_link:
                public_links.append(public_link)

        # Get model info
        model_info = []
        for public_link in public_links:
            # Get the text between the last '/' and the last '?'
            model_name = public_link[public_link.rfind('/') + 1:public_link.rfind('?')]
            model_info.append({
                'model_name': model_name.rstrip(".zip"),
                'model_file': model_name,
                'public_link': public_link
            })

        return model_info

    @staticmethod
    def _download_model_from_url(model_url, model_dir, model_file):
        if not os.path.isfile(model_file):
            print(f"Download model from Venus hub: {model_url} to {model_dir}")
            _logger.info(f"Download model from Venus hub: {model_url} to {model_dir}")
            http_get(model_url, model_file)

        if not os.path.isdir(model_file):
            try:
                shutil.unpack_archive(model_file, model_dir)
                _logger.info(f"Extract file {model_file} and save in {model_dir}")

                if os.path.isfile(model_file):
                    os.system(f"rm -rf {model_file}")
            except:
                _logger.warning(f"File {model_file} is not compressed file.")

        if os.path.isdir(model_dir):
            file_names = os.listdir(model_dir)
            model_dir_name = model_dir.split("/")[-1]
            if model_dir_name in file_names:
                exact_model_dir = os.path.join(model_dir, model_dir_name)
                for f in os.listdir(exact_model_dir):
                    shutil.move(os.path.join(exact_model_dir, f), os.path.join(model_dir, f))

                shutil.rmtree(exact_model_dir)
            return model_dir
        else:
            raise Exception(f"Model not found in {model_dir}")


if __name__ == "__main__":
    model_hub = ModelHub()
    # model_hub.upload_model(
    #     model_name="timi-keepitreal-H768-faq-870k-v1.1.5",
    #     model_path="/Users/hao/.cache/comet/models/timi-keepitreal-H768-faq-870k-v1.1.5",
    # )
    model_hub.download_model(
        model_name="timi-keepitreal-H768-faq-870k-v1.1.5",
        model_path="models",
    )
