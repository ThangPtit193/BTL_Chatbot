import tempfile
import os
import shutil
from logging import Logger

from axiom_client.utils import print_utils

from meteor.constants import METEOR_DIR, METEOR_DIR_MODEL
from meteor.utils.io import load_json, write_json, http_get

_logger = Logger(__name__)

try:
    from axiom_client.client import Axiom
    from axiom_client.utils.file_utils import create_config, update_config
    from axiom_client import AXIOM_CONFIG_FILE
    from axiom_client.utils.table_utils import (
        create_list_item_table,
        create_resource_detail_table,
        create_resource_list_table,
    )
except Exception as err:
    _logger.warning(f"Cannot init axiom: {err}")


class AxiomWrapper:
    meteor_models_hub_path = f"{METEOR_DIR}/METEOR_MODELS_HUB"

    def __init__(
            self,
            axiom_email: str,
            axiom_password: str,
            model_ids: int,
            model_version: str,
            hub_id: int,
            hub_version: str
    ):
        self.axiom_email = axiom_email
        self.axiom_password = axiom_password
        self.model_ids = model_ids
        self.model_version = model_version
        self.hub_id = hub_id
        self.hub_version = hub_version
        self.client = Axiom(base_url="https://axiom.dev.ftech.ai")

        if self.axiom_email and self.axiom_password:
            try:
                self.client.login(email=self.axiom_email, password=axiom_password)
                _logger.info("Login axiom success")
            except Exception as e:
                _logger.warning(f"Cannot login axiom: {e}")

        if os.path.exists(self.meteor_models_hub_path) is False:
            os.makedirs(self.meteor_models_hub_path)

    def download_meteor_models_hub(self):
        result = self.client.resource_detail(id=self.hub_id)
        self.client.resource_download(
            result["name"],
            rtype="model",
            version=self.hub_version,
            dir_path=self.meteor_models_hub_path
        )

    def get_new_meteor_models_hub(self):
        self.download_meteor_models_hub()
        meteor_models_hub = load_json(os.path.join(self.meteor_models_hub_path, 'VENUS_MODELS_HUB.json'))
        return meteor_models_hub

    def ls_model(self):
        meteor_models_hub = self.get_new_meteor_models_hub()
        print(f"List model in meteor hub: ID:{self.hub_id} Version: {self.hub_version}")
        headers = ["model_name", "url"]
        data = []
        for model_name, model_info in meteor_models_hub.items():
            data.append([model_name, model_info["url"]])

        try:
            from columnar import columnar
            table = columnar(data, headers, no_borders=False)
            print(table)
        except:
            print(data)

    def upload_meteor_models_hub(self):
        dir_path = self.meteor_models_hub_path
        if not os.path.isdir(dir_path):
            _logger.error(f"Folder: {dir_path} not exist!!")
            raise FileNotFoundError(f"Folder: {dir_path} not exist!!")
        if not os.path.exists(f"{dir_path}/VENUS_MODELS_HUB.json"):
            raise FileNotFoundError(f"Folder: {dir_path} must have file VENUS_MODELS_HUB.json!!")

        _logger.info(f"Upload meteor_models_hub from folder: {dir_path}\n"
                     f"\tID: {id}\n"
                     f"\tStatus: public")

        # self.client.resource_delete(name=self.hub_id, rtype="model", version=self.hub_version)
        result = self.client.resource_detail(self.hub_id)

        if self.hub_version not in [v['name'] for v in result['versions']]:
            result = self.client.resource_new_version(self.hub_id, self.hub_version)

        self.client.resource_item_delete_by_key(
            resource=result["name"],
            version=self.hub_version,
            rtype="model",
            key=f"VENUS_MODELS_HUB.json"
        )

        self.client.resource_upload(
            result["name"], rtype="model", version=self.hub_version, dir_path=dir_path
        )

    def download_model_from_url(self, model_url, model_dir, model_file):
        if not os.path.isdir(model_dir) and not os.path.isfile(model_file):
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

    def fetch_model(self, model_name_or_path):
        if os.path.exists(model_name_or_path):
            return model_name_or_path
        if not os.path.exists(os.path.join(self.meteor_models_hub_path, 'VENUS_MODELS_HUB.json')):
            self.download_meteor_models_hub()

        meteor_models_hub = load_json(os.path.join(self.meteor_models_hub_path, 'VENUS_MODELS_HUB.json'))

        if model_name_or_path in meteor_models_hub:
            model_info = meteor_models_hub[model_name_or_path]
        else:
            meteor_models_hub = self.get_new_meteor_models_hub()
            if model_name_or_path not in meteor_models_hub:
                _logger.info(f"List model in hub: {list(meteor_models_hub.keys())}")
                _logger.exception(f"Model {model_name_or_path} is not found in meteor hub.\n")
                return model_name_or_path
            else:
                model_info = meteor_models_hub[model_name_or_path]

        base_dir = METEOR_DIR_MODEL
        model_dir = os.path.join(base_dir, model_info['folder_name'])
        model_file = os.path.join(base_dir, model_info['file_name'])
        model_url = model_info['url']
        self.download_model_from_url(model_url, model_dir, model_file)

        if os.path.exists(model_dir):
            return model_dir
        else:
            Exception("Fetch model error.")

    def download_model(self, model_name, dir_path, replace=False):
        if os.path.exists(dir_path) is False:
            os.makedirs(dir_path)
        if not os.path.exists(f"{self.meteor_models_hub_path}/VENUS_MODELS_HUB.json"):
            self.download_meteor_models_hub()

        meteor_models_hub = load_json(f"{self.meteor_models_hub_path}/VENUS_MODELS_HUB.json")

        if model_name in meteor_models_hub:
            model_info = meteor_models_hub[model_name]
        else:
            meteor_models_hub = self.get_new_meteor_models_hub()
            if model_name not in meteor_models_hub:
                model_info = None
            else:
                model_info = meteor_models_hub[model_name]

        if model_info is None:
            raise Exception(f"Model {model_name} is not found on meteor hub")

        # model_base = dir_path
        model_dir = os.path.join(dir_path, model_info['folder_name'])
        model_file = os.path.join(dir_path, model_info['file_name'])
        model_url = model_info['url']

        if os.path.exists(model_dir):
            if replace is False:
                raise FileExistsError(f"Model {model_name} is existed in {model_dir}")
            else:
                shutil.rmtree(model_dir)
        self.download_model_from_url(model_url, model_dir, model_file)

    def upload_model(self, model_name, dir_path, replace=False):
        meteor_models_hub = self.get_new_meteor_models_hub()
        if model_name in meteor_models_hub:
            if replace is False:
                raise Exception(f"Name {model_name} is existed. Please change model_name!")
            else:
                _logger.info(f"Upload model {model_name} to meteor hub and replace existed model")
        if os.path.exists(dir_path) is False:
            raise FileNotFoundError(f"Model path {dir_path} is not existed")

        temp_dir_path = tempfile.mkdtemp()
        if os.path.isdir(dir_path):
            model_path = shutil.make_archive(base_name=model_name,
                                             format='zip',
                                             root_dir=dir_path)
            shutil.move(model_path, temp_dir_path)
        elif os.path.isfile(dir_path):
            model_path = dir_path
            shutil.copy2(model_path, temp_dir_path)

        result = self.client.resource_detail(self.model_ids)
        if self.model_version not in [v['name'] for v in result['versions']]:
            result = self.client.resource_new_version(self.model_ids, self.model_version)

        items = self.client.resource_list_item(name=result["name"], rtype="model", version=self.model_version)
        model_in_hub_names = [item["key"] for item in items]
        if f"{model_name}.zip" in model_in_hub_names:
            if replace:
                self.client.resource_item_delete_by_key(
                    resource=result["name"],
                    version=self.model_version,
                    rtype="model",
                    key=f"{model_name}.zip"
                )
            else:
                raise Exception(f"Name {model_name} is existed. Please change model_name!")

        _logger.info("Model Uploading...")

        self.client.resource_upload(
            result["name"], rtype="model", version=self.model_version, dir_path=temp_dir_path
        )
        _logger.info(f"Upload model {model_name} Done")

        connection = None
        for v in result["versions"]:
            if v["name"] == self.model_version:
                connection = v["connection"]

        if connection is None:
            raise Exception(f"Model ID: {self.model_ids} Version {self.model_version} is not exist!")

        model_url = f"http://minio.dev.ftech.ai/{connection}/{model_name}.zip"
        meteor_models_hub[model_name] = {
            "url": model_url,
            "file_name": f"{model_name}.zip",
            "folder_name": model_name
        }

        write_json(meteor_models_hub, file_path=f"{self.meteor_models_hub_path}/VENUS_MODELS_HUB.json")
        self.upload_meteor_models_hub()

        # shutil.rmtree(temp_dir_path)


if __name__ == "__main__":
    AXIOM_EMAIL = os.getenv('AXIOM_EMAIL', 'phongnt@ftech.ai')
    AXIOM_PASSWORD = os.getenv('AXIOM_PASSWORD', 'b8dJfQFq6DL3')
    HUB_ID = os.getenv('HUB_ID', 369)
    HUB_VERSION = os.getenv('HUB_VERSION', 'v0.1.0')
    MODELS_ID = os.getenv('MODELS_ID', 369)
    MODELS_VERSION = os.getenv('MODELS_VERSION', 'v8.0.0')

    axiom_wrapper = AxiomWrapper(
        axiom_email=AXIOM_EMAIL,
        axiom_password=AXIOM_PASSWORD,
        model_ids=MODELS_ID,
        model_version=MODELS_VERSION,
        hub_id=HUB_ID,
        hub_version=HUB_VERSION
    )
    print(axiom_wrapper.ls_model())
#
#
#     # print(axiom_wrapper.ls_model())
#
#     def ls(id, version):
#         print_utils.print_title("DATASET INFORMATION")
#         os.system(f"axiom dataset ls --id {id} --version {version}")
#
#
#     print(ls(174, "v0.3"))
