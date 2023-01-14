from typing import Text
from comet.lib import file_util


class ConfigParser:
    def __init__(self, config_path: Text):
        self.config_data = file_util.load_yaml(config_path)

    def trainer_config(self):
        return self.config_data['TRAINER'] if 'TRAINER' in self.config_data else {}

    def data_generation_config(self):
        return self.config_data['DATA_GENERATION'] if 'DATA_GENERATION' in self.config_data else {}

    def general_config(self):
        return self.config_data['GENERAL'] if 'GENERAL' in self.config_data else {}

    def eval_config(self):
        return self.config_data['EVALUATION'] if 'EVALUATION' in self.config_data else {}

    def release_config(self):
        return self.config_data['RELEASE'] if 'RELEASE' in self.config_data else {}
