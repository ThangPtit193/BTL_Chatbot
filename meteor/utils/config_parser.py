from comet.lib import file_util
from typing import Text


class ConfigParser:
    def __init__(self, config_path: Text):
        self.config_data = file_util.load_yaml(config_path)

    def general_config(self):
        return self.config_data['GENERAL']

    def embedder_config(self):
        return self.config_data['EMBEDDER']

    def eval_config(self):
        return self.config_data['EVALUATION'] if 'EVALUATION' in self.config_data else {}

    def trainer_config(self):
        return self.config_data['TRAINER'] if 'TRAINER' in self.config_data else {}
