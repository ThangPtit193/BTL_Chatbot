from saturn.utils.config_parser import ConfigParser
from typing import Text, Union
import os


class SaturnAbstract:
    device = None

    def __init__(self, config: Union[Text, ConfigParser], **kwargs):
        if isinstance(config, Text):
            self.config_parser = ConfigParser(config)
        else:
            self.config_parser = config
        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
        if self.device is not None:
            import torch
            self.device = "cpu" if not torch.cuda.is_available() else self.device

    def get_model_dir(self):
        return os.path.join(
            self.config_parser.general_config()['output_model'],
            self.config_parser.general_config()['project'],
            self.config_parser.general_config()['version']
        )

    def get_data_dir(self):
        return os.path.join(
            self.config_parser.general_config()['output_data'],
            self.config_parser.general_config()['project'],
            self.config_parser.general_config()['version']
        )

    def get_report_dir(self):
        return os.path.join(
            self.config_parser.general_config()['output_report'],
            self.config_parser.general_config()['project'],
            self.config_parser.general_config()['version']
        )

    def get_checkpoint_dir(self):
        return os.path.join(
            self.config_parser.general_config()['output_model'],
            self.config_parser.general_config()['project'],
            self.config_parser.general_config()['version'],
            "checkpoints"
        )
