import os
from typing import Text, Union

from saturn.utils.config_parser import ConfigParser
import torch


class SaturnAbstract:
    device = None
    ready = False
    skipped = False
    skipped_gen_data = False
    skipped_training = False
    skipped_eval = False
    is_warning_action = False

    def __init__(self, config: Union[Text, ConfigParser], **kwargs):
        if isinstance(config, Text):
            self.config_parser = ConfigParser(config)
        else:
            self.config_parser = config
        self.device = self.config_parser.general_config().get('device', None)

        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
        for k, v in self.config_parser.general_config().items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.device is not None:
            self.device = "cpu" if not torch.cuda.is_available() else self.device
        if self.device is None:
            self.device = "cpu" if not torch.cuda.is_available() else "cuda"

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
