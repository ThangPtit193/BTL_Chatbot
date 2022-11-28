import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def read_config_from_yaml(path: Path) -> Dict[str, Any]:
    """
    Parses YAML files into Python objects.
    Fails if the file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)
