from typing import Dict, List, Optional, Any

import copy
import json
import inspect
import logging
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
from pandas.core.frame import DataFrame
import yaml
from networkx import DiGraph
from networkx.drawing.nx_agraph import to_agraph
from venus.pipelines.pipeline import BaseStandardPipeline
from venus.retriever.base import Retriever

from pipelines.pipelines.config import (
    get_component_definitions,
    get_pipeline_definition,
    read_pipeline_config_from_yaml,
)


class BaseServicePipeline(BaseStandardPipeline):
    def __init__(
            self,
            retriever: Retriever
    ):
        pass
