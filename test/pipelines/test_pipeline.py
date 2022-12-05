import logging

import pytest

from saturn.nodes.base import BaseComponent
from saturn.pipelines.base import Pipeline
from test.conftest import SAMPLES_PATH

logger = logging.getLogger(__name__)


class ParentComponent(BaseComponent):
    outgoing_edges = 1

    def __init__(self, dependent: BaseComponent) -> None:
        super().__init__()

    def run(*args, **kwargs):
        logging.info("ParentComponent run() was called")

    def run_batch(*args, **kwargs):
        pass


# @pytest.mark.elasticsearch
def test_to_code_creates_same_pipelines():
    query_pipeline = Pipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test.saturn-pipeline.yml", pipeline_name="indexing_pipeline"
    )
    query_pipeline_code = query_pipeline.to_code(pipeline_variable_name="indexing_pipeline_from_code")

    exec(query_pipeline_code)
    assert locals()["indexing_pipeline_from_code"] is not None
    assert query_pipeline.get_config() == locals()["indexing_pipeline_from_code"].get_config()
