import os

import click

from loguru import logger

from meteor.nodes.evaluator.benchmark import BenchMarker


def execute_cmd(cmd):
    os.system(cmd)


@click.group()
def benchmark():
    # print("Starting benchmark the dataset ...")
    pass


@benchmark.command()
@click.option('--config', '-c',
              required=True,
              default="config/config.yaml")
@click.option('--debug',
              is_flag=True,
              default=False)
def run(config, debug=None):
    # if debug:
    #     logger.__getattribute__("DEBUG")
    # else:
    #     logger.__getattribute__("INFO")

    if config is None:
        raise ValueError('Config required to get initialized eval parameters')
    eval_pipeline = BenchMarker.load_from_config(config_path=config)
    eval_pipeline.querying()
