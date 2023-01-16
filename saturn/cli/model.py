import click

from comet.lib import logger
from comet.shared.model_hub import ModelHub

_logger = logger.get_logger(__name__)


@click.group()
def model():
    pass


@model.command()
def ls():
    ModelHub().list_model()
