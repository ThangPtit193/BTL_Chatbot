import click
from saturn.kr_manager import KRManager
from comet.lib import logger
from saturn.cli.model import model

logger.configure_logger("DEBUG")


@click.group()
def entry_point():
    pass


@click.command()
def version():
    print(f"Saturn version: {click.__version__}")


@click.command()
@click.option('--config', '-c', required=True, default="config/config.yaml")
def train(config):
    kr_manager = KRManager(config_path=config)
    kr_manager.train_embedder()


@click.command()
@click.option('--config', '-c', required=True, default="config/config.yaml")
def test(config):
    kr_manager = KRManager(config_path=config)
    kr_manager.evaluate_embedder()


entry_point.add_command(version)
entry_point.add_command(train)
entry_point.add_command(test)
entry_point.add_command(model)
if __name__ == '__main__':
    entry_point()
