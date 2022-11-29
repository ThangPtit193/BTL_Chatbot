import click
from meteor.kr_manager import KRManager


@click.group()
def entry_point():
    pass


@click.command()
def version():
    print(f"Meteor version: {click.__version__}")


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

if __name__ == '__main__':
    entry_point()
