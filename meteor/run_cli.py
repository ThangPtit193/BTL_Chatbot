import click
from meteor.kr_manager import KRManager
from comet.lib import logger

logger.configure_logger("DEBUG")


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


@click.command()
@click.option('--model_path', '-p',
              required=True,
              help="path to folder",
              default=None)
@click.option('--name', '-n',
              required=True,
              help="The name of the model",
              default=None)
@click.option('--replace', '-rf',
              help="Replace existed data/model",
              is_flag=True)
def push(model_path, name, replace):
    from venus.wrapper import axiom_wrapper
    axiom_wrapper.upload_model(model_name=name, dir_path=model_path, replace=replace)


@click.command()
def ls():
    from venus.wrapper import axiom_wrapper
    axiom_wrapper.ls_model()


@click.command()
@click.option('--name', '-n',
              required=True,
              help="The name of the model to pull",
              default=None)
@click.option('--output_path', '-dp',
              required=True,
              help="Where to save the model",
              default="models")
@click.option('--replace', '-rf',
              help="Replace existed data/model",
              is_flag=True
              )
def pull(name, dir_path, replace):
    from venus.wrapper import axiom_wrapper
    axiom_wrapper.download_model(model_name=name, dir_path=dir_path, replace=replace)


entry_point.add_command(version)
entry_point.add_command(train)
entry_point.add_command(test)
entry_point.add_command(push)
entry_point.add_command(pull)
entry_point.add_command(ls)

if __name__ == '__main__':
    entry_point()
