import os

import click
import questionary

from comet.lib import logger, print_utils
from termcolor import colored

from saturn.cli.model import model
from saturn.data_generation.tripple_generator import TripleGenerator
from saturn.kr_manager import KRManager
from saturn.utils.config_parser import ConfigParser
from .version import get_saturn_version
from comet.shared.vaxiom_model_wrapper import ModelHub

logger.configure_logger("DEBUG")
_logger = logger.get_logger(__name__)


@click.group()
def entry_point():
    pass


@click.command()
def version():
    ver = get_saturn_version()
    print(f"Saturn version: {ver}")


@click.command()
@click.option('--config', '-c', required=True, default="config/config.yaml")
def run_e2e(config):
    config_parser: ConfigParser = ConfigParser(config)
    kr_manager = KRManager(config=config_parser)
    general_config = config_parser.general_config()
    if not general_config.get('skipped_gen_data'):
        print_utils.print_line(f"Starting generate triples")
        triple_generator = TripleGenerator(config=config_parser)
        triple_generator.load()
        triple_generator.generate_triples()
        print_utils.print_line(f"✅ ✅ ✅ DONE generate triples ✅ ✅ ✅")
    else:
        print(colored('Data Generation is skipped', 'red'))

    # Train the model
    if not general_config.get('skipped_training'):
        print_utils.print_line(f"Starting train the model")
        kr_manager.train_embedder()
        print_utils.print_line(f"✅ ✅ ✅  DONE train the model ✅ ✅ ✅")
    else:
        print(colored('Training model is skipped', 'red'))

    # Evaluate the model
    if not general_config.get('skipped_eval'):
        print_utils.print_line(f"Starting evaluate the model")
        kr_manager.evaluate()
        print_utils.print_line(f"✅ ✅ ✅ DONE evaluate the model ✅ ✅ ✅")
    else:
        print(colored('Evaluating model is skipped', 'red'))


@click.command()
@click.option('--config', '-c', required=True, default="config/config.yaml")
def gen_data(config):
    config_parser: ConfigParser = ConfigParser(config)
    print_utils.print_line(f"Starting generate triples")
    triple_generator = TripleGenerator(config=config_parser)
    triple_generator.load()
    triple_generator.generate_triples()
    print_utils.print_line(f"✅ ✅ ✅ DONE generate triples ✅ ✅ ✅")


@click.command()
@click.option('--config', '-c', required=True, default="config/config.yaml")
def release(config):
    config_parser: ConfigParser = ConfigParser(config)
    release_config = config_parser.release_config()
    if not release_config:
        raise ValueError("Release config is not found")

    model_path = release_config["model_path"]
    model_name = release_config.get("model_name")
    assert model_path, "Model path is not found"
    assert model_name, "Model name is not defined"
    is_agree_upload = questionary.confirm(
        f"Are you sure to upload the model from '{model_path}' with name: '{model_name}' to model hub?"
    ).ask()
    if is_agree_upload:
        # axiom_wrapper.upload_model(model_name=model_name, model_path=model_path, replace=True)
        ModelHub().upload_model(model_name=model_name, model_path=model_path, replace=True)
    else:
        print("Aborting...")
        return


@click.command()
@click.option('--config', '-c', required=True, default="config/config.yaml")
def train(config):
    config_parser: ConfigParser = ConfigParser(config)
    kr_manager = KRManager(config=config_parser)
    kr_manager.train_embedder()


@click.command()
@click.option('--config', '-c',
              required=True,
              default="config/config.yaml")
@click.option('--rtype', '-rt',
              required=False,
              help="Supported report types are `detail, overall, all` with default value is all",
              default="all")
@click.option('--top_k', '-k',
              required=False,
              help="Top_k for limiting the retrieval report",
              type=int,
              default=None)
@click.option('--save_md', '-md',
              required=False,
              type=bool,
              help="Save report with markdown file",
              default=True)
def test(config, rtype, top_k, save_md):
    config_parser: ConfigParser = ConfigParser(config)
    kr_manager = KRManager(config=config_parser)
    kr_manager.evaluate()


@click.command()
@click.option('--path', '-p',
              required=False,
              help="Path to run streamlit",
              default=None)
def ui(path):
    if path is not None:
        os.system(f'streamlit run {path}')
    os.system('streamlit run ui/navigation.py')


entry_point.add_command(version)
entry_point.add_command(run_e2e)
entry_point.add_command(gen_data)
entry_point.add_command(train)
entry_point.add_command(test)
entry_point.add_command(release)
entry_point.add_command(ui)
entry_point.add_command(model)

if __name__ == '__main__':
    entry_point()
