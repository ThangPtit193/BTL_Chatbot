import os

import click
from saturn.kr_manager import KRManager
from comet.lib import logger
from saturn.cli.model import model
from .version import get_saturn_version

logger.configure_logger("DEBUG")


@click.group()
def entry_point():
    pass


@click.command()
def version():
    ver = get_saturn_version()
    print(f"Saturn version: {ver}")


@click.command()
@click.option('--config', '-c', required=True, default="config/config.yaml")
def train(config):
    kr_manager = KRManager(config_path=config)
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
              default=False)
def test(config, rtype, top_k, save_md):
    kr_manager = KRManager(config_path=config)
    kr_manager.save(report_type=rtype, top_k=top_k, save_markdown=save_md)


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
entry_point.add_command(train)
entry_point.add_command(test)
entry_point.add_command(ui)
entry_point.add_command(model)

if __name__ == '__main__':
    entry_point()
