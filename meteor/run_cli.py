import click
from meteor.cli.eval import benchmark


@click.group()
def entry_point():
    pass


@click.command()
def version():
    print(f"Meteor version: {click.__version__}")


entry_point.add_command(benchmark)
