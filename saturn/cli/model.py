import click
from comet.cli import login_axiom
import questionary
from comet.lib import print_utils, logger


@click.group()
def model():
    login_axiom()


@model.command()
@click.option('--model_path', '-p',
              required=False,
              help="path to folder",
              default=None)
@click.option('--name', '-n',
              required=False,
              help="The name of the model",
              default=None)
@click.option('--replace', '-rf',
              help="Replace existed data/model",
              is_flag=True)
def push(model_path, name, replace):
    from venus.wrapper import axiom_wrapper

    if not model_path:
        model_path = questionary.text("What is the path to model folder?").ask()

    if not name:
        print_utils.print_line(f"Specify the model name following: "
                               f"<ProjectName>-<PretrainedModel>-<ModelPurpose>-<Data Size>-<Version>")

        name = ""
        project_name = questionary.text("What's the project name: ?").ask()
        if project_name:
            name += project_name
        else:
            raise ValueError("Project name is required")

        pretrained_model = questionary.text("What's the pretrained model: ?").ask()
        if pretrained_model:
            name += f"-{pretrained_model}"
        else:
            raise ValueError("Pretrained model is required")

        model_purpose = questionary.text("What's the model purpose: (faq, RC, QA ...) default is: Faq ?").ask()
        if model_purpose:
            name += f"-{model_purpose}"
        else:
            name += "-faq"

        data_size = questionary.text("What's the data size: ?").ask()
        if data_size:
            name += f"-{data_size}"

        version = questionary.text("What's the version: ?").ask()
        if version:
            name += f"-{version}"
        else:
            raise ValueError("Version is required")
        # Replace _ with - in name and " " by "-"
        name = name.replace("_", "-").replace(" ", "-")

    axiom_wrapper.upload_model(model_name=name, dir_path=model_path, replace=replace)


@model.command()
def ls():
    from venus.wrapper import axiom_wrapper
    axiom_wrapper.ls_model()


@model.command()
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
