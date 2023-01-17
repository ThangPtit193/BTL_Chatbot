import click

from saturn.cli.utils import create_list_item_table
import questionary
from comet.lib import print_utils, logger
from axiom_client import AXIOM_CONFIG_FILE
from axiom_client.client import Axiom
from axiom_client.utils.file_utils import get_config, create_config
import os
from comet.constants import env

_logger = logger.get_logger(__name__)

AXIOM_EMAIL = env.str('AXIOM_EMAIL', '')
AXIOM_PASSWORD = env.str('AXIOM_PASSWORD', '')

MODELS_ID = os.getenv('MODELS_ID', 187)
MODELS_VERSION = os.getenv('MODELS_VERSION', 'v0.1')


@click.group(invoke_without_command=True)
@click.option('--base-url', type=str, required=False, default="", help="Axiom endpoint")
@click.option("--token", type=str, required=False, help="User token")
@click.pass_context
def model(ctx, base_url, token):
    create_config()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()
    if not base_url:
        base_url = os.environ.get('AXIOM_ENDPOINT', 'https://axiom.dev.ftech.ai')
    if not token:
        token = get_config(AXIOM_CONFIG_FILE, "token")
    ctx.obj = {}
    ctx.obj["client"] = Axiom(base_url=base_url, token=token)

    # login_axiom()
    if AXIOM_EMAIL and AXIOM_PASSWORD:
        try:
            ctx.obj["client"].login(email=AXIOM_EMAIL, password=AXIOM_PASSWORD)
            _logger.info(f"Login axiom successfully")
        except Exception as e:
            _logger.warning(f"cannot login axiom: {e}")


@model.command()
@click.pass_context
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
@click.pass_context
def ls(ctx):
    client = ctx.obj["client"]

    # table item
    result = client.resource_detail(MODELS_ID)
    page = 1
    is_continue = True
    item_list = []
    while is_continue:
        data = client.resource_list_item(result["name"], "model", MODELS_VERSION, page=page)
        item_list.extend(data["results"])
        next_url = data["next"]
        if not next_url:
            break
        page += 1
    item_list = [item for item in item_list if item["key"].endswith(".zip")]
    # Sort by created_at
    item_list = sorted(item_list, key=lambda x: x["created_at"], reverse=True)
    for item in item_list:
        # Ignore the item that is not model
        item_id = item["id"]
        # item_detail = client.resource_item_detail(item_id)
        # result = client.resource_item_get_url(
        #     item_detail["resource_name"],
        #     "model",
        #     item_detail["version_name"],
        #     item_detail["key"],
        # )
        item['url'] = os.path.join(
            "http://minio.dev.ftech.ai/venus-model-v0.1-ca24fe0d",
            item["key"]
        )
        # Get key name
        item['key'] = item["key"].strip(".zip")
    print(create_list_item_table(item_list))
    return item_list

    #
    # # Get all item urls
    # links = []
    # resource_detail = client.resource_detail(MODELS_ID)
    # items = client.resource_list_item(resource_detail["name"], "model", MODELS_VERSION, page=1)
    # if items is not None:
    #     for item in items['results']:
    #         item_id = item['id']
    #         item_detail = client.resource_item_detail(item_id)
    #         result = client.resource_item_get_url(
    #             item_detail["resource_name"],
    #             "model",
    #             item_detail["version_name"],
    #             item_detail["key"],
    #         )
    #         comet_url = urlsplit(result)._replace(query=None).geturl()
    #         item['url'] = comet_url
    #         links.append(comet_url)
    #     print_utils.print_title("All data links")
    #     for link in links:
    #         print(link)
    # else:
    #     print(f"items is None")


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
