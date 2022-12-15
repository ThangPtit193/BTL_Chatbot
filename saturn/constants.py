import inspect
import os
from environs import Env
from os.path import expanduser

env = Env()
env.read_env(f"{os.getcwd()}/.env")  # read .env file, if it exists

HOME = expanduser("~")
SATURN_DIR = os.path.join(HOME, ".saturn")
SATURN_DIR_CONFIGURE = os.path.join(SATURN_DIR, "config")
SATURN_DIR_DATASET = os.path.join(SATURN_DIR, "dataset")
SATURN_DIR_MODEL = os.path.join(SATURN_DIR, "models")

DEFAULT_API_SECRET_TOKEN = "t17ySBMD2XeX0cG0yypF/XlsH7dIGiThBsz/f4pr5Sc="


def get_cfd(backward=0):
    """
    Get current file directory
    :param backward:
    :type backward:
    :return:
    :rtype:
    """

    stack = inspect.stack()
    abspath = stack[1][1]
    abspath = os.path.dirname(abspath)
    for i in range(backward):
        abspath = os.path.dirname(abspath)
    abspath = os.path.join(os.getcwd(), abspath)
    return abspath


WORKING_DIR = os.getcwd()
SOURCE_DIR = get_cfd(1)
EVAL_INDEX = "eval_index"

# Init environment
env = Env()
env.read_env(f"{WORKING_DIR}/.env")
