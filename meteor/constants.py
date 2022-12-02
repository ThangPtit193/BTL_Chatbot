import inspect
import os
from environs import Env
from os.path import expanduser

env = Env()
env.read_env(f"{os.getcwd()}/.env")  # read .env file, if it exists

HOME = expanduser("~")
METEOR_DIR = os.path.join(HOME, ".meteor")
METEOR_DIR_CONFIGURE = os.path.join(METEOR_DIR, "config")
METEOR_DIR_DATASET = os.path.join(METEOR_DIR, "dataset")
METEOR_DIR_MODEL = os.path.join(METEOR_DIR, "models")

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
INDEX_RESULT_FILES = "reports/retriever.csv"

# Init environment
env = Env()
env.read_env(f"{WORKING_DIR}/.env")
