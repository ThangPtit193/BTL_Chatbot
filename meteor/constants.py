import os
from environs import Env
from os.path import expanduser

env = Env()
env.read_env(f"{os.getcwd()}/.env")  # read .env file, if it exists

HOME = expanduser("~")
METEOR_DIR = os.path.join(HOME, ".meteor")
METEOR_DIR_CONFIGURE = os.path.join(METEOR_DIR, "configure")
METEOR_DIR_DATASET = os.path.join(METEOR_DIR, "dataset")
METEOR_DIR_MODEL = os.path.join(METEOR_DIR, "models")