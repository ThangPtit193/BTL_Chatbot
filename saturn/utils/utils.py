import logging
import os
import random
import re
import string
import numpy as np
import torch
from transformers import set_seed
from transformers import (
    AutoTokenizer,
    RobertaConfig
)
from saturn.components.models.model import BiencoderRobertaModel

MODEL_CLASSES = {
    "phobert-base": (RobertaConfig, BiencoderRobertaModel, AutoTokenizer),
    "phobert-base-v2": (RobertaConfig, BiencoderRobertaModel, AutoTokenizer),
    "phobert-large": (RobertaConfig, BiencoderRobertaModel, AutoTokenizer),
    "sim-cse-vietnamese": (RobertaConfig, BiencoderRobertaModel, AutoTokenizer),
    "unsim-cse-vietnamese": (RobertaConfig, BiencoderRobertaModel, AutoTokenizer),
}

MODEL_PATH_MAP = {
    "phobert-base": "vinai/phobert-base",
    "phobert-base-v2": "/home/vth/backbone/models--vinai--phobert-base-v2/snapshots/5388b8ddc52de647dc81e81bbe174fa1fc37e12c",
    "phobert-large": "/home/vth/backbone/models--vinai--phobert-large/snapshots/9ce4eafcd8e601d798295b17c75c5f5f1b1509b9",
    "sim-cse-vietnamese": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
    "unsim-cse-vietnamese": "VoVanPhuc/unsup-SimCSE-VietNamese-phobert-base"
}

vowel = [
    ["a", "à", "á", "ả", "ã", "ạ", "a"],
    ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "aw"],
    ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "aa"],
    ["e", "è", "é", "ẻ", "ẽ", "ẹ", "e"],
    ["ê", "ề", "ế", "ể", "ễ", "ệ", "ee"],
    ["i", "ì", "í", "ỉ", "ĩ", "ị", "i"],
    ["o", "ò", "ó", "ỏ", "õ", "ọ", "o"],
    ["ô", "ồ", "ố", "ổ", "ỗ", "ộ", "oo"],
    ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ", "ow"],
    ["u", "ù", "ú", "ủ", "ũ", "ụ", "u"],
    ["ư", "ừ", "ứ", "ử", "ữ", "ự", "uw"],
    ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ", "y"],
]

vowel_to_idx = {}
for i in range(len(vowel)):
    for j in range(len(vowel[i]) - 1):
        vowel_to_idx[vowel[i][j]] = (i, j)


def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    data_dir = './logs/'
    os.makedirs(data_dir, exist_ok=True)
    file_handler = logging.FileHandler('{}/log.txt'.format(data_dir))
    file_handler.setFormatter(log_format)

    logger.handlers = [console_handler, file_handler]

    return logger


logger = _setup_logger()


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def is_valid_vietnam_word(word):
    chars = list(word)
    vowel_index = -1
    for index, char in enumerate(chars):
        x, _ = vowel_to_idx.get(char, (-1, -1))
        if x != -1:
            if vowel_index == -1:
                vowel_index = index
            else:
                if index - vowel_index != 1:
                    return False
                vowel_index = index
    return True


def preprocessing(text):
    punctuation = string.punctuation.replace('/', '').replace('-', '')

    text = text.replace('/', '-')
    # remove punctuation
    for punc in punctuation:
        text = text.replace(punc, ' ')
    text = " ".join(text.strip().split()).lower()

    # remove duplicate space
    text = re.sub(r"[\s]+", " ", text)
    text = text.strip("\n ")

    # convert time
    text = re.sub(r"năm (\d+) đến năm (\d+)", r'\1-\2', text)
    text = re.sub(r"(\d+) tháng (\d+) năm (\d+)", r'\1-\2-\3', text)
    text = text.replace(' - ', '-')

    text = " ".join(text.strip().split())

    return text


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(
        args.model_name_or_path,
        use_fast=args.use_fast_tokenizer
    )
