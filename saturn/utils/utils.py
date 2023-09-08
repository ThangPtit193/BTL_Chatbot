import os
import re
import torch
import random
import string
import logging
import numpy as np
from bs4 import BeautifulSoup
from pyvi import ViTokenizer
from transformers import set_seed
from transformers import AutoTokenizer, RobertaConfig
from saturn.utils.io import load_json
from saturn.components.models.model import BiencoderRobertaModel

MODEL_CLASSES = {
    "unsim-cse-vietnamese": (RobertaConfig, BiencoderRobertaModel, AutoTokenizer),
    "sim-cse-vietnamese": (RobertaConfig, BiencoderRobertaModel, AutoTokenizer),
    "phobert-base-v2": (RobertaConfig, BiencoderRobertaModel, AutoTokenizer),
}

MODEL_PATH_MAP = {
    "unsim-cse-vietnamese": "VoVanPhuc/unsup-SimCSE-VietNamese-phobert-base",
    "sim-cse-vietnamese": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
    "phobert-base-v2": "vinai/phobert-base-v2"
}

metadata = load_json(file_path="/home/black/saturn/data/benchmark/metadata.json")
trans = {}

for data in metadata['history_metadatas']:
    for word in metadata['history_metadatas'][data]:
        trans[word.lower()] = data.lower()

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


def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator=' ')
    return text


def is_word(text):
    text = list(text)
    upper, punc = 0, 0
    for c in text:
        if c in string.punctuation:
            punc += 1
            continue
        if c.lower() in vowel_to_idx or c.lower() in string.ascii_letters:
            if 'A' <= c <= 'Z':
                upper += 1
            continue
        return False

    if upper <= 1 and punc <= 1:
        return True

    return False


def is_digits_punc(text):
    text = list(text)
    for i in range(len(text) - 1):
        if text[i] in string.punctuation and text[i + 1] in string.punctuation:
            return False
    punc, digits = 0, 0
    for c in text:
        if c in "/-.":
            punc += 1
        elif c in string.digits:
            digits += 1

    if punc + digits == len(text):
        return True

    return False


def convert_metadata(text):
    text = text.replace('_', ' ')
    text = " ".join(text.split()).strip()

    for key in trans:
        text = text.replace(key, trans[key])

    text = ViTokenizer.tokenize(text)

    return " ".join(text.split()).strip()


def preprocessing(text):
    text = text.replace('www', '').lower()
    # text = convert_metadata(text)
    text = text.replace(' - ', '-').replace(' / ', '/').replace('/', '-')
    punctuation = "[!\"#$%&'()*+,.:;<=>?@[\\]^`{|}~]“”"

    text = text.translate(str.maketrans(' ', ' ', punctuation))
    text = text.lower()

    # remove duplicate space
    text = re.sub(r"[\s]+", " ", text)

    return text
    # remove html tags
    # text = remove_html_tags(text)
    # text = re.sub(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', '', text, flags=re.IGNORECASE)
    #
    # text = re.sub('<[^<]+?>', ' ', text)
    # text = re.sub(r'http\S+|https\S+', ' ', text, flags=re.IGNORECASE)
    # text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', ' ', text, flags=re.IGNORECASE)
    #
    # # convert time
    # text = re.sub(r"năm (\d+) đến năm (\d+)", r'\1-\2', text)
    # text = re.sub(r"(\d+) tháng (\d+) năm (\d+)", r'\1-\2-\3', text)
    #
    # text = " ".join(text.strip().split()).strip().strip('\n').strip(punctuation)

    # return text


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(
        args.model_name_or_path,
        use_fast=args.use_fast_tokenizer
    )
