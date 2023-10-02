import re
import torch
import string
import numpy as np
from pyvi import ViTokenizer
from transformers import AutoTokenizer, RobertaConfig
from saturn.utils.io import load_json
from saturn.components.models.model import BiencoderRobertaModel

MODEL_CLASSES = {
    "phobert-base-v2": (RobertaConfig, BiencoderRobertaModel, AutoTokenizer),
}

MODEL_PATH_MAP = {
    "phobert-base-v2": "vinai/phobert-base-v2"
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
    punctuation = "[!\"#$%&'()*+,.:;<=>?@[\\]^`{|}~]“”"
    text = text.translate(str.maketrans(' ', ' ', punctuation))
    text = text.lower()
    
    # remove duplicate space
    text = re.sub(r"[\s]+", " ", text)

    return text

def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(
        args.model_name_or_path,
    )
