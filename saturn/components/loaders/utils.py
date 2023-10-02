from typing import List, Tuple

from transformers import PreTrainedTokenizer

from saturn.utils.normalize import (
    normalize_encode,
    normalize_word_diacritic,
)
from saturn.utils.utils import preprocessing


def convert_text_to_features(
        text: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 128,
        special_tokens_count: int = 2,
        **kwargs
) -> Tuple[List[int], List[int]]:
    unk_token = tokenizer.unk_token

    cls_token = tokenizer.cls_token

    sep_token = tokenizer.sep_token

    pad_token_id = tokenizer.pad_token_id

    # Normalize text
    text = normalize_encode(normalize_word_diacritic(preprocessing(text)))

    text = text.split()  # Some are spaced twice

    tokens = []
    # Tokenizer
    for word in text:
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [unk_token]  # For handling the bad-encoded word
        tokens.extend(word_tokens)

    # Truncate data
    if len(tokens) > max_seq_len - special_tokens_count:
        tokens = tokens[: (max_seq_len - special_tokens_count)]

    # Add [SEP] token
    tokens += [sep_token]

    # Add [CLS] token
    tokens = [cls_token] + tokens

    # Convert tokens to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    # TODO use smart padding in here
    # Zero-pad up to the sequence length. This is static method padding
    padding_length = max_seq_len - len(input_ids)
    input_ids = input_ids + ([pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)

    return input_ids, attention_mask
