import copy
import json
import os
from glob import glob

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from saturn.utils.normalize import normalize_encode, normalize_word_diacritic
from saturn.utils.utils import logger

from saturn.utils.io import load_jsonl


class InputExample(object):
    """
    A single training/test example for simple sequence.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
    """

    def __init__(self, guid, query, document, response=None):
        self.guid = guid
        self.query = query
        self.document = document
        self.response = response

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids_query,
        attention_mask_query,
        token_type_ids_query,
        input_ids_document,
        attention_mask_document,
        token_type_ids_document,
        input_ids_response,
        attention_mask_response,
        token_type_ids_response,
    ):
        self.input_ids_query = input_ids_query
        self.attention_mask_query = attention_mask_query
        self.token_type_ids_query = token_type_ids_query
        self.input_ids_document = input_ids_document
        self.attention_mask_document = attention_mask_document
        self.token_type_ids_document = token_type_ids_document
        self.input_ids_response = input_ids_response
        self.attention_mask_response = attention_mask_response
        self.token_type_ids_response = token_type_ids_response

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


# Prepare offline dataset for training
## build postive pairs from a sigle document
def create_data_to_features(
    args,
    examples,
    tokenizer,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    bos_token = tokenizer.bos_token
    bos_token_id = tokenizer.bos_token_id

    eos_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id

    unk_token = tokenizer.unk_token
    unk_token_id = tokenizer.unk_token_id

    pad_token = tokenizer.pad_token
    pad_token_id = tokenizer.pad_token_id

    cls_token = tokenizer.cls_token
    cls_token_id = tokenizer.cls_token_id

    sep_token = tokenizer.sep_token
    sep_token_id = tokenizer.sep_token_id

    # Account for [CLS] and [SEP]
    special_tokens_count = 2

    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        query_tokens = []
        for word in example.query:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            query_tokens.extend(word_tokens)

        # Inverse Cloze Task, Body First Selection, Wiki Link Prediction (asymetric-retrieval-downstream)
        document_tokens = []
        for word in example.document:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            document_tokens.extend(word_tokens)

        # Symetric-Similarity
        response_tokens = []
        for word in example.response:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            response_tokens.extend(word_tokens)

        # Truncate data
        if len(query_tokens) > args.max_seq_len_query - special_tokens_count:
            query_tokens = query_tokens[
                : (args.max_seq_len_query - special_tokens_count)
            ]
        if len(document_tokens) > args.max_seq_len_document - special_tokens_count:
            document_tokens = document_tokens[
                : (args.max_seq_len_document - special_tokens_count)
            ]
        if len(response_tokens) > args.max_seq_len_response - special_tokens_count:
            response_tokens = response_tokens[
                : (args.max_seq_len_response - special_tokens_count)
            ]

        # Add [SEP] token
        query_tokens += [sep_token]
        token_type_ids_query = [sequence_a_segment_id] * len(query_tokens)
        document_tokens += [sep_token]
        token_type_ids_document = [sequence_a_segment_id] * len(document_tokens)
        response_tokens += [sep_token]
        token_type_ids_response = [sequence_a_segment_id] * len(response_tokens)

        # Add [CLS] token
        query_tokens = [cls_token] + query_tokens
        token_type_ids_query = [cls_token_segment_id] + token_type_ids_query
        document_tokens = [cls_token] + document_tokens
        token_type_ids_document = [cls_token_segment_id] + token_type_ids_document
        response_tokens = [cls_token] + response_tokens
        token_type_ids_response = [cls_token_segment_id] + token_type_ids_response

        # Convert tokens to ids
        input_ids_query = tokenizer.convert_tokens_to_ids(query_tokens)
        attention_mask_query = [1 if mask_padding_with_zero else 0] * len(
            input_ids_query
        )
        input_ids_document = tokenizer.convert_tokens_to_ids(document_tokens)
        attention_mask_document = [1 if mask_padding_with_zero else 0] * len(
            input_ids_document
        )
        input_ids_response = tokenizer.convert_tokens_to_ids(response_tokens)
        attention_mask_response = [1 if mask_padding_with_zero else 0] * len(
            input_ids_response
        )

        # Zero-pad up to the sequence length. This is static method padding
        padding_length = args.max_seq_len_query - len(input_ids_query)
        input_ids_query = input_ids_query + ([pad_token_id] * padding_length)
        attention_mask_query = attention_mask_query + (
            [0 if mask_padding_with_zero else 1] * padding_length
        )
        token_type_ids_query = token_type_ids_query + (
            [pad_token_segment_id] * padding_length
        )

        padding_length = args.max_seq_len_document - len(input_ids_document)
        input_ids_document = input_ids_document + ([pad_token_id] * padding_length)
        attention_mask_document = attention_mask_document + (
            [0 if mask_padding_with_zero else 1] * padding_length
        )
        token_type_ids_document = token_type_ids_document + (
            [pad_token_segment_id] * padding_length
        )

        padding_length = args.max_seq_len_response - len(input_ids_response)
        input_ids_response = input_ids_response + ([pad_token_id] * padding_length)
        attention_mask_response = attention_mask_response + (
            [0 if mask_padding_with_zero else 1] * padding_length
        )
        token_type_ids_response = token_type_ids_response + (
            [pad_token_segment_id] * padding_length
        )

        assert (
            len(input_ids_query) == args.max_seq_len_query
        ), "Error with input length {} vs {}".format(
            len(input_ids_query), args.max_seq_len_query
        )
        assert (
            len(input_ids_document) == args.max_seq_len_document
        ), "Error with input length {} vs {}".format(
            len(input_ids_document), args.max_seq_len_document
        )
        assert (
            len(input_ids_response) == args.max_seq_len_response
        ), "Error with input length {} vs {}".format(
            len(input_ids_response), args.max_seq_len_response
        )

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens_query: %s" % " ".join([str(x) for x in query_tokens]))
            logger.info(
                "tokens_document: %s" % " ".join([str(x) for x in document_tokens])
            )

        features.append(
            InputFeatures(
                input_ids_query=input_ids_query,
                attention_mask_query=attention_mask_query,
                token_type_ids_query=token_type_ids_query,
                input_ids_document=input_ids_document,
                attention_mask_document=attention_mask_document,
                token_type_ids_document=token_type_ids_document,
                input_ids_response=input_ids_response,
                attention_mask_response=attention_mask_response,
                token_type_ids_response=token_type_ids_response,
            )
        )
    return features


class Processor:
    def __init__(self, args) -> None:
        self.args = args

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        # read jsonline
        _data = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                _data.append(json.loads(line))
        return _data

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, datapoint in enumerate(dataset):
            guid = "%s-%s" % (set_type, i)
            # 1. query
            query = datapoint["query"]
            query = normalize_encode(
                normalize_word_diacritic(query)
            ).split()  # Some are spaced twice

            # 2. document
            document = datapoint["document"]
            document = normalize_encode(
                normalize_word_diacritic(document)
            ).split()  # Some are spaced twice

            # 3. response
            response = datapoint.get("response", "")
            if not response:
                response = query

            examples.append(
                InputExample(
                    guid=guid, query=query, document=document, response=response
                )
            )
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(
            dataset=self._read_file(os.path.join(data_path, "data.jsonl")),
            set_type=mode,
        )


def load_and_cache_examples(args, tokenizer, mode):
    processor = Processor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop()
        ),
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "eval":
            examples = processor.get_examples("eval")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, eval, test is available")

        features = create_data_to_features(
            args=args, examples=examples, tokenizer=tokenizer
        )
        # logger.info("Saving features into cached file %s", cached_features_file)
        # torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids_query = torch.tensor(
        [f.input_ids_query for f in features], dtype=torch.long
    )
    all_attention_mask_query = torch.tensor(
        [f.attention_mask_query for f in features], dtype=torch.long
    )
    all_token_type_ids_query = torch.tensor(
        [f.token_type_ids_query for f in features], dtype=torch.long
    )
    all_input_ids_document = torch.tensor(
        [f.input_ids_document for f in features], dtype=torch.long
    )
    all_attention_mask_document = torch.tensor(
        [f.attention_mask_document for f in features], dtype=torch.long
    )
    all_token_type_ids_document = torch.tensor(
        [f.token_type_ids_document for f in features], dtype=torch.long
    )
    all_input_ids_response = torch.tensor(
        [f.input_ids_response for f in features], dtype=torch.long
    )
    all_attention_mask_response = torch.tensor(
        [f.attention_mask_response for f in features], dtype=torch.long
    )
    all_token_type_ids_response = torch.tensor(
        [f.token_type_ids_response for f in features], dtype=torch.long
    )

    dataset = TensorDataset(
        all_input_ids_query,
        all_attention_mask_query,
        all_token_type_ids_query,
        all_input_ids_document,
        all_attention_mask_document,
        all_token_type_ids_document,
        all_input_ids_response,
        all_attention_mask_response,
        all_token_type_ids_response,
    )
    return dataset


# Prepare online dataset for training
class OnlineDataset(Dataset):
    def __init__(self, args, tokenizer, mode) -> None:
        super().__init__()

        self.args = args
        # Reading corpus
        file_path = os.path.join(self.args.data_dir, mode, "data.jsonl")
        logger.info("LOOKING AT {}".format(file_path))


        self.data = load_jsonl(file_path)
        self.tokenizer = tokenizer


        self.bos_token = tokenizer.bos_token
        self.bos_token_id = tokenizer.bos_token_id

        self.eos_token = tokenizer.eos_token
        self.eos_token_id = tokenizer.eos_token_id

        self.unk_token = tokenizer.unk_token
        self.unk_token_id = tokenizer.unk_token_id

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id

        self.cls_token = tokenizer.cls_token
        self.cls_token_id = tokenizer.cls_token_id

        self.sep_token = tokenizer.sep_token
        self.sep_token_id = tokenizer.sep_token_id

        # Account for [CLS] and [SEP]
        self.special_tokens_count = 2

    def __len__(self) -> int:
        return len(self.data) - 1

    def __getitem__(self, index: int):
        # preprocessing data
        data_point = self.data[index]

        # 1. query
        query = data_point["query"]
        query = normalize_encode(
            normalize_word_diacritic(query)
        ).split()  # Some are spaced twice

        # 2. document # Suggest for Text augmentation 
        document = data_point["document"]
        document = normalize_encode(
            normalize_word_diacritic(document)
        ).split()  # Some are spaced twice

        query_tokens = []
        for word in query:
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [self.unk_token]  # For handling the bad-encoded word
            query_tokens.extend(word_tokens)

        # Inverse Cloze Task, Body First Selection, Wiki Link Prediction (asymetric-retrieval-downstream)
        # Because the data_point has been processed as an inverse cloze task
        document_tokens = []
        for word in document:
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [self.unk_token]  # For handling the bad-encoded word
            document_tokens.extend(word_tokens)


        # Truncate data
        if len(query_tokens) > self.args.max_seq_len_query - self.special_tokens_count:
            query_tokens = query_tokens[
                : (self.args.max_seq_len_query - self.special_tokens_count)
            ]
        if len(document_tokens) > self.args.max_seq_len_document - self.special_tokens_count:
            document_tokens = document_tokens[
                : (self.args.max_seq_len_document - self.special_tokens_count)
            ]

        # Add [SEP] token
        query_tokens += [self.sep_token]
        token_type_ids_query = [0] * len(query_tokens)
        document_tokens += [self.sep_token]
        token_type_ids_document = [0] * len(document_tokens)


        # Add [CLS] token
        query_tokens = [self.cls_token] + query_tokens
        token_type_ids_query = [0] + token_type_ids_query
        document_tokens = [self.cls_token] + document_tokens
        token_type_ids_document = [0] + token_type_ids_document

        # Convert tokens to ids
        input_ids_query = self.tokenizer.convert_tokens_to_ids(query_tokens)
        attention_mask_query = [1] * len(
            input_ids_query
        )
        input_ids_document = self.tokenizer.convert_tokens_to_ids(document_tokens)
        attention_mask_document = [1] * len(
            input_ids_document
        )

        # Zero-pad up to the sequence length. This is static method padding
        padding_length = self.args.max_seq_len_query - len(input_ids_query)
        input_ids_query = input_ids_query + ([self.pad_token_id] * padding_length)
        attention_mask_query = attention_mask_query + (
            [0] * padding_length
        )
        token_type_ids_query = token_type_ids_query + (
            [0] * padding_length
        )

        padding_length = self.args.max_seq_len_document - len(input_ids_document)
        input_ids_document = input_ids_document + ([self.pad_token_id] * padding_length)
        attention_mask_document = attention_mask_document + (
            [0] * padding_length
        )
        token_type_ids_document = token_type_ids_document + (
            [0] * padding_length
        )

        assert (
            len(input_ids_query) == self.args.max_seq_len_query
        ), "Error with input length {} vs {}".format(
            len(input_ids_query), self.args.max_seq_len_query
        )
        assert (
            len(input_ids_document) == self.args.max_seq_len_document
        ), "Error with input length {} vs {}".format(
            len(input_ids_document), self.args.max_seq_len_document
        )


        return (
            torch.tensor(input_ids_query, dtype=torch.long),
            torch.tensor(attention_mask_query, dtype=torch.long),
            torch.tensor(token_type_ids_query, dtype=torch.long),
            torch.tensor(input_ids_document, dtype=torch.long),
            torch.tensor(attention_mask_document, dtype=torch.long),
            torch.tensor(token_type_ids_document, dtype=torch.long),
        )

