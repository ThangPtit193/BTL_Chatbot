import os
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from saturn.components.loaders.utils import convert_text_to_features
from saturn.utils.io import load_jsonl


# Prepare online dataset for training
class OnlineDataset(Dataset):
    def __init__(
            self, args, tokenizer: PreTrainedTokenizer, mode: str = "train"
    ) -> None:
        super().__init__()

        self.args = args
        # Reading corpus
        file_path = os.path.join(
            self.args.data_dir, mode, "data.jsonl"
        )
        logger.info("LOOKING AT {}".format(file_path))

        self.data = load_jsonl(file_path)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data) - 1

    def __getitem__(self, index: int):
        # preprocessing data
        data_point = self.data[index]

        # 1. query
        query = data_point["query"]
        # 2. document # Suggest for Text augmentation
        document = data_point["document"]

        input_ids_query, attention_mask_query = convert_text_to_features(
            text=query,
            tokenizer=self.tokenizer,
            max_seq_len=self.args.max_seq_len_query,
        )
        input_ids_document, attention_mask_document = convert_text_to_features(
            text=document,
            tokenizer=self.tokenizer,
            max_seq_len=self.args.max_seq_len_document,
        )
        return (
            torch.tensor(input_ids_query, dtype=torch.long),
            torch.tensor(attention_mask_query, dtype=torch.long),
            torch.tensor(input_ids_document, dtype=torch.long),
            torch.tensor(attention_mask_document, dtype=torch.long),
        )
