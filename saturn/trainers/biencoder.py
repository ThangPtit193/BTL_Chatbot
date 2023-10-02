import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from lion_pytorch import Lion
from transformers.optimization import get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import get_parameter_names

from saturn.components.loaders.utils import convert_text_to_features
from saturn.utils.io import load_json, load_jsonl

class BiencoderTrainer:
    def __init__(
            self,
            args,
            model
            train_dataset
            eval_dataset
            tokenizer
    ):
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.model = model
        self.tokenizer = tokenizer

    def train(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
        )
        t_total = (
                len(train_dataloader)
                * self.args.num_train_epochs
        )
        optimizer = self.get_optimizer()

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
    
        tr_loss = 0.0

        self.model.zero_grad()

        train_iterator = int(self.args.num_train_epochs)

        for _ in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader, desc="Iteration", position=0, leave=True
            )
            logger.info(f"Epoch {_}")

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.model.device) for t in batch)  # GPU or CPU

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "input_ids_positive": batch[2],
                    "attention_mask_positive": batch[3],
                    "is_train": True,
                }
                
                loss = self.model(**inputs)
                    
                loss.backward()
                tr_loss += loss.item()
                optimizer.step()
                self.model.zero_grad()
        return

    def get_optimizer(self):
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = Lion(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
        )
        return optimizer
