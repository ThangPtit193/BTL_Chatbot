import statistics
from typing import List, Optional

import bitsandbytes as bnb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from tqdm.auto import tqdm, trange
from transformers import AdamW
from transformers.optimization import get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import get_parameter_names

import wandb
from saturn.components.loaders.utils import convert_text_to_features
from saturn.components.models.module import (
    cosine_scores_numpy,
    dot_product_scores_numpy,
)
from saturn.utils.early_stopping import EarlyStopping
from saturn.utils.io import load_json
from saturn.utils.metrics import recall
from saturn.utils.utils import logger


class BiencoderTrainer:
    def __init__(
        self,
        args,
        model: Optional[torch.nn.Module],
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        self.args = args

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.model = model

        self.tokenizer = tokenizer

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps
                // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = (
                len(train_dataloader)
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer = self.get_optimizer()

        scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info(
            "  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps
        )
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0

        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

        # Automatic Mixed Precision
        scaler = torch.cuda.amp.GradScaler()

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
                # outputs = self.model(**inputs)
                with torch.cuda.amp.autocast():
                    (
                        total_loss,
                        loss_ct,
                        loss_ct_dpi_query,
                        loss_ct_dpi_positive,
                        loss_alignment,
                        loss_uniformity,
                    ) = self.model(**inputs)

                wandb.log({"Total Loss": total_loss.item()})
                wandb.log({"Contrastive Loss": loss_ct.item()})
                if self.args.dpi_query:
                    wandb.log({"DPI Query Loss": loss_ct_dpi_query.item()})
                if self.args.dpi_positive:
                    wandb.log({"DPI Positive Loss": loss_ct_dpi_positive.item()})
                if self.args.use_align_loss:
                    wandb.log({"Alignment Loss": loss_alignment.item()})
                if self.args.use_uniformity_loss:
                    wandb.log({"Uniformity Loss": loss_uniformity.item()})

                if self.args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / self.args.gradient_accumulation_steps

                scaler.scale(total_loss).backward()

                tr_loss += total_loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )

                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()

                    self.model.zero_grad()
                    global_step += 1

                if (
                    self.args.logging_steps > 0
                    and global_step % self.args.logging_steps == 0
                ):
                    logger.info(f"Tuning metrics: {self.args.tuning_metric}")

                    results = self.evaluate_on_benchmark()
                    for k, v in results.items():
                        results[k] = statistics.mean(v)
                    # results.update(self.evaluate())

                    wandb.log({"Loss eval": results})
                    early_stopping(
                        results[self.args.tuning_metric], self.model, self.args
                    )
                    if early_stopping.early_stop:
                        logger.info("Early stopping")
                        break

                    # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    #     self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step or early_stopping.early_stop:
                train_iterator.close()
                break
        return

    def evaluate(self):
        dataset = self.eval_dataset
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        logger.info("***** Running evaluation on eval dataset *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        eval_loss = 0.0
        eval_ct_loss = 0.0
        eval_ct_dpi_query_loss = 0.0
        eval_ct_dpi_document_loss = 0.0
        eval_ct_dpt = 0.0
        eval_alignment_loss = 0.0
        eval_uniformity_loss = 0.0

        nb_eval_steps = 0

        self.model.eval()

        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                batch = tuple(t.to(self.model.device) for t in batch)  # GPU or CPU

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "input_ids_positive": batch[2],
                    "attention_mask_positive": batch[3],
                    "is_train": True,
                }

                (
                    total_loss,
                    loss_ct,
                    loss_ct_dpi_query,
                    loss_ct_dpi_positive,
                    loss_alignment,
                    loss_uniformity,
                ) = self.model(**inputs)

                eval_loss += total_loss.item()
                eval_ct_loss += loss_ct.item()
                if self.args.dpi_query:
                    eval_ct_dpi_query_loss += loss_ct_dpi_query.item()
                if self.args.dpi_positive:
                    eval_ct_dpi_document_loss += loss_ct_dpi_positive.item()
                if self.args.use_align_loss:
                    eval_alignment_loss += loss_alignment.item()
                if self.args.use_uniformity_loss:
                    eval_uniformity_loss += loss_uniformity.item()

            nb_eval_steps += 1

        eval_loss /= nb_eval_steps
        eval_ct_loss /= nb_eval_steps
        eval_ct_dpi_query_loss /= nb_eval_steps
        eval_ct_dpi_document_loss /= nb_eval_steps
        eval_ct_dpt = 0.5 * eval_ct_dpi_query_loss + 0.5 * eval_ct_dpi_document_loss
        eval_alignment_loss /= nb_eval_steps
        eval_uniformity_loss /= nb_eval_steps

        return {
            "total_loss": eval_loss,
            "contrastive_loss": eval_ct_loss,
            "dpi_loss": eval_ct_dpt,
            "alignment_loss": eval_alignment_loss,
            "uniformity_loss": eval_uniformity_loss,
        }

    def evaluate_on_benchmark(self, top_k_results: List[int] = [5, 10, 20]):
        """
        Evaluate the performance of the model on a benchmark dataset.
        Args:
            query_and_ground_truth (List[dict]): A list of dictionaries containing query and ground truth pairs.
                Each dictionary has the following keys:
                    - 'query': The query string.
                    - 'gt': The ground truth identifier.
            corpus (List[dict]): A list of dictionaries representing the corpus.
                Each dictionary has the following keys:
                    - 'text': The text content of the document.
                    - 'meta': A dictionary containing metadata information with the following keys:
                        - 'id': The identifier of the document.
                        - 'title': The title of the document.
                        - 'grade': The grade level of the document.
                        - 'unit': The unit of the document.
                        - 'section': The section of the document.
        """

        results = {}

        for paths in self.args.benchmark_dir:
            (benchmark_path, corpus_path) = paths
            name_benchmark = benchmark_path.split("/")[-1].split(".")[0]
            benchmark = load_json(benchmark_path)
            corpus = load_json(corpus_path)

            embedding_corpus = None
            ids_corpus = []

            for i in range(0, len(corpus), self.args.eval_batch_size):
                batch_input_ids = []
                batch_attention_mask = []

                for doc in corpus[i : i + self.args.eval_batch_size]:
                    ids_corpus.append(doc["meta"]["id"])
                    document = doc["text"]
                    (
                        input_ids_document,
                        attention_mask_document,
                    ) = convert_text_to_features(
                        text=document,
                        tokenizer=self.tokenizer,
                        max_seq_len=self.args.max_seq_len_document,
                        lower_case=self.args.use_lowercase,
                        remove_punc=self.args.use_remove_punc,
                    )
                    batch_input_ids.append(input_ids_document)
                    batch_attention_mask.append(attention_mask_document)

                batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(
                    self.model.device
                )
                batch_attention_mask = torch.tensor(
                    batch_attention_mask, dtype=torch.long
                ).to(self.model.device)

                with torch.no_grad():
                    inputs = {
                        "input_ids": batch_input_ids,
                        "attention_mask": batch_attention_mask,
                    }

                    embedding = self.model(**inputs)

                if embedding_corpus is None:
                    embedding_corpus = embedding.detach().cpu().numpy()
                else:
                    embedding_corpus = np.append(
                        embedding_corpus, embedding.detach().cpu().numpy(), axis=0
                    )

            embedding_query = None
            ground_truths = []

            for i in range(0, len(benchmark), self.args.eval_batch_size):
                batch_input_ids = []
                batch_attention_mask = []

                for query in benchmark[i : i + self.args.eval_batch_size]:
                    ground_truths.append(query["gt"])
                    query = query["query"]
                    (
                        input_ids_document,
                        attention_mask_document,
                    ) = convert_text_to_features(
                        text=query,
                        tokenizer=self.tokenizer,
                        max_seq_len=self.args.max_seq_len_query,
                        lower_case=self.args.use_lowercase,
                        remove_punc=self.args.use_remove_punc,
                    )
                    batch_input_ids.append(input_ids_document)
                    batch_attention_mask.append(attention_mask_document)

                batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(
                    self.model.device
                )
                batch_attention_mask = torch.tensor(
                    batch_attention_mask, dtype=torch.long
                ).to(self.model.device)

                with torch.no_grad():
                    inputs = {
                        "input_ids": batch_input_ids,
                        "attention_mask": batch_attention_mask,
                    }

                    embedding = self.model(**inputs)

                if embedding_query is None:
                    embedding_query = embedding.detach().cpu().numpy()
                else:
                    embedding_query = np.append(
                        embedding_query, embedding.detach().cpu().numpy(), axis=0
                    )

            # scores = cosine_scores_numpy(embedding_query, embedding_corpus)
            scores = dot_product_scores_numpy(embedding_query, embedding_corpus)

            for score, ground_truth in zip(scores, ground_truths):
                for k in top_k_results:
                    if f"recall_{name_benchmark}_{k}" not in results:
                        results[f"recall_{name_benchmark}_{k}"] = []

                    ind = np.argpartition(score, -k)[-k:]
                    pred = map(ids_corpus.__getitem__, ind)

                    results[f"recall_{name_benchmark}_{k}"].append(
                        recall(pred, [ground_truth])
                    )
        return results

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
        if self.args.optimizer == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                lr=self.args.learning_rate,
                eps=self.args.adam_epsilon,
            )
        elif self.args.optimizer == "8bitAdam":
            optimizer = bnb.optim.Adam8bit(
                optimizer_grouped_parameters,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                lr=self.args.learning_rate,
                eps=self.args.adam_epsilon,
            )
        else:
            raise NotImplementedError(
                "Support is currently available only for the Adam optimizer.'"
            )
        return optimizer
