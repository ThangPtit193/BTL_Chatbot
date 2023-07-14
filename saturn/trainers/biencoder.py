import os

import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from tqdm.auto import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from saturn.utils.early_stopping import EarlyStopping
from saturn.utils.utils import logger

from typing import List
from saturn.utils.normalize import normalize_encode, normalize_word_diacritic
from saturn.components.models.module import cosine_similarity
from saturn.utils.io import load_json, load_jsonl

class BiencoderTrainer:
    def __init__(
        self,
        args,
        model,
        train_dataset=None,
        dev_dataset=None,
        test_dataset=None,
        tokenizer=None
    ):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.model = model

        self.tokenizer = tokenizer

        self.query_and_ground_truth = load_jsonl(self.args.benchmark)
        self.corpus = load_json(self.args.corpus)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
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
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
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

        for _ in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader, desc="Iteration", position=0, leave=True
            )
            print("\nEpoch", _)

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.args.device) for t in batch)  # GPU or CPU

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "input_ids_positive": batch[2],
                    "attention_mask_positive": batch[3],
                }
                outputs = self.model(**inputs)
                loss = outputs[0]

                wandb.log({"Loss train": loss.item()})

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if (
                        self.args.logging_steps > 0
                        and global_step % self.args.logging_steps == 0
                    ):
                        logger.info(f"Tuning metrics: {self.args.tuning_metric}")
                        
                        self.evaluate_on_benchmark(self.query_and_ground_truth, self.corpus)
                        results = self.evaluate("eval")
                        wandb.log({"Loss eval": results})
                        early_stopping(results, self.model, self.args)
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
        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "eval":
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size
        )

        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0


        self.model.eval()

        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                batch = tuple(t.to(self.args.device) for t in batch)  # GPU or CPU

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "input_ids_positive": batch[2],
                    "attention_mask_positive": batch[3],
                }

                outputs = self.model(**inputs)
                contrastive_loss = outputs[0]
                eval_loss += contrastive_loss.item()

            nb_eval_steps += 1

        return eval_loss / nb_eval_steps

    def evaluate_on_benchmark(self, query_and_ground_truth: List[dict], corpus: List[dict]):
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
        
        # Embdding corpus:
        embedding_corpus = None
        ids_corpus = []

        for i in range(0, len(corpus), self.args.eval_batch_size):
            batch_input_ids = []
            batch_attention_mask = []
            batch_token_type_ids = []

            batch_ids = []
            for document in corpus[i: i+self.args.eval_batch_size]:
                batch_ids.append(
                    document['meta']['id']
                )
                document = normalize_encode(
                    normalize_word_diacritic(document['text'])
                ).split()  # Some are spaced twice
                document_tokens = []
                for word in document:
                    word_tokens = self.tokenizer.tokenize(word)
                    if not word_tokens:
                        word_tokens = [self.tokenizer.unk_token]  # For handling the bad-encoded word
                    document_tokens.extend(word_tokens)
                if len(document_tokens) > self.args.max_seq_len_document - 2:
                    document_tokens = document_tokens[
                        : (self.args.max_seq_len_document - 2)
                    ]
                document_tokens += [self.tokenizer.sep_token]
                token_type_ids_document = [0] * len(document_tokens)

                document_tokens = [self.tokenizer.cls_token] + document_tokens
                token_type_ids_document = [0] + token_type_ids_document

                input_ids_document = self.tokenizer.convert_tokens_to_ids(document_tokens)
                attention_mask_document = [1] * len(
                    input_ids_document
                )

                padding_length = self.args.max_seq_len_document - len(input_ids_document)
                input_ids_document = input_ids_document + ([self.tokenizer.pad_token_id] * padding_length)
                attention_mask_document = attention_mask_document + (
                    [0] * padding_length
                )
                token_type_ids_document = token_type_ids_document + (
                    [0] * padding_length
                )
                
                batch_input_ids.append(
                    input_ids_document
                )
                batch_attention_mask.append(
                    attention_mask_document
                )
                batch_token_type_ids.append(
                    token_type_ids_document
                )

            assert len(batch_input_ids) == len(batch_token_type_ids) == len(batch_attention_mask) == len(batch_ids)

            batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(self.args.device)
            batch_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long).to(self.args.device)
            batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long).to(self.args.device)

            with torch.no_grad():
                inputs = {
                        "input_ids": batch_input_ids,
                        "attention_mask": batch_attention_mask,
                        "is_trainalbe": False
                        }

                outputs = self.model(**inputs)

            if embedding_corpus is None:
                embedding_corpus = outputs.detach().cpu().numpy()
                ids_corpus.extend(
                    batch_ids
                )
            else:
                embedding_corpus = np.append(embedding_corpus, outputs.detach().cpu().numpy(), axis=0)
                ids_corpus.extend(
                    batch_ids
                )

        # Embedding query:
        embedding_query = None
        ground_truths = []

        for i in range(0, len(query_and_ground_truth), self.args.eval_batch_size):
            batch_input_ids = []
            batch_attention_mask = []
            batch_token_type_ids = []

            batch_gts = []
            for query in query_and_ground_truth[i: i+self.args.eval_batch_size]:
                batch_gts.append(
                    query['gt']
                )
                query = normalize_encode(
                    normalize_word_diacritic(query['query'])
                ).split()  # Some are spaced twice
                query_tokens = []
                for word in query:
                    word_tokens = self.tokenizer.tokenize(word)
                    if not word_tokens:
                        word_tokens = [self.unk_token]  # For handling the bad-encoded word
                    query_tokens.extend(word_tokens)
                if len(query_tokens) > self.args.max_seq_len_query - 2:
                    query_tokens = query_tokens[
                        : (self.args.max_seq_len_query - 2)
                    ]
                query_tokens += [self.tokenizer.sep_token]
                token_type_ids_query = [0] * len(query_tokens)

                query_tokens = [self.tokenizer.cls_token] + query_tokens
                token_type_ids_query = [0] + token_type_ids_query

                input_ids_query = self.tokenizer.convert_tokens_to_ids(query_tokens)
                attention_mask_query = [1] * len(
                    input_ids_query
                )

                padding_length = self.args.max_seq_len_query - len(input_ids_query)
                input_ids_query = input_ids_query + ([self.tokenizer.pad_token_id] * padding_length)
                attention_mask_query = attention_mask_query + (
                    [0] * padding_length
                )
                token_type_ids_query = token_type_ids_query + (
                    [0] * padding_length
                )
                
                batch_input_ids.append(
                    input_ids_query
                )
                batch_attention_mask.append(
                    attention_mask_query
                )
                batch_token_type_ids.append(
                    token_type_ids_query
                )

            assert len(batch_input_ids) == len(batch_token_type_ids) == len(batch_attention_mask) == len(batch_gts)

            batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(self.args.device)
            batch_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long).to(self.args.device)
            batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long).to(self.args.device)

            with torch.no_grad():
                inputs = {
                        "input_ids": batch_input_ids,
                        "attention_mask": batch_attention_mask,
                        "is_trainalbe": False
                        }

                outputs = self.model(**inputs)

            if embedding_query is None:
                embedding_query = outputs.detach().cpu().numpy()
                ground_truths.extend(
                    batch_gts
                )
            else:
                embedding_query = np.append(embedding_query, outputs.detach().cpu().numpy(), axis=0)
                ground_truths.extend(
                    batch_gts
                )
        
        scores = cosine_similarity(embedding_query, embedding_corpus)
        indices = np.argsort(scores, axis=-1)

        # TODO get top k indices in each row and assign them with ids -> get recall scall -> 


    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(
                self.args.model_dir, config=self.config, args=self.args
            )
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except Exception:
            raise Exception("Some model files might be missing...")
