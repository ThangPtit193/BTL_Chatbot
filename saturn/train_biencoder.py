import argparse
import logging
from functools import partial
from typing import Dict

import torch
import wandb
from saturn.components.loaders.dataloader import load_and_cache_examples
from saturn.trainers.biencoder import BiencoderTrainer
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    set_seed,
)
from transformers.trainer_callback import PrinterCallback
from transformers.utils.logging import enable_explicit_format
from saturn.utils.utils import MODEL_CLASSES, MODEL_PATH_MAP, load_tokenizer, logger

from saturn.components.loaders.dataloader import OnlineDataset


def main(args):
    logger.info("Args={}".format(str(args)))
    run = wandb.init(
        project=args.wandb_project, name=args.wandb_run_name, config=vars(args)
    )

    set_seed(args.seed)

    # Load tokenizer and model
    tokenizer = load_tokenizer(args)
    config_class, model_class, _ = MODEL_CLASSES[args.model_type]

    if args.pretrained:
        model = model_class.from_pretrained(
            args.pretrained_path, 
            torch_dtype=torch.bfloat16
            if args.compute_dtype == torch.bfloat16
            else torch.float16,
            args=args
            )
    else:
        model_config = config_class.from_pretrained(
            args.model_name_or_path,
            finetuning_task=args.token_level
        )
        model = model_class.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16
            if args.compute_dtype == torch.bfloat16
            else torch.float16,
            config=model_config,
            args=args
        )
    logger.info(model)
    logger.info(model.dtype)
    logger.info("Vocab size: {}".format(len(tokenizer)))

    # GPU or CPU
    # torch.cuda.set_device(args.gpu_id)
    print("GPU ID :", args.gpu_id)
    print("Cuda device:", torch.cuda.current_device())
    model.to(args.device)

    # Load data
    # train_dataset = load_and_cache_examples(args, tokenizer, "train")
    # eval_dataset = load_and_cache_examples(args, tokenizer, "eval")
    train_dataset = OnlineDataset(args, tokenizer, "train")
    eval_dataset = OnlineDataset(args, tokenizer, "eval")

    trainer = BiencoderTrainer(
        args=args, 
        model=model,
        train_dataset=train_dataset,
        dev_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    trainer.evaluate_on_benchmark(trainer.query_and_ground_truth, trainer.corpus)
    # results = trainer.evaluate("eval")
    # print(f"Loss eval: {results}")
    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.evaluate_on_benchmark()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        default=None,
        required=True,
        type=str,
        help="Path to save, load model",
    )
    parser.add_argument(
        "--data_dir", default="./data", type=str, help="The input data dir"
    )
    parser.add_argument(
        "--benchmark", default="./bm/benchmark.json", type=str, help="The path to the benchmark file in JSON format."
    )
    parser.add_argument(
        "--corpus", default="./bm/corpus.json", type=str, help="The path to the corpus file in JSON format."
    )
    parser.add_argument(
        "--data_name", default="corpus.txt", type=str, help="The input data name"
    )
    parser.add_argument(
        "--token_level",
        type=str,
        default="word-level",
        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
    )

    parser.add_argument(
        "--model_type",
        default="phobert",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--logging_steps", type=int, default=200, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="semantic-similarity",
        help="Weight and bias project name.",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default="test-source", help="Run name of wandb."
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the test set"
    )

    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="Select gpu id")

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Whether to init model from pretrained base model",
    )
    parser.add_argument(
        "--pretrained_path",
        default="./baseline",
        type=str,
        help="The pretrained model path",
    )

    # hyperparam training
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--train_batch_size", default=32, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", default=64, type=int, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--max_seq_len_query",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--max_seq_len_document",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--max_seq_len_response",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=2.0,
        type=float,
        help="Total number of training epochs to perform.",
    )

    # Optimizer
    parser.add_argument(
        "--compute_dtype",
        type=torch.dtype,
        default=torch.float16,
        help="Used in quantization configs. Do not specify this argument manually.",
    )


    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-9, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta1", default=0.9, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", default=0.98, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )

    parser.add_argument(
        "--early_stopping",
        type=int,
        default=50,
        help="Number of unincreased validation step to wait for early stopping",
    )

    parser.add_argument(
        "--tuning_metric",
        default="loss",
        type=str,
        help="Metrics to tune when training",
    )
    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    main(args)
