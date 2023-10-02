import argparse
import math
import os
import torch

from saturn.components.loaders.dataloader import OnlineDataset
from saturn.trainers.biencoder import BiencoderTrainer
from saturn.utils.utils import MODEL_CLASSES, MODEL_PATH_MAP, load_tokenizer


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model
    tokenizer = load_tokenizer(args)
    config_class, model_class, _ = MODEL_CLASSES[args.model_type]
    model_config = config_class.from_pretrained(
        args.model_name_or_path, finetuning_task=args.token_level
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        device_map=args.device,
        args=args,
    )
    
    # Load data
    train_dataset = OnlineDataset(args, tokenizer, "train")
    eval_dataset = OnlineDataset(args, tokenizer, "train")

    trainer = BiencoderTrainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    if args.do_train:
        trainer.train()

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
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir",
    )
    parser.add_argument(
        "--token_level",
        type=str,
        default="word-level",
        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=200,
        help="Number of steps between each logging update.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Number of steps between each checkpoint saving.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Flag indicating whether to run the training process.",
    )
    # CUDA Configuration
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Flag indicating whether to avoid using CUDA when available.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="ID of the GPU to be used for computation.",
    )

    # Hyperparameters for training
    parser.add_argument(
        "--num_train_epochs",
        default=2.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Batch size used for training.",
    )
    # Tokenizer Configuration
    parser.add_argument(
        "--max_seq_len_query",
        default=64,
        type=int,
        help="The maximum total input sequence length for query after tokenization.",
    )
    parser.add_argument(
        "--max_seq_len_document",
        default=256,
        type=int,
        help="The maximum total input sequence length for document after tokenization.",
    )

    # Optimizer Configuration
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    
    main(args)
