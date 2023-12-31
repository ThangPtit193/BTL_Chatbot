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
        args.model_name_or_path
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        device_map=args.device,
        args=args,
    )
    
    # Load data
    train_dataset = OnlineDataset(args, tokenizer)

    trainer = BiencoderTrainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
    )
    
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
        "--model_type",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    # Hyperparameters for training
    parser.add_argument(
        "--num_train_epochs",
        default=10,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=277,
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
