import argparse
import math
import os

import torch
from transformers import set_seed

import wandb
from saturn.components.loaders.dataloader import OnlineDataset
from saturn.trainers.biencoder import BiencoderTrainer
from saturn.utils.utils import MODEL_CLASSES, MODEL_PATH_MAP, load_tokenizer, logger


def main(args):
    logger.info("Args={}".format(str(args)))

    set_seed(args.seed)

    # Pre Setup
    args.device_map = "auto"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.ddp = args.world_size != 1
    if args.ddp:
        args.device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        args.gradient_accumulation_steps = (
                args.gradient_accumulation_steps // args.world_size
        )

    # Load tokenizer and model
    tokenizer = load_tokenizer(args)
    config_class, model_class, _ = MODEL_CLASSES[args.model_type]

    if args.pretrained:
        print("Loading model ....")
        model = model_class.from_pretrained(
            args.pretrained_path,
            torch_dtype=args.compute_dtype,
            device_map=args.device_map,
            args=args,
        )
    else:
        model_config = config_class.from_pretrained(
            args.model_name_or_path, finetuning_task=args.token_level
        )
        model = model_class.from_pretrained(
            args.model_name_or_path,
            torch_dtype=args.compute_dtype,
            config=model_config,
            device_map=args.device_map,
            args=args,
        )

        for param in model.parameters():
            param.requires_grad = False

    if args.resize_embedding_model:
        model.resize_token_embeddings(
            int(8 * math.ceil(len(tokenizer) / 8.0))
        )  # make the vocab size multiple of 8 # magic number

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    logger.info(model)
    logger.info(model.dtype)
    logger.info("Vocab size: {}".format(len(tokenizer)))

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
    import statistics

    results = trainer.evaluate_on_benchmark()
    for k, v in results.items():
        results[k] = statistics.mean(v)
    print(results)

    if args.do_train:
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, config=vars(args)
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
        "--token_level",
        type=str,
        default="word-level",
        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
    )
    parser.add_argument(
        "--model_type",
        default="unsim-cse-vietnamese",
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
        "--eval_steps",
        type=int,
        default=100,
        help="Number of steps between each model evaluation.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Number of steps between each checkpoint saving.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="semantic-similarity",
        help="Name of the Weight and Bias project.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="test-source",
        help="Name of the run for Weight and Bias.",
    )
    parser.add_argument(
        "--wandb_watch",
        type=str,
        default="false",
        help="Whether to enable tracking of gradients and model topology in Weight and Bias.",
    )
    parser.add_argument(
        "--wandb_log_model",
        type=str,
        default="false",
        help="Whether to enable model versioning in Weight and Bias.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Flag indicating whether to run the training process.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Whether to initialize the model from a pretrained base model.",
    )
    parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str,
        help="Path to the pretrained model.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed used for initialization.",
    )
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
    parser.add_argument(
        "--eval_batch_size",
        default=64,
        type=int,
        help="Batch size used for evaluation.",
    )
    parser.add_argument(
        "--dataloader_drop_last",
        type=bool,
        default=True,
        help="Toggle whether to drop the last incomplete batch in the dataloader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of workers for the dataloader.",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        type=bool,
        default=True,
        help="Toggle whether to use pinned memory in the dataloader.",
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
    parser.add_argument(
        "--max_seq_len_response",
        default=64,
        type=int,
        help="The maximum total input sequence length for response after tokenization.",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        default=False,
        type=bool,
        help="Whether to use the fast tokenizer. If set to True, a faster tokenizer will be used for tokenizing the input data. This can improve the performance of tokenization but may sacrifice some tokenization quality. If set to False, a slower but more accurate tokenizer will be used. Default value is True.",
    )

    # Optimizer Configuration
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage during training. "
             "When this flag is set, intermediate activations are recomputed during "
             "backward pass, which can be memory-efficient but might increase "
             "training time.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="cosine",
        type=str,
        help="Type of learning rate scheduler to use. Available options are: 'cosine', 'step', 'plateau'. "
             "The default is 'cosine', which uses a cosine annealing schedule. ",
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
        "--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps."
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

    # Model Configuration
    parser.add_argument(
        "--resize_embedding_model",
        action="store_true",
        help="Resize model embedding model following length of vocab.",
    )
    parser.add_argument(
        "--compute_dtype",
        type=torch.dtype,
        default=torch.float,
        help="Used in quantization configs. Do not specify this argument manually.",
    )
    parser.add_argument(
        "--pooler_type",
        default="cls",
        type=str,
        help="What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last).",
    )
    parser.add_argument(
        "--sim_fn",
        default="cosine",
        type=str,
        help="Similarity function to use for calculations (default: 'consie').",
    )
    parser.add_argument(
        "--label_smoothing",
        default=0.0,
        type=float,
        help="Amount of label smoothing to apply (default: 0.0).",
    )
    parser.add_argument(
        "--dpi_query",
        action="store_true",
        help="Flag to enable DPI query.",
    )
    parser.add_argument(
        "--coff_dpi_query",
        default=0.1,
        type=float,
        help="Coefficient for DPI query calculations (default: 0.1).",
    )
    parser.add_argument(
        "--dpi_positive",
        action="store_true",
        help="Flag to enable DPI positive.",
    )
    parser.add_argument(
        "--coff_dpi_positive",
        default=0.1,
        type=float,
        help="Coefficient for DPT positive calculations (default: 0.1).",
    )
    parser.add_argument(
        "--use_align_loss",
        action="store_true",
        help="Enable alignment loss mode.",
    )
    parser.add_argument(
        "--coff_alignment",
        default=0.05,
        type=float,
        help="Coefficient for alignment loss calculations (default: 0.05).",
    )
    parser.add_argument(
        "--use_uniformity_loss",
        action="store_true",
        help="Enable uniformity loss mode.",
    )
    parser.add_argument(
        "--coff_uniformity",
        default=0.05,
        type=float,
        help="Coefficient for uniformity loss calculations (default: 0.05).",
    )
    parser.add_argument(
        "--use_negative",
        default=True,
        action="store_true"
    )
    parser.add_argument(
        "--weight_hard_negative",
        default=0.2,
        type=float,
    )

    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    # Check if parameter passed or if set within environ
    args.use_wandb = len(args.wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if len(args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = args.wandb_watch
    if len(args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model

    args.benchmark_dir = [
        [
            "/home/black/data/benchmark_history/bm_history_v400.jsonl",
            "/home/black/data/benchmark_history/corpus_history.json",
        ],
        [
            "/home/black/data/benchmark_history/bm_history_v200.jsonl",
            "/home/black/data/benchmark_history/corpus_history.json",
        ],
        [
            "/home/black/data/benchmark_history/bm_history_cttgt2.jsonl",
            "/home/black/data/benchmark_history/corpus_history.json",
        ],
        [
            "/home/black/saturn/data/benchmark/bm_visquad.jsonl",
            "/home/black/saturn/data/benchmark/corpus_visquad.json"
        ]
    ]
    main(args)
