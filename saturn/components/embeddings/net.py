import json
import os
from typing import *
from typing import Dict, Tuple, Iterable, Type, Callable

import torch
import transformers
from comet.lib import logger
from sentence_transformers import SentenceTransformer
from sentence_transformers.model_card_templates import ModelCardTemplate
from sentence_transformers.util import fullname
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange
from transformers import (
    AutoModel,
)
import shutil
from venus.wrapper import axiom_wrapper

_logger = logger.get_logger(__name__)


class AutoModelForSentenceEmbedding(torch.nn.Module):
    def __init__(self, model_name, tokenizer, normalize=True):
        super(AutoModelForSentenceEmbedding, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.tokenizer = tokenizer

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)

    def save_pretrained(self, output_path):
        self.tokenizer.save_pretrained(output_path)
        self.model.config.save_pretrained(output_path)
        torch.save(self.model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))


# class CustomSentenceEmbedding(SentenceEmbedding):
#     def train_negative_ranking(
#         self,
#         train_dataset,
#         dev_evaluator=None,
#         batch_size=16,
#         epochs=10,
#         warmup_steps=1000,
#         model_save_path='models',
#         evaluation_steps=5000,
#         use_amp=True,
#         optimizer_params: Dict[str, object] = {'lr': 2e-5},
#         optimizer_class: Type[Optimizer] = transformers.AdamW,
#         scheduler='WarmupLinear',
#         weight_decay: float = 0.01,
#         max_grad_norm: float = 1,
#         save_best_model=True,
#         show_progress_bar: bool = True,
#         **kwargs,
#     ):
#         train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
#         train_loss = losses.TripletLoss(model=self.model)
#
#         # Train the model
#         self.model.fit(
#             train_objectives=[(train_dataloader, train_loss)],
#             evaluator=dev_evaluator,
#             epochs=epochs,
#             warmup_steps=warmup_steps,
#             output_path=model_save_path,
#             evaluation_steps=evaluation_steps,
#             use_amp=use_amp,
#             optimizer_params=optimizer_params,
#             optimizer_class=optimizer_class,
#             scheduler=scheduler,
#             weight_decay=weight_decay,
#             max_grad_norm=max_grad_norm,
#             save_best_model=save_best_model,
#             show_progress_bar=show_progress_bar,
#             **kwargs,
#         )
#         self.model.save(model_save_path)
#

class CustomSentenceTransformer(SentenceTransformer):
    @classmethod
    def from_pretrained(cls, model_name_or_path: Text = None) -> "CustomSentenceTransformer":
        """

        Args:
            model_name_or_path: The model name or path. If name provided,
                                It will be loaded from hugging face hub or from axiom
        """
        model = None
        if os.path.isdir(model_name_or_path):
            try:
                model = CustomSentenceTransformer(model_name_or_path)
            except Exception as e:
                _logger.error(f"Cannot load pretrained model from {model_name_or_path}, "
                              f"Because {e}")
        else:
            model_name_or_path = axiom_wrapper.fetch_model(model_name_or_path)
            model = CustomSentenceTransformer(model_name_or_path)
            model.max_seq_length = 128
        return model

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator=None,
            epochs: int = 1,
            steps_per_epoch=None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_epoch: int = 500,
            checkpoint_save_total_limit: int = 1,
            resume_from_checkpoint: str = False,
            save_by_epoch: bool = True,
            model_save_total_limit: int = None
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_epoch: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        :param resume_from_checkpoint: Path to checkpoint to resume training from
        """

        ##Add info to model card
        # info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions = []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps(
            {"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch,
             "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),
             "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps,
             "max_grad_norm": max_grad_norm, "callback": callback}, indent=4, sort_keys=True)
        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}",
                                                                                                     info_loss_functions).replace(
            "{FIT_PARAMETERS}", info_fit_parameters)

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []

        # Handle checkpoint
        checkpoints = None
        epoch_offset = None
        if resume_from_checkpoint:
            if not os.path.exists(resume_from_checkpoint):
                raise ValueError("Checkpoint folder {} does not exist".format(resume_from_checkpoint))
            checkpoints = torch.load(
                resume_from_checkpoint, map_location=self._target_device
            )
            epoch_offset = checkpoints["epoch"]
            self.load_state_dict(checkpoints["model_state_dict"])
        #     param_optimizer = list(loss_model.named_parameters())
        #
        #     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        #     optimizer_grouped_parameters = [
        #         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        #          'weight_decay': weight_decay},
        #         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        #     ]
        #     optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        #     scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
        #                                         t_total=num_train_steps)
        #     optimizers = [optimizer.load_state_dict(optimizer_state)
        #                   for optimizer_state in checkpoints["optimizers_state_dict"]]
        #     schedulers = [scheduler_obj.load_state_dict(scheduler_state)
        #                   for scheduler_state in checkpoints["schedulers_state_dict"]]
        #     epoch_offset = checkpoints["epoch"]
        # else:
        for idx, loss_model in enumerate(loss_models):
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                                t_total=num_train_steps)
            if not checkpoints:
                optimizers.append(optimizer)
                schedulers.append(scheduler_obj)
            else:
                _logger.info("Load optimizer and scheduler state from checkpoint")
                optimizer.load_state_dict(checkpoints["optimizers_state_dict"][idx])
                scheduler_obj.load_state_dict(checkpoints["schedulers_state_dict"][idx])
                optimizers.append(optimizer)
                schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False

        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            if epoch_offset and epoch < epoch_offset:
                continue
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1
            if bool(
                checkpoint_path is not None
                and checkpoint_save_epoch is not None
                and checkpoint_save_epoch > 0
                and (epoch + 1) % checkpoint_save_epoch == 0
            ):
                self.save_checkpoint(epoch, optimizers, schedulers, checkpoint_path, checkpoint_save_total_limit)

            if bool(
                save_by_epoch is True
                and output_path is not None
            ):
                self.save_model(output_path,model_save_total_limit, epoch)

        if evaluator is None and output_path is not None:  # No evaluator, but output path: save final model version
            self.save(os.path.join(output_path, "final_model"))

    def save_checkpoint(self, epoch, optimizers, schedulers, checkpoint_path, checkpoint_save_total_limit):
        _logger.info(f"Saving checkpoint for epoch {epoch}")
        model_checkpoint_path = os.path.join(checkpoint_path, "epoch-{}".format(epoch))
        if not os.path.exists(model_checkpoint_path):
            os.makedirs(model_checkpoint_path)

        model_checkpoint_file = os.path.join(model_checkpoint_path, "checkpoint.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizers_state_dict': [optimizer.state_dict() for optimizer in optimizers],
            'schedulers_state_dict': [scheduler.state_dict() for scheduler in schedulers],
        }, model_checkpoint_file)
        # Delete old checkpoints
        if checkpoint_save_total_limit is not None and checkpoint_save_total_limit > 0:
            old_checkpoints = []
            for subdir in os.listdir(checkpoint_path):
                prefix, epoch = subdir.split("-")
                if epoch.isdigit():
                    old_checkpoints.append({'epoch': int(epoch), 'path': os.path.join(checkpoint_path, subdir)})

            if len(old_checkpoints) > checkpoint_save_total_limit:
                old_checkpoints = sorted(old_checkpoints, key=lambda x: x['epoch'])
                shutil.rmtree(old_checkpoints[0]['path'])

    def save_model(self, model_path, model_save_total_limit, epoch):
        # Store new checkpoint
        self.save(os.path.join(model_path, "epoch-{}".format(epoch)))

        # Delete old checkpoints
        if model_save_total_limit is not None and model_save_total_limit > 0:
            old_checkpoints = []
            for subdir in os.listdir(model_path):
                epoch_numer = subdir.split("-")[-1]
                old_checkpoints.append({'epochs': int(epoch_numer), 'path': os.path.join(model_path, subdir)})

            if len(old_checkpoints) > model_save_total_limit:
                old_checkpoints = sorted(old_checkpoints, key=lambda x: x['epochs'])
                shutil.rmtree(old_checkpoints[0]['path'])

