import torch
from torch.utils.data import DataLoader, Dataset
from lion_pytorch import Lion
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from saturn.components.loaders.utils import convert_text_to_features

class BiencoderTrainer:
    def __init__(self,args,modeltrain_dataset):
        self.args = args
        self.train_dataset = train_dataset
        self.model = model

    def train(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
        )

        optimizer = self.get_optimizer()

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)

        self.model.zero_grad()

        train_iterator = int(self.args.num_train_epochs)

        for _ in train_iterator:
            logger.info(f"Epoch {_}")

            for step, batch in enumerate(train_dataloader):
                self.model.train()
                batch = tuple(t.to(self.model.device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "input_ids_positive": batch[2],
                    "attention_mask_positive": batch[3],
                    "is_train": True,
                }
                
                loss = self.model(**inputs)
                    
                loss.backward()
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
