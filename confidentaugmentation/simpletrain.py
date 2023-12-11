import os
import sys

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from confidentaugmentation import cli
from confidentaugmentation.data import AugmentedMNISTDataModule
import torch.nn as nn
import torch.nn.functional as F

from .model.SimpleTrainer import SimpleTrainer


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 1, 1)
        self.conv3 = nn.Conv2d(1, 16, 3)
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x) + x[:, :, 1:-1, 1:-1]
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = self.conv3(x) + x[:, :, 1:-1, 1:-1]
        x = self.pool(F.relu(x))

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


@cli.command()
def simpletrain(
    augmentation_policy_path: str = "./policies/mnist.yaml",
    lr_method: str = "plateau",
    lr: float = 1e-3,
    optimizer: str = "Adam",
):
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True, warn_only=True)

    dm = AugmentedMNISTDataModule(augmentation_policy_path)

    model = SimpleTrainer(
        Net(),
        num_classes=dm.num_classes,
        lr_method=lr_method,
        lr=lr,
        optimizer=optimizer,
    )

    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        model_checkpoint,
    ]

    save_dir = os.path.join(
        "simple_logs",
        lr_method,
        optimizer,
    )

    trainer_logger = TensorBoardLogger(
        save_dir=save_dir,
        name="mnist",
    )

    trainer = L.Trainer(
        logger=trainer_logger,
        num_sanity_val_steps=sys.maxsize,
        max_epochs=sys.maxsize,
        deterministic=True,
        callbacks=callbacks,
        log_every_n_steps=10,
        accumulate_grad_batches=3
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)