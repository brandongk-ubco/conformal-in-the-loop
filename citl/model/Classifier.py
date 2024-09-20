from statistics import mean

import numpy as np
import pandas as pd
import pytorch_lightning as L
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from torchmetrics.classification.accuracy import Accuracy


class Classifier(L.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        lr=1e-3,
        lr_method="plateau",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.num_classes = num_classes

        self.accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )
        self.test_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )
        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )

        self.lr = lr
        self.lr_method = lr_method

    def forward(self, x):
        if x.dim() == 2:
            y_hat = self.model(x.unsqueeze(0).unsqueeze(0))
        elif x.dim() == 3:
            y_hat = self.model(x.unsqueeze(0))
        elif x.dim() == 4:
            y_hat = self.model(x)
        else:
            raise ValueError("Input must be 2, 3 or 4 dimensional")

        if isinstance(y_hat, tuple):
            y_hat, _ = y_hat

        return y_hat

    def on_train_epoch_start(self) -> None:
        self.accuracy.reset()

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y, reduction="none").mean()

        accs = self.accuracy(y_hat, y)
        self.log("accuracy", torch.mean(accs))
        self.log_dict(
            dict(
                zip(
                    [f"accuracy_{c}" for c in self.trainer.datamodule.classes],
                    accs,
                )
            ),
            on_step=True,
            on_epoch=False,
        )

        self.log("loss", loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        val_loss = F.cross_entropy(y_hat, y)

        self.val_accuracy.update(y_hat, y)

        self.log("val_loss", val_loss)


    def on_validation_epoch_end(self):
        accs = self.val_accuracy.compute()
        self.log("val_accuracy", torch.mean(accs), prog_bar=True)
        self.log_dict(
            dict(
                zip(
                    [f"val_accuracy_{c}" for c in self.trainer.datamodule.classes],
                    accs,
                )
            ),
        )

    def on_test_epoch_start(self):
        self.test_accuracy.reset()

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        self.test_accuracy.update(y_hat, y)
        test_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        accs = self.test_accuracy.compute()
        self.log("test_accuracy", torch.mean(accs), prog_bar=True)
        self.log_dict(
            dict(
                zip(
                    [f"test_accuracy_{c}" for c in self.trainer.datamodule.classes],
                    accs,
                )
            ),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.lr * 0.1
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=10,
            min_lr=1e-6,
            verbose=True,
        )
        interval = "epoch"

        if scheduler:
            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "interval": interval,
                    "monitor": "val_loss",
                }
            ]

        return optimizer


__all__ = ["Classifier"]
