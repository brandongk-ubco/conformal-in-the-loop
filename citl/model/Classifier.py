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

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y, reduction="none").mean()

        self.accuracy(y_hat, y)
        self.log("accuracy", self.accuracy.compute().mean())
        self.log_dict(
            dict(
                zip(
                    [f"accuracy_{c}" for c in self.trainer.datamodule.classes],
                    self.accuracy.compute().cpu().numpy(),
                )
            ),
            on_step=True,
            on_epoch=False,
        )

        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        val_loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)
        self.log("val_accuracy", self.accuracy.compute().mean())
        self.log_dict(
            dict(
                zip(
                    [f"val_accuracy_{c}" for c in self.trainer.datamodule.classes],
                    self.accuracy.compute().cpu().numpy(),
                )
            ),
            on_step=True,
            on_epoch=False,
        )

        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        self.accuracy(y_hat, y)
        self.log("test_accuracy", self.accuracy.compute().mean())
        self.log_dict(
            dict(
                zip(
                    [f"test_accuracy_{c}" for c in self.trainer.datamodule.classes],
                    self.accuracy.compute().cpu().numpy(),
                )
            ),
            on_step=True,
            on_epoch=False,
        )

        test_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)

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
