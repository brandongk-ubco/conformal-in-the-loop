import numpy as np
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification.accuracy import Accuracy


class SimpleTrainer(L.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        lr=1e-3,
        lr_method="plateau",
        optimizer="Adam"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.classes_ = range(num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.classes_))

        self.lr = lr
        self.lr_method = lr_method
        self.optimizer = optimizer

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
        x, y = batch

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)

        self.log("val_accuracy", self.accuracy, on_step=False, on_epoch=True)
        self.log("val_loss", test_loss, on_step=False, on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.accuracy.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)
        self.log("test_accuracy", self.accuracy, on_step=False, on_epoch=True)

        self.log("test_loss", test_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        dataloader = self.trainer.datamodule.train_dataloader()

        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.lr * 0.1,
                nesterov=True,
            )
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.lr * 0.1
            )
        else:
            raise NotImplementedError("Optimizer not implemented.")

        scheduler = None

        if self.lr_method == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr * 10,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(dataloader),
                anneal_strategy="cos",
                pct_start=self.warmup_epochs / self.trainer.max_epochs,
                cycle_momentum=False,
                div_factor=10,
                final_div_factor=100,
                three_phase=True,
            )
            interval = "step"

        if self.lr_method == "plateau":
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
                    "monitor": "val_loss"
                }
            ]

        return optimizer


__all__ = ["SimpleTrainer"]
