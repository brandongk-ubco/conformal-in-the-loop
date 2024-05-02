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
from torchmetrics.classification.jaccard import JaccardIndex

from ..ConformalClassifier import ConformalClassifier
from ..utils.visualize_segmentation import visualize_segmentation


class CITLSegmenter(L.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        selectively_backpropagate=False,
        control_on_realized=False,
        alpha=0.10,
        val_alpha=0.10,
        lr=1e-3,
        lr_method="plateau",
        method="score",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.conformal_classifier = ConformalClassifier(method=method)

        self.num_classes = num_classes

        self.accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="micro", ignore_index=0
        )
        self.jaccard = JaccardIndex(
            task="multiclass", num_classes=num_classes, average="micro", ignore_index=0
        )

        self.selectively_backpropagate = selectively_backpropagate
        self.alpha = alpha
        self.val_alpha = val_alpha
        self.lr = lr
        self.pixel_dropout = 0.0
        self.lr_method = lr_method
        self.weight_decay = 0.0
        self.method = method
        self.control_on_realized = control_on_realized

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

    def on_train_start(self) -> None:
        self.initial_train_set = self.trainer.train_dataloader.dataset
        self.initial_train_size = float(len(self.trainer.train_dataloader.dataset))

    def on_train_epoch_start(self) -> None:
        current_train_size = float(len(self.trainer.train_dataloader.dataset))
        self.log("Train Dataset Size", current_train_size / self.initial_train_size)
        self.class_weights = dict(
            zip(range(self.num_classes), [0.0] * self.num_classes)
        )
        self.class_counts = dict(
            zip(range(self.num_classes), [1e-5] * self.num_classes)
        )

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        y_hat = self(x)

        self.conformal_classifier.reset()
        self.conformal_classifier.append(y_hat, y)
        _, uncertainty = self.conformal_classifier.measure_uncertainty(alpha=self.alpha)

        metrics = dict([(k, v.float().mean()) for k, v in uncertainty.items()])
        self.log_dict(
            metrics, on_step=True, on_epoch=False, prog_bar=False, logger=True
        )

        if type(self.trainer.logger) is TensorBoardLogger and self.current_epoch == 0:
            img, target = x[1, :, :, :], y[1]
            if img.ndim > 2:
                img = img.moveaxis(0, -1)
            fig = visualize_segmentation(img.detach().cpu(), mask=target.detach().cpu())
            self.logger.experiment.add_figure("example_image", fig, self.global_step)
            plt.close()

        if self.selectively_backpropagate:
            prediction_set_size = uncertainty["prediction_set_size"].reshape(y.shape)
            loss = F.cross_entropy(y_hat, y.long(), reduction="none")
            loss = loss * prediction_set_size
            loss = loss.mean()

            y_flt = y.flatten()
            p_flt = prediction_set_size.flatten()
            for clazz in range(self.num_classes):
                class_idxs = y_flt == clazz
                count = class_idxs.sum()
                weights = p_flt[class_idxs].sum()
                self.class_counts[clazz] += count
                self.class_weights[clazz] += weights

        else:
            loss = F.cross_entropy(y_hat, y.long(), reduction="none").mean()

        self.accuracy(y_hat, y)
        self.log("accuracy", self.accuracy)

        self.jaccard(y_hat, y)
        self.log("jaccard", self.jaccard)

        self.log("loss", loss)
        return loss

    def on_train_epoch_end(self) -> None:
        plt.figure()

        weights = {}

        for k in range(self.num_classes):
            label = self.trainer.datamodule.classes[k]
            weight = float(self.class_weights[k] / self.class_counts[k])
            weights[label] = weight
            self.log(f"mean_weight_{label}", weight, on_step=False, on_epoch=True)
            self.log(
                f"count_{label}",
                self.class_counts[k],
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"weight_{label}",
                self.class_weights[k],
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        weights_df = pd.DataFrame([weights]).T
        weights_df = weights_df.reset_index()

        weights_df = weights_df.rename(columns={"index": "class", 0: "mean_weight"})

        sns_plot = sns.barplot(data=weights_df, x="class", y="mean_weight")

        sns_plot.set_title(
            f"Mean Weighting of Each Class (epoch: {self.current_epoch + 1})"
        )
        sns_plot.set_xlabel(f"Class")
        sns_plot.set_ylabel("Mean Weighting")
        plt.xticks(rotation=90)
        plt.tight_layout()

        if type(self.trainer.logger) is TensorBoardLogger:
            self.logger.experiment.add_figure(
                "mean_class_weights",
                sns_plot.get_figure(),
                self.global_step,
            )

        elif type(self.trainer.logger) is NeptuneLogger:
            self.logger.experiment["training/mean_class_weights"].append(
                sns_plot.get_figure()
            )
        plt.close()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.conformal_classifier.reset()
        self.val_batch_idx_fit_uncertainty = (
            len(self.trainer.datamodule.val_dataloader()) // 5
        )

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        val_loss = F.cross_entropy(y_hat, y.long(), reduction="none").mean()

        if batch_idx < self.val_batch_idx_fit_uncertainty:
            self.conformal_classifier.append(y_hat, y)
        elif batch_idx == self.val_batch_idx_fit_uncertainty:
            self.conformal_classifier.fit()
        else:
            self.conformal_classifier.append(y_hat, y)
            _, uncertainty = self.conformal_classifier.measure_uncertainty(
                alpha=self.val_alpha
            )

            metrics = dict(
                [(f"val_{k}", v.float().mean()) for k, v in uncertainty.items()]
            )
            self.log_dict(metrics, prog_bar=True)

        self.accuracy(y_hat, y)
        self.log("val_accuracy", self.accuracy, on_step=False, on_epoch=True)

        self.jaccard(y_hat, y)
        self.log("val_jaccard", self.jaccard, on_step=False, on_epoch=True)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        test_loss = F.cross_entropy(y_hat, y.long(), reduction="none").mean()

        self.conformal_classifier.reset()
        self.conformal_classifier.append(y_hat, y)
        _, uncertainty = self.conformal_classifier.measure_uncertainty(
            alphas=self.val_alpha
        )

        metrics = dict(
            [(f"test_{k}", v.float().mean()) for k, v in uncertainty.items()]
        )
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        img, target = x[1, :, :, :], y[1]
        if img.ndim > 2:
            img = img.moveaxis(0, -1)
        fig = visualize_segmentation(
            img.detach().cpu(),
            mask=target.detach().cpu(),
            prediction=y_hat[1].detach().cpu(),
            prediction_set_size=uncertainty["prediction_set_size"].reshape(y.shape)[1],
        )
        self.logger.experiment.add_figure("test_example", fig, self.global_step)
        plt.close()

        self.accuracy(y_hat, y)
        self.log("test_accuracy", self.accuracy, on_step=False, on_epoch=True)

        self.jaccard(y_hat, y)
        self.log("test_jaccard", self.jaccard, on_step=False, on_epoch=True)

        self.log("test_loss", test_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.lr * 0.1
        )

        scheduler = None

        if self.lr_method == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max" if self.control_on_realized else "min",
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
                    "monitor": (
                        "val_realized" if self.control_on_realized else "val_loss"
                    ),
                }
            ]

        return optimizer


__all__ = ["CITLSegmenter"]
