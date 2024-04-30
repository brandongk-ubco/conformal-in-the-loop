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

from ..ConformalClassifier import ConformalClassifier


class CITLClassifier(L.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        selectively_backpropagate=False,
        control_on_realized=False,
        mapie_alpha=0.10,
        val_mapie_alpha=0.10,
        lr=1e-3,
        lr_method="plateau",
        mapie_method="score",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.conformal_classifier = ConformalClassifier(mapie_method=mapie_method)

        self.num_classes = num_classes

        self.accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
        )

        self.selectively_backpropagate = selectively_backpropagate
        self.control_on_realized = control_on_realized
        self.mapie_alpha = mapie_alpha
        self.val_mapie_alpha = val_mapie_alpha
        self.lr = lr
        self.pixel_dropout = 0.0
        self.lr_method = lr_method
        self.weight_decay = 0.0
        self.mapie_method = mapie_method
        self.examples_without_uncertainty = {}

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
        self.conformal_classifier.reset()

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
        x, y, indeces = batch

        y_hat = self(x)

        if type(self.trainer.logger) is TensorBoardLogger and self.current_epoch == 0:
            img, target = x[1, :, :, :], y[1]
            img = img - img.min()
            img = img / img.max()
            label = self.trainer.datamodule.classes[target]
            self.logger.experiment.add_image(f"{label}", img, self.global_step)

        self.conformal_classifier.reset()
        self.conformal_classifier.append(y_hat, y)
        _, uncertainty = self.conformal_classifier.measure_uncertainty(
            alphas=[self.mapie_alpha]
        )

        metrics = dict([(k, v.mean()) for k, v in uncertainty.items()])
        self.log_dict(
            metrics, on_step=True, on_epoch=False, prog_bar=False, logger=True
        )

        uncertain = torch.tensor(uncertainty["uncertain"]).to(device=self.device)

        for i, idx in enumerate(indeces.detach().cpu().numpy()):
            if uncertain[i]:
                self.examples_without_uncertainty[idx] = 0
            else:
                if idx in self.examples_without_uncertainty:
                    self.examples_without_uncertainty[idx] += 1
                else:
                    self.examples_without_uncertainty[idx] = 1

        if self.selectively_backpropagate:
            prediction_set_size = torch.tensor(uncertainty["prediction_set_size"]).to(
                device=self.device
            )
            loss = F.cross_entropy(y_hat, y, reduction="none")
            loss = loss * prediction_set_size

            y_flt = y.flatten()
            p_flt = prediction_set_size.flatten()
            for clazz in range(self.num_classes):
                class_idxs = y_flt == clazz
                count = class_idxs.sum()
                weights = p_flt[class_idxs].sum()
                self.class_counts[clazz] += count
                self.class_weights[clazz] += weights

            loss = loss.mean()
        else:
            self.has_backpropped = True
            loss = F.cross_entropy(y_hat, y, reduction="none").mean()

        self.log("loss", loss)
        return loss

    def on_train_epoch_end(self) -> None:
        num_bins = max(21, self.current_epoch + 1)
        bins = np.arange(0, num_bins)
        plt.figure()
        sns_plot = sns.histplot(
            data=self.examples_without_uncertainty,
            stat="percent",
            bins=bins,
        )

        sns_plot.set_title(
            f"Epochs without uncertainty for training examples (epoch: {self.current_epoch + 1})"
        )
        sns_plot.set_xlabel(
            f"Number of epochs without uncertainty (mean: {mean([v for k,v in self.examples_without_uncertainty.items()]):.2f})"
        )
        sns_plot.set_xticks(bins[:-1] + 0.5)
        sns_plot.set_xticklabels(bins[:-1], rotation=90)
        sns_plot.set_ylabel("Percentage of examples")
        sns_plot.set_ylim(0, 100)

        if type(self.trainer.logger) is TensorBoardLogger:
            self.logger.experiment.add_figure(
                "examples_without_uncertainty",
                sns_plot.get_figure(),
                self.global_step,
            )
        elif type(self.trainer.logger) is NeptuneLogger:
            self.logger.experiment["training/examples_without_uncertainty"].append(
                sns_plot.get_figure()
            )
        plt.close()

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

        weight_range = max(weights.values()) - min(weights.values())
        self.log(
            "weight_range", weight_range, on_step=False, on_epoch=True, logger=True
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
        self.accuracy.reset()
        self.conformal_classifier.reset()
        self.val_batch_idx_fit_uncertainty = (
            len(self.trainer.datamodule.val_dataloader()) // 5
        )

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        val_loss = F.cross_entropy(y_hat, y)

        if batch_idx < self.val_batch_idx_fit_uncertainty:
            self.conformal_classifier.append(y_hat, y)
        elif batch_idx == self.val_batch_idx_fit_uncertainty:
            self.conformal_classifier.fit()
        else:
            self.conformal_classifier.append(y_hat, y)
            _, uncertainty = self.conformal_classifier.measure_uncertainty(
                alphas=[self.val_mapie_alpha]
            )

            metrics = dict([(f"val_{k}", v.mean()) for k, v in uncertainty.items()])
            self.log_dict(metrics, prog_bar=True)

        self.accuracy(y_hat, y)
        self.log("val_accuracy", self.accuracy, prog_bar=True)

        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        test_loss = F.cross_entropy(y_hat, y)

        self.conformal_classifier.append(y_hat, y)
        _, uncertainty = self.conformal_classifier.measure_uncertainty(
            alphas=[self.val_mapie_alpha]
        )

        metrics = dict([(f"test_{k}", v.mean()) for k, v in uncertainty.items()])
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        self.accuracy(y_hat, y)
        self.log("test_accuracy", self.accuracy, on_step=False, on_epoch=True)

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


__all__ = ["CITLClassifier"]
