from statistics import mean

import numpy as np
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
        pruning=False,
        mapie_alpha=0.10,
        val_mapie_alpha=0.10,
        warmup_epochs=3,
        lr=1e-3,
        lr_method="plateau",
        mapie_method="score",
        uncertainty_pruning_threshold=2,
        reclamation_interval=10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.conformal_classifier = ConformalClassifier()

        self.classes_ = range(num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.classes_))

        self.selectively_backpropagate = selectively_backpropagate
        self.mapie_alpha = mapie_alpha
        self.val_mapie_alpha = val_mapie_alpha
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.pixel_dropout = 0.0
        self.lr_method = lr_method
        self.weight_decay = 0.0
        self.mapie_method = mapie_method
        self.examples_without_uncertainty = {}
        self.uncertainty_pruning_threshold = uncertainty_pruning_threshold
        self.reclamation_interval = reclamation_interval
        self.pruning = pruning
        self.control_on_realized = selectively_backpropagate or pruning

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

    def training_step(self, batch, batch_idx):
        x, y, indeces = batch

        y_hat = self(x)

        if self.current_epoch == 0:
            img, target = x[1, :, :, :], y[1]
            img = img - img.min()
            img = img / img.max()
            label = self.trainer.datamodule.classes[target]
            if type(self.trainer.logger) is TensorBoardLogger:
                self.logger.experiment.add_image(f"{label}", img, self.global_step)
            elif type(self.trainer.logger) is NeptuneLogger:
                fig = plt.figure()
                if img.shape[0] == 1:
                    plt.imshow(img.detach().cpu().moveaxis(0, -1), cmap="gray")
                else:
                    plt.imshow(img.detach().cpu().moveaxis(0, -1))
                self.logger.experiment[f"training/examples/{label}"].append(fig)
                plt.close()

        self.conformal_classifier.cp_examples = list(
            zip(y_hat.detach().softmax(axis=1).cpu().numpy(), y.detach().cpu().numpy())
        )

        num_classes = (

        )

        predicted = y_hat.argmax(axis=1)

        correct = predicted == y

        atypical = torch.tensor(num_classes == 0).to(device=self.device)
        realized = torch.logical_and(
            correct, torch.tensor(num_classes == 1).to(device=self.device)
        )
        confused = torch.logical_and(
            ~correct, torch.tensor(num_classes == 1).to(device=self.device)
        )
        uncertain = torch.tensor(num_classes > 1).to(device=self.device)

        for i, idx in enumerate(indeces.detach().cpu().numpy()):
            if uncertain[i]:
                self.examples_without_uncertainty[idx] = 0
            else:
                if idx in self.examples_without_uncertainty:
                    self.examples_without_uncertainty[idx] += 1
                else:
                    self.examples_without_uncertainty[idx] = 1

        atypical_percentage = atypical.sum() / len(atypical)
        realized_percentage = realized.sum() / len(realized)
        confused_percentage = confused.sum() / len(confused)
        uncertain_percentage = uncertain.sum() / len(uncertain)

        torch.testing.assert_close(
            atypical_percentage
            + realized_percentage
            + confused_percentage
            + uncertain_percentage,
            torch.tensor(1.0).to(self.device),
        )

        metrics = {
            "atypical": atypical_percentage,
            "uncertain": uncertain_percentage,
            "confused": confused_percentage,
            "realized": realized_percentage,
        }
        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if self.selectively_backpropagate:
            loss = F.cross_entropy(y_hat, y, reduction="none")[
                torch.logical_or(uncertain, realized)
            ].mean()
        else:
            loss = F.cross_entropy(y_hat, y)

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

        if self.pruning:
            if self.current_epoch % self.reclamation_interval == 0:
                self.trainer.datamodule.reset_train_data()
            else:
                examples_to_prune = [
                    k
                    for k, v in self.examples_without_uncertainty.items()
                    if v >= self.uncertainty_pruning_threshold
                ]
                self.trainer.datamodule.remove_train_data(examples_to_prune)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        import pdb
        pdb.set_trace()

        y_hat = self(x)

        test_loss = F.cross_entropy(y_hat, y)

        self.conformal_classifier.append(y_hat, y)

        self.val_labels += list(y.detach().cpu().numpy())

        self.accuracy(y_hat, y)
        self.log("val_accuracy", self.accuracy, on_step=False, on_epoch=True)

        self.log("val_loss", test_loss, on_step=False, on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.accuracy.reset()
        self.cp_examples = []
        self.val_labels = []

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

        calib_idx = len(self.cp_examples) // 2

        conformal_sets = self.mapie_classifier.predict(
            range(calib_idx, len(self.cp_examples)), alpha=[self.val_mapie_alpha]
        )[1]

        num_classes = conformal_sets.sum(axis=1).squeeze()

        conformal_predictions = conformal_sets.argmax(axis=1).squeeze()

        correct = conformal_predictions == self.val_labels[calib_idx:]

        atypical = num_classes == 0
        realized = np.logical_and(correct, num_classes == 1)
        confused = np.logical_and(~correct, num_classes == 1)
        uncertain = num_classes > 1

        atypical_percentage = atypical.sum() / len(atypical)
        realized_percentage = realized.sum() / len(realized)
        confused_percentage = confused.sum() / len(confused)
        uncertain_percentage = uncertain.sum() / len(uncertain)

        metrics = {
            "val_atypical": atypical_percentage,
            "val_uncertain": uncertain_percentage,
            "val_confused": confused_percentage,
            "val_realized": realized_percentage,
        }
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        if self.lr_method == "uncertainty":
            self.optimizers().optimizer.param_groups[0]["lr"] = (
                self.lr * 0.9 * metrics["val_uncertain"] + self.lr * 0.1
            )

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        test_loss = F.cross_entropy(y_hat, y)

        self.cp_examples = list(
            zip(y_hat.detach().softmax(axis=1).cpu().numpy(), y.detach().cpu().numpy())
        )

        num_classes = (
            self.mapie_classifier.predict(
                range(len(self.cp_examples)), alpha=[self.val_mapie_alpha]
            )[1]
            .sum(axis=1)
            .squeeze()
        )

        predicted = y_hat.argmax(axis=1)
        correct = predicted == y

        atypical = torch.tensor(num_classes == 0).to(device=self.device)
        realized = torch.logical_and(
            correct, torch.tensor(num_classes == 1).to(device=self.device)
        )
        confused = torch.logical_and(
            ~correct, torch.tensor(num_classes == 1).to(device=self.device)
        )
        uncertain = torch.tensor(num_classes > 1).to(device=self.device)

        atypical_percentage = atypical.sum() / len(atypical)
        realized_percentage = realized.sum() / len(realized)
        confused_percentage = confused.sum() / len(confused)
        uncertain_percentage = uncertain.sum() / len(uncertain)

        metrics = {
            "test_atypical": atypical_percentage,
            "test_uncertain": uncertain_percentage,
            "test_confused": confused_percentage,
            "test_realized": realized_percentage,
        }

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


__all__ = ["ConformalTrainer"]