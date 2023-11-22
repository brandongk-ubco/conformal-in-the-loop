import numpy as np
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from mapie.classification import MapieClassifier
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class ConformalTrainer(L.LightningModule):
    def __init__(
        self, model, num_classes, selectively_backpropagate=False, mapie_alpha=0.10
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.classes_ = range(num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.classes_))

        self.atypical_percentage = MeanMetric()
        self.uncertain_percentage = MeanMetric()
        self.confused_percentage = MeanMetric()
        self.realized_percentage = MeanMetric()

        self.selectively_backpropagate = selectively_backpropagate
        self.mapie_alpha = mapie_alpha

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, x, y):
        raise NotImplementedError("Cannot fit this way.")

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def predict_proba(self, x):
        return np.array([v[0] for v in self.cp_examples])[x]

    def forward(self, x):
        if x.dim() == 2:
            return self.model(x.unsqueeze(0).unsqueeze(0))
        elif x.dim() == 3:
            return self.model(x.unsqueeze(0))
        elif x.dim() == 4:
            return self.model(x)
        else:
            import pdb

            pdb.set_trace()
            raise ValueError("Input must be 2, 3 or 4 dimensional")

    def on_train_epoch_start(self) -> None:
        self.atypical_percentage.reset()
        self.uncertain_percentage.reset()
        self.confused_percentage.reset()
        self.realized_percentage.reset()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.cp_examples = list(
            zip(y_hat.detach().softmax(axis=1).cpu().numpy(), y.detach().cpu().numpy())
        )

        num_classes = (
            self.mapie_classifier.predict(
                range(len(self.cp_examples)), alpha=[self.mapie_alpha]
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

        self.atypical_percentage(atypical_percentage)
        self.realized_percentage(realized_percentage)
        self.confused_percentage(confused_percentage)
        self.uncertain_percentage(uncertain_percentage)

        torch.testing.assert_close(
            atypical_percentage + realized_percentage + confused_percentage + uncertain_percentage,
            torch.tensor(1.0).to(self.device),
        )

        metrics = {
            "atypical": self.atypical_percentage.compute(),
            "uncertain": self.uncertain_percentage.compute(),
            "confused": self.confused_percentage.compute(),
            "realized": self.realized_percentage.compute(),
        }

        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if self.selectively_backpropagate:
            loss = F.cross_entropy(y_hat, y, reduction="none")[uncertain].mean()
        else:
            loss = F.cross_entropy(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)

        self.cp_examples += list(
            zip(y_hat.detach().softmax(axis=1).cpu().numpy(), y.detach().cpu().numpy())
        )

        self.accuracy(y_hat, y)

        self.log("accuracy", self.accuracy, on_step=False, on_epoch=True)

        self.log("val_loss", test_loss, on_step=False, on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        self.accuracy.reset()
        self.cp_examples = []

    def on_validation_epoch_end(self) -> None:
        self.mapie_classifier = MapieClassifier(
            estimator=self, method="score", cv="prefit"
        ).fit(np.array(range(len(self.cp_examples))), [v[1] for v in self.cp_examples])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


__all__ = ["get_model"]
