import numpy as np
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from mapie.classification import MapieClassifier
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch import nn
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 3
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # Second 2D convolutional layer, taking in the 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        # First fully connected layer
        self.fc1 = nn.Linear(9216, 128)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # Run max pooling over x
        x = F.max_pool2d(x, 2)
        # Pass data through dropout1
        x = self.dropout1(x)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through ``fc1``
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class ConformalTrainer(L.LightningModule):
    def __init__(self, model, selectively_backpropagate=False):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.classes_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.accuracy =  Accuracy(task="multiclass", num_classes=len(self.classes_))

        self.selectively_backpropagate = selectively_backpropagate

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
            return (
                self.model(x.unsqueeze(0).unsqueeze(0))
            )
        elif x.dim() == 3:
            return self.model(x.unsqueeze(0))
        elif x.dim() == 4:
            return self.model(x)
        else:
            import pdb

            pdb.set_trace()
            raise ValueError("Input must be 2, 3 or 4 dimensional")

    def on_train_epoch_start(self) -> None:
        self.uncertain_percentage = MeanMetric().to(self.device)
        self.confused_percentage = MeanMetric().to(self.device)
        self.realized_percentage = MeanMetric().to(self.device)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.cp_examples = list(
            zip(y_hat.detach().softmax(axis=1).cpu().numpy(), y.detach().cpu().numpy())
        )

        num_classes = (
            self.mapie_classifier.predict(range(len(self.cp_examples)), alpha=[0.10])[1]
            .sum(axis=1)
            .squeeze()
        )

        predicted = y_hat.argmax(axis=1)
        correct = predicted == y

        realized = torch.logical_and(correct, torch.tensor(num_classes == 1).to(device=self.device))
        confused = torch.logical_and(~correct, torch.tensor(num_classes == 1).to(device=self.device))
        uncertain = torch.tensor(num_classes != 1).to(device=self.device)

        realized_percentage = realized.sum() / len(realized)
        confused_percentage = confused.sum() / len(confused)
        uncertain_percentage = uncertain.sum() / len(uncertain)

        self.realized_percentage(realized_percentage)
        self.confused_percentage(confused_percentage)
        self.uncertain_percentage(uncertain_percentage)

        torch.testing.assert_close(realized_percentage + confused_percentage + uncertain_percentage, torch.tensor(1.0).to(self.device))

        metrics = {
            "uncertain": self.uncertain_percentage.compute(),
            "confused": self.confused_percentage.compute(),
            "realized": self.realized_percentage.compute(),
        }

        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=True, logger=True)


        if self.selectively_backpropagate:
            loss = F.cross_entropy(y_hat, y, reduction='none')[uncertain].mean()
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

        self.log('accuracy', self.accuracy, on_step=False, on_epoch=True)

        self.log("val_loss", test_loss, on_step=False, on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        self.cp_examples = []

    def on_validation_epoch_end(self) -> None:
        self.mapie_classifier = MapieClassifier(
            estimator=self, method="score", cv="prefit"
        ).fit(np.array(range(len(self.cp_examples))), [v[1] for v in self.cp_examples])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def get_model():
    return ConformalTrainer(Net())


__all__ = ["get_model"]
