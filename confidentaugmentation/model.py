from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as L
from mapie.classification import MapieClassifier
import numpy as np


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

class LitAutoEncoder(L.LightningModule):
    def __init__(self, model, calib_set):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'calib_set'])
        self.model = model

        self.classes_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.mapie_classifier = MapieClassifier(estimator=self, method='score', cv="prefit")
        self.calib_set = calib_set
        self.mapie_classifier = self.mapie_classifier.fit([c[0] for c in self.calib_set], [c[1] for c in self.calib_set])

        # This will give you the number of classes in the prediction set on a 90% confidence interval.
        num_classes = self.mapie_classifier.predict(calib_set[0][0], alpha=[0.10])[1].sum()

        import pdb
        pdb.set_trace()


    def __sklearn_is_fitted__(self):
        return True
    
    def fit(self, x, y):
        raise NotImplementedError("Cannot fit this way.")

    def predict(self, x):
        with torch.no_grad():
            return self.predict_proba(x).argmax(axis=1)
    
    def predict_proba(self, x):
        with torch.no_grad():
            return np.array([self(x_n) for x_n in x])
             
    def forward(self, x):
        if x.dim() == 2:
            return self.model(x.unsqueeze(0).unsqueeze(0)).softmax(1).squeeze(0).squeeze(0)
        elif x.dim() == 3:
            return self.model(x.unsqueeze(0)).softmax(1).squeeze(0)
        elif x.dim() == 4:
            return self.model(x).softmax(1)
        else:
            import pdb
            pdb.set_trace()
            raise ValueError("Input must be 3 or 4 dimensional")

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = F.cross_entropy(x_hat, y)
        self.log("loss", loss)
        return loss
    
    def val_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        test_loss =  F.cross_entropy(x_hat, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def get_model(calib_set):
    return LitAutoEncoder(Net(), calib_set)

__all__ = ["get_model"]