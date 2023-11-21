import sys

import pytorch_lightning as L
from monai.networks.nets import EfficientNetBN

from confidentaugmentation import cli
from confidentaugmentation.data.MNist import MNISTDataModule

from .model.ConformalTrainer import ConformalTrainer


@cli.command()
def train():
    L.seed_everything(42, workers=True)

    dm = MNISTDataModule()

    net = EfficientNetBN(
        "efficientnet-b0", in_channels=dm.dims[0], num_classes=dm.num_classes
    )

    model = ConformalTrainer(net)

    trainer = L.Trainer(
        num_sanity_val_steps=sys.maxsize, max_epochs=20, deterministic=True
    )

    trainer.fit(model=model, datamodule=dm)
