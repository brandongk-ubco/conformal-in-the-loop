import sys

import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping
from monai.networks.nets import EfficientNetBN

from confidentaugmentation import cli
from confidentaugmentation.data import AugmentedCIFAR10DataModule

from .model.ConformalTrainer import ConformalTrainer


@cli.command()
def train(selectively_backpropagate: bool = False, mapie_alpha: float = 0.10):
    L.seed_everything(42, workers=True)

    # dm = MNISTDataModule()
    dm = AugmentedCIFAR10DataModule()

    net = EfficientNetBN(
        "efficientnet-b0", in_channels=dm.dims[0], num_classes=dm.num_classes
    )

    model = ConformalTrainer(
        net,
        num_classes=dm.num_classes,
        selectively_backpropagate=selectively_backpropagate,
        mapie_alpha=mapie_alpha,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=10)
    ]

    trainer = L.Trainer(
        num_sanity_val_steps=sys.maxsize,
        max_epochs=sys.maxsize,
        deterministic=True,
        callbacks=callbacks,
    )

    trainer.fit(model=model, datamodule=dm)
