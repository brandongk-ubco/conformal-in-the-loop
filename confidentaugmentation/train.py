import os
import sys

import pytorch_lightning as L
import torch
from monai.networks.nets import EfficientNetBN
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from confidentaugmentation import cli
from confidentaugmentation.data import AugmentedCIFAR10DataModule

from .model.ConformalTrainer import ConformalTrainer


@cli.command()
def train(
    augmentation_policy_path: str = "./policies/noop.yaml",
    selectively_backpropagate: bool = False,
    mapie_alpha: float = 0.10,
    model_name: str = "efficientnet-b0",
):
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    # dm = MNISTDataModule()
    dm = AugmentedCIFAR10DataModule(augmentation_policy_path)

    net = EfficientNetBN(
        model_name, in_channels=dm.dims[0], num_classes=dm.num_classes, pretrained=False
    )

    model = ConformalTrainer(
        net,
        num_classes=dm.num_classes,
        selectively_backpropagate=selectively_backpropagate,
        mapie_alpha=mapie_alpha,
    )

    callbacks = [
        # EarlyStopping(monitor="accuracy", mode="max", patience=10),
        LearningRateMonitor(logging_interval="step"),
    ]

    policy, _ = os.path.splitext(os.path.basename(augmentation_policy_path))

    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=f"{model_name}-{selectively_backpropagate}-{policy}-{mapie_alpha}",
        name="lightning_logs",
    )

    trainer = L.Trainer(
        logger=logger,
        num_sanity_val_steps=sys.maxsize,
        max_epochs=100,
        deterministic=True,
        callbacks=callbacks,
    )

    trainer.fit(model=model, datamodule=dm)
