import os
import sys

import pytorch_lightning as L
import torch
from monai.networks.nets import EfficientNetBN
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from confidentaugmentation import cli
from confidentaugmentation.data import AugmentedCIFAR10DataModule

from .model.ConformalTrainer import ConformalTrainer


@cli.command()
def train(
    dataset: str,
    augmentation_policy_path: str = "./policies/noop.yaml",
    selectively_backpropagate: bool = False,
    mapie_alpha: float = 0.10,
    model_name: str = "efficientnet-b0",
    pretrained: bool = False,
):
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    # dm = MNISTDataModule()
    if dataset == "cifar10":
        dm = AugmentedCIFAR10DataModule(augmentation_policy_path)
    else:
        raise NotImplementedError("Dataset not implemented.")

    net = EfficientNetBN(
        model_name,
        in_channels=dm.dims[0],
        num_classes=dm.num_classes,
        pretrained=pretrained,
    )

    model = ConformalTrainer(
        net,
        num_classes=dm.num_classes,
        selectively_backpropagate=selectively_backpropagate,
        mapie_alpha=mapie_alpha,
    )

    if selectively_backpropagate:
        model_checkpoint = ModelCheckpoint(
            filename="{epoch}-{val_realized:.3f}",
            monitor="val_realized",
            mode="max",
            save_top_k=1,
            save_last=True,
        )
    else:
        model_checkpoint = ModelCheckpoint(
            filename="{epoch}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )

    callbacks = [
        EarlyStopping(
            monitor="val_realized" if selectively_backpropagate else "val_loss",
            mode="max" if selectively_backpropagate else "min",
            patience=20,
        ),
        LearningRateMonitor(logging_interval="step"),
        model_checkpoint,
    ]

    policy, _ = os.path.splitext(os.path.basename(augmentation_policy_path))

    save_dir = os.path.join(
        "lightning_logs",
        "backprop_uncertain" if selectively_backpropagate else "backprop_all",
        "pretrained" if pretrained else "scratch",
    )
    logger = TensorBoardLogger(
        save_dir=save_dir,
        version=f"{model_name}-{selectively_backpropagate}-{policy}-{mapie_alpha}",
        name=dataset,
    )

    trainer = L.Trainer(
        logger=logger,
        num_sanity_val_steps=sys.maxsize,
        max_epochs=sys.maxsize,
        deterministic=True,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)
