import os
import sys

import pytorch_lightning as L
import torch
from loguru import logger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from timm import create_model

from confidentaugmentation import cli
from confidentaugmentation.control import PID
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
    max_epochs: int = sys.maxsize,
    use_pid: bool = False,
    Kp: float = 5e-3,
    lr_method: str = "plateau",
    lr: float = 1e-3,
    optimizer: str = "Adam"
):
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    if not selectively_backpropagate and mapie_alpha != 0.10:
        logger.info("Can't use MAPIE with backprop_all.")
        sys.exit(0)

    if use_pid:
        pid = PID(Kp, 1.0, output_limits=(0, 0.1))
    else:
        pid = None

    # dm = MNISTDataModule()
    if dataset == "cifar10":
        dm = AugmentedCIFAR10DataModule(augmentation_policy_path)
    else:
        raise NotImplementedError("Dataset not implemented.")

    model = ConformalTrainer(
        create_model(model_name, pretrained=pretrained, num_classes=len(dm.classes)),
        num_classes=dm.num_classes,
        selectively_backpropagate=selectively_backpropagate,
        mapie_alpha=mapie_alpha,
        pid=pid,
        lr_method=lr_method,
        lr=lr,
        optimizer=optimizer
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
        "pid" if use_pid else "no_pid",
        lr_method,
        optimizer
    )
    trainer_logger = TensorBoardLogger(
        save_dir=save_dir,
        version=f"{model_name}-{policy}-{mapie_alpha}",
        name=dataset,
    )

    trainer = L.Trainer(
        logger=trainer_logger,
        num_sanity_val_steps=sys.maxsize,
        max_epochs=max_epochs,
        deterministic=True,
        callbacks=callbacks,
        log_every_n_steps=10,
        accumulate_grad_batches=3,
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)
