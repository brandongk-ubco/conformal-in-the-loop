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
    use_pid: bool = False,
    Kp: float = 5e-3,
    lr_method: str = "plateau",
    lr: float = 1e-3,
    optimizer: str = "Adam",
    control_weight_decay: bool = False,
    control_pixel_dropout: bool = False,
):
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    if not selectively_backpropagate and mapie_alpha != 0.10:
        logger.info("Can't use MAPIE with backprop_all.")
        sys.exit(0)

    if use_pid:
        if not (control_weight_decay or control_pixel_dropout):
            logger.info("Can't use PID without control.")
            sys.exit(0)
        pid = PID(Kp, 1.0, output_limits=(0, 1))
    else:
        if control_weight_decay or control_pixel_dropout:
            logger.info("Can't use control without PID.")
            sys.exit(0)
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
        optimizer=optimizer,
        control_weight_decay=control_weight_decay,
        control_pixel_dropout=control_pixel_dropout,
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
        LearningRateMonitor(logging_interval="step"),
        model_checkpoint,
    ]

    if lr_method in ["plateau", "uncertainty"]:
        if not pretrained:
            logger.info("Only Use plateau or uncertainty with pretrained training.")
            sys.exit(0)
        callbacks.append(
            EarlyStopping(
                monitor="val_realized" if selectively_backpropagate else "val_loss",
                mode="max" if selectively_backpropagate else "min",
                patience=20,
            )
        )

    if lr_method == "one_cycle" and pretrained:
        logger.info("Only Use one_cycle with scratch training.")
        sys.exit(0)

    policy, _ = os.path.splitext(os.path.basename(augmentation_policy_path))

    save_dir = os.path.join(
        "lightning_logs",
        "backprop_uncertain" if selectively_backpropagate else "backprop_all",
        "pretrained" if pretrained else "scratch",
        "pid" if use_pid else "no_pid",
        "control_weight_decay" if control_weight_decay else "no_control_weight_decay",
        "control_pixel_dropout"
        if control_pixel_dropout
        else "no_control_pixel_dropout",
        lr_method,
        optimizer,
    )
    trainer_logger = TensorBoardLogger(
        save_dir=save_dir,
        version=f"{model_name}-{policy}-{mapie_alpha}",
        name=dataset,
    )

    trainer = L.Trainer(
        logger=trainer_logger,
        num_sanity_val_steps=sys.maxsize,
        max_epochs=100 if lr_method == "one_cycle" else sys.maxsize,
        deterministic=True,
        callbacks=callbacks,
        log_every_n_steps=10,
        accumulate_grad_batches=3,
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)
