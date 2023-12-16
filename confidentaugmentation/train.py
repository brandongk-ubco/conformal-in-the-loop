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
from confidentaugmentation.data import AugmentedCIFAR10DataModule, AugmentedImageNetDataModule, AugmentedMNISTDataModule

from .model.ConformalTrainer import ConformalTrainer
from .net import SimpleNet, MicroNet

@cli.command()
def train(
    dataset: str,
    augmentation_policy_path: str = "./policies/noop.yaml",
    selectively_backpropagate: bool = False,
    mapie_alpha: float = 0.10,
    model_name: str = "efficientnet_b0",
    pretrained: bool = False,
    Kp: float = 1e-4,
    lr_method: str = "plateau",
    lr: float = 5e-4,
    optimizer: str = "Adam",
    control_weight_decay: bool = False,
    control_pixel_dropout: bool = False,
    mapie_method="score",
):
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    if not selectively_backpropagate and mapie_alpha != 0.10:
        logger.info("Can't use MAPIE with backprop_all.")
        sys.exit(0)

    use_pid = control_weight_decay or control_pixel_dropout

    pid = None
    if use_pid:
        pid = PID(Kp, 0.5, initial_value=0.30, output_limits=(0.1, 0.5))

    # dm = MNISTDataModule()
    if dataset == "cifar10":
        dm = AugmentedCIFAR10DataModule(augmentation_policy_path)
    elif dataset == "imagenet":
        dm = AugmentedImageNetDataModule(augmentation_policy_path)
    elif dataset == "mnist":
        dm = AugmentedMNISTDataModule(augmentation_policy_path)
    else:
        raise NotImplementedError("Dataset not implemented.")

    if model_name == "SimpleNet":
        if pretrained:
            logger.info("SimpleNet does not support pretrained.")
            sys.exit(0)
        net = SimpleNet(num_classes=dm.num_classes)
    elif model_name == "MicroNet":
        if pretrained:
            logger.info("MicroNet does not support pretrained.")
            sys.exit(0)
        net = MicroNet(num_classes=dm.num_classes)
    else:
        net = create_model(model_name, pretrained=pretrained, num_classes=dm.num_classes)

    model = ConformalTrainer(
        net,
        num_classes=dm.num_classes,
        selectively_backpropagate=selectively_backpropagate,
        mapie_alpha=mapie_alpha,
        pid=pid,
        lr_method=lr_method,
        lr=lr,
        optimizer=optimizer,
        control_weight_decay=control_weight_decay,
        control_pixel_dropout=control_pixel_dropout,
        mapie_method=mapie_method,
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
        # if not pretrained:
        #     logger.info("Only Use plateau or uncertainty with pretrained training.")
        #     sys.exit(0)
        callbacks.append(
            EarlyStopping(
                monitor="val_realized" if selectively_backpropagate else "val_loss",
                mode="max" if selectively_backpropagate else "min",
                patience=20,
            )
        )

    # if lr_method == "one_cycle" and pretrained:
    #     logger.info("Only Use one_cycle with scratch training.")
    #     sys.exit(0)

    policy, _ = os.path.splitext(os.path.basename(augmentation_policy_path))

    save_dir = os.path.join(
        "lightning_logs",
        "backprop_uncertain" if selectively_backpropagate else "backprop_all",
        "pretrained" if pretrained else "scratch",
        lr_method,
        optimizer,
        mapie_method,
        "pid" if use_pid else "no_pid",
        "control_weight_decay" if control_weight_decay else "no_control_weight_decay",
        "control_pixel_dropout"
        if control_pixel_dropout
        else "no_control_pixel_dropout",
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
        # accumulate_grad_batches=3,
        # limit_train_batches=100,
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)
