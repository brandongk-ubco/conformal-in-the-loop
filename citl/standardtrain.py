import os
import sys

import pytorch_lightning as L
import segmentation_models_pytorch as smp
import torch
from loguru import logger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.tuner import Tuner
from timm import create_model
from torch import nn

from citl import cli

from .dataset import Dataset
from .model.Classifier import Classifier
from .model.Segmenter import Segmenter


@cli.command()
def standardtrain(
    dataset: Dataset,
    model_name: str,
    greyscale: bool = False,
    augmentation_policy_path: str = "./policies/noop.yaml",
    lr_method: str = "plateau",
    lr: float = 5e-4,
    noise_level: float = 0.0
):
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    assert os.path.exists(augmentation_policy_path)
    datamodule = Dataset.get(dataset)(augmentation_policy_path, noise_level=noise_level)

    if datamodule.task == "classification":
        net = create_model(
            model_name, num_classes=datamodule.num_classes, drop_rate=0.2
        )
    elif datamodule.task == "segmentation":
        net = smp.Unet(
            encoder_name=model_name,
            in_channels=3,
            classes=datamodule.num_classes,
        )

    if greyscale:
        net = nn.Sequential(nn.Conv2d(1, 3, 1), net)

    if datamodule.task == "classification":
        model = Classifier
    elif datamodule.task == "segmentation":
        model = Segmenter
    else:
        raise ValueError("Unknown task")

    model = model(
        net,
        num_classes=datamodule.num_classes,
        lr_method=lr_method,
        lr=lr,
    )

    policy, _ = os.path.splitext(os.path.basename(augmentation_policy_path))

    save_dir = os.path.join(
        "lightning_logs",
        "standard",
        lr_method,
    )

    trainer_logger = TensorBoardLogger(
        save_dir=save_dir,
        version=f"{model_name}-{policy}",
        name=dataset,
    )
    if os.environ.get("NEPTUNE_API_TOKEN"):
        trainer_logger = NeptuneLogger(
            project="conformal-in-the-loop/citl",
            name=f"{model_name}-{dataset}",
            api_key=os.environ["NEPTUNE_API_TOKEN"],
        )
        trainer_logger.experiment["parameters/architecture"] = model_name
        trainer_logger.experiment["parameters/dataset"] = dataset
        trainer_logger.experiment["parameters/greysacale"] = greyscale
        trainer_logger.experiment["parameters/augmentation_policy"] = policy
        trainer_logger.experiment["sys/tags"].add(model_name)
        trainer_logger.experiment["sys/tags"].add(dataset)
        trainer_logger.experiment["sys/tags"].add("Standard")

    model_callback_config = {
        "filename": "{epoch}-{val_loss:.3f}",
        "monitor": "val_loss",
        "mode": "min",
        "save_top_k": 1,
        "save_last": True,
    }

    if trainer_logger.log_dir:
        model_callback_config["dirpath"] = os.path.join(
            trainer_logger.log_dir, "checkpoints"
        )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(**model_callback_config),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=20,
        ),
    ]

    trainer = L.Trainer(
        logger=trainer_logger,
        num_sanity_val_steps=0,
        max_epochs=sys.maxsize,
        deterministic=True,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)

    if os.environ.get("NEPTUNE_API_TOKEN"):
        trainer_logger.experiment["sys/tags"].add("complete")
