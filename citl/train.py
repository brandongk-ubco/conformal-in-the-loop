import itertools
import json
import os
import shutil
import sys
import tempfile

import pytorch_lightning as L
import segmentation_models_pytorch as smp
import torch
from loguru import logger
from neptune.types import File
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
from .model.CITLClassifier import CITLClassifier
from .model.CITLSegmenter import CITLSegmenter


@cli.command()
def train(
    dataset: Dataset,
    model_name: str,
    image_size: int = None,
    greyscale: bool = False,
    augmentation_policy_path: str = "./policies/noop.yaml",
    selectively_backpropagate: bool = False,
    control_on_realized: bool = False,
    alpha: float = 0.10,
    lr_method: str = "plateau",
    lr: float = 5e-4,
    method="score",
):
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    if not selectively_backpropagate and alpha != 0.10:
        logger.info("Can't use Conformal Prediction with backprop_all.")
        sys.exit(0)

    assert os.path.exists(augmentation_policy_path)
    datamodule = Dataset.get(dataset)(augmentation_policy_path)

    if datamodule.task == "classification":
        net = create_model(
            model_name, num_classes=datamodule.num_classes, drop_rate=0.2
        )
        datamodule.set_image_size(image_size, greyscale)
    elif datamodule.task == "segmentation":
        net = smp.Unet(
            encoder_name=model_name,
            in_channels=3,
            classes=datamodule.num_classes,
        )

    if greyscale:
        net = nn.Sequential(nn.Conv2d(1, 3, 1), net)

    if datamodule.task == "classification":
        model = CITLClassifier
    elif datamodule.task == "segmentation":
        model = CITLSegmenter
    else:
        raise ValueError("Unknown task")

    model = model(
        net,
        num_classes=datamodule.num_classes,
        selectively_backpropagate=selectively_backpropagate,
        alpha=alpha,
        lr_method=lr_method,
        lr=lr,
        method=method,
    )

    policy, _ = os.path.splitext(os.path.basename(augmentation_policy_path))

    save_dir = os.path.join(
        "lightning_logs",
        "backprop_uncertain" if selectively_backpropagate else "backprop_all",
        "control_on_realized" if control_on_realized else "control_on_loss",
        lr_method,
        method,
    )

    trainer_logger = TensorBoardLogger(
        save_dir=save_dir,
        version=f"{model_name}-{policy}-{alpha}",
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
        trainer_logger.experiment["parameters/image_size"] = image_size
        trainer_logger.experiment["parameters/greysacale"] = greyscale
        trainer_logger.experiment["parameters/augmentation_policy"] = policy
        trainer_logger.experiment["sys/tags"].add(model_name)
        trainer_logger.experiment["sys/tags"].add(dataset)

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

    if control_on_realized:
        model_callback_config["monitor"] = "val_realized"
        model_callback_config["mode"] = "max"

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(**model_callback_config),
        EarlyStopping(
            monitor="val_realized" if control_on_realized else "val_loss",
            mode="max" if control_on_realized else "min",
            patience=20,
        ),
    ]

    trainer = L.Trainer(
        logger=trainer_logger,
        num_sanity_val_steps=sys.maxsize,
        max_epochs=sys.maxsize,
        deterministic=True,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, datamodule=datamodule, max_trials=7)
    tuner.lr_find(model, datamodule=datamodule, max_lr=1e-2)

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)

    quantiles = model.conformal_classifier.quantiles
    quantiles = {k: v.detach().cpu().numpy().tolist() for k, v in quantiles.items()}
    quantiles_json = json.dumps(quantiles)
    if trainer_logger.log_dir:
        with open(os.path.join(trainer_logger.log_dir, "quantiles.json"), "w") as fh:
            fh.write(quantiles_json)

    if type(trainer_logger) is L.loggers.NeptuneLogger:
        trainer_logger.experiment["quantiles.json"].upload(
            File.from_content(quantiles_json)
        )

    x, _, _ = next(itertools.islice(datamodule.train_dataloader(), 1, None))
    input_sample = x[0]
    with tempfile.NamedTemporaryFile() as tmp:
        model.float().to_onnx(tmp.name, input_sample, export_params=True)

        if trainer_logger.log_dir:
            shutil.copy(tmp.name, os.path.join(trainer_logger.log_dir, "model.onnx"))

        if type(trainer_logger) is L.loggers.NeptuneLogger:
            trainer_logger.experiment["model.onnx"].upload(tmp.name)
            trainer_logger.experiment.sync(wait=True)
