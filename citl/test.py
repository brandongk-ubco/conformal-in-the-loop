import logging
import os

import pytorch_lightning as L
import segmentation_models_pytorch as smp
import torch
from loguru import logger
from pytorch_lightning.loggers import TensorBoardLogger
from timm import create_model

from citl import cli

from .dataset import Dataset
from .model.CITLClassifier import CITLClassifier
from .model.CITLSegmenter import CITLSegmenter

logging.getLogger("neptune").setLevel(logging.CRITICAL)


@cli.command()
def test(
    dataset: Dataset,
    model_name: str,
    augmentation_policy_path: str = "./policies/noop.yaml",
    checkpoint: str = None,
    alpha: float = 0.10,
    quantile: float = 0.95,
):
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    assert os.path.exists(augmentation_policy_path)
    datamodule = Dataset.get(dataset)(augmentation_policy_path)

    if datamodule.task == "classification":
        net = create_model(
            model_name,
            num_classes=datamodule.num_classes,
            drop_rate=0.2,
        )
    elif datamodule.task == "segmentation":
        net = smp.Unet(
            encoder_name=model_name,
            in_channels=3,
            classes=datamodule.num_classes,
        )

    if datamodule.task == "classification":
        model = CITLClassifier
    elif datamodule.task == "segmentation":
        model = CITLSegmenter
    else:
        raise ValueError("Unknown task")

    save_dir = os.path.join("lightning_logs")

    trainer_logger = TensorBoardLogger(
        save_dir=save_dir,
        version=f"{checkpoint}",
        name=dataset,
    )

    trainer = L.Trainer(
        logger=trainer_logger,
    )

    model = model.load_from_checkpoint(
        checkpoint, model=net, num_classes=datamodule.num_classes, alpha=alpha
    )
    datamodule.setup()
    trainer.validate(model=model, datamodule=datamodule)
    logger.info(f"Found quantile: {model.conformal_classifier.quantiles}")

    trainer.test(model=model, datamodule=datamodule)
