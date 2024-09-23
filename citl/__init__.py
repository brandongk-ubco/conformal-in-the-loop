import typer

cli = typer.Typer()

from .standardtrain import standardtrain
from .train import train
from .visualize import visualize
from .test import test

__all__ = ["cli", "train", "standardtrain", "visualize", "test"]
