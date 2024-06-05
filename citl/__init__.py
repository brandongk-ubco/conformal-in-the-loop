import typer

cli = typer.Typer()

from .standardtrain import standardtrain
from .train import train
from .visualize import visualize

__all__ = ["cli", "train", "standardtrain", "visualize"]
