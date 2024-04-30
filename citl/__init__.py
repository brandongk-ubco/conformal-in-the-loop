import typer

cli = typer.Typer()

from .train import train
from .visualize import visualize
from .standardtrain import standardtrain

__all__ = ["cli", "train", "standardtrain", "visualize"]
