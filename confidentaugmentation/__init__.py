import typer

cli = typer.Typer()

from .simpletrain import simpletrain
from .train import train
from .visualize import visualize

__all__ = ["cli", "train", "visualize"]
