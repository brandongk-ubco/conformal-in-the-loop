import typer

cli = typer.Typer()

from .train import train
from .visualize import visualize
from .simpletrain import simpletrain

__all__ = ["cli", "train", "visualize"]
