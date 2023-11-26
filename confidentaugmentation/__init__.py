import typer

cli = typer.Typer()

from .train import train
from .visualize import visualize

__all__ = ["cli"]
