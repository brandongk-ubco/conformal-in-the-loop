import typer

cli = typer.Typer()

from .train import train

__all__ = ["cli"]
