"""`velora models`: list engines, download or remove weights, list voices."""

from typing import Annotated

import typer
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TransferSpeedColumn,
)

from velora.engines.base import Engine, EngineError
from velora.engines.registry import engine_names, get_engine
from velora.models.render import engine_row, render_engines, render_voices

models_app = typer.Typer(help="List TTS engines and manage their downloaded weights.")

EngineName = Annotated[str, typer.Argument(help="Engine name, as listed by `velora models`.")]


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(1)


def _engine(name: str) -> Engine:
    try:
        return get_engine(name)
    except EngineError as exc:
        raise _fail(str(exc)) from exc


@models_app.callback(invoke_without_command=True)
def list_engines(ctx: typer.Context) -> None:
    """List engines with install and download status."""
    if ctx.invoked_subcommand is None:
        typer.echo(render_engines([engine_row(get_engine(n)) for n in engine_names()]))


@models_app.command()
def download(name: EngineName) -> None:
    """Download an engine's weights into the local cache."""
    engine = _engine(name)
    if engine.weights_status().downloaded:
        typer.echo(f"{name} weights are already downloaded.")
        return

    columns = (
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
    )
    with Progress(*columns) as bar:
        tasks: dict[str, TaskID] = {}

        def report(file: str, done: int, total: int) -> None:
            if file not in tasks:
                tasks[file] = bar.add_task(file, total=total)
            bar.update(tasks[file], completed=done)

        try:
            engine.download(report)
        except EngineError as exc:
            raise _fail(str(exc)) from exc
    typer.echo(f"{name} weights downloaded.")


@models_app.command()
def remove(name: EngineName) -> None:
    """Delete an engine's cached weights."""
    engine = _engine(name)
    had_weights = engine.weights_status().downloaded
    engine.remove_weights()
    typer.echo(f"{name} weights removed." if had_weights else f"{name} had no downloaded weights.")


@models_app.command()
def voices(name: EngineName) -> None:
    """List an engine's preset voices."""
    engine = _engine(name)
    if not engine.weights_status().downloaded:
        raise _fail(f"{name} weights are not downloaded. Run: velora models download {name}")
    typer.echo(render_voices(name, engine.voices()))
