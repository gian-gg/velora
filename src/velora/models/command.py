"""`velora models`: list engines, download or remove weights, list voices."""

from collections.abc import Mapping
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

from velora.config import config_file
from velora.engines.base import Engine, EngineError
from velora.engines.registry import engine_names, get_engine
from velora.engines.settings import Setting, find_setting
from velora.models.render import (
    engine_row,
    render_engines,
    render_settings,
    render_voices,
    short_path,
    show_value,
)
from velora.read.settings import READ_SETTINGS
from velora.user_config import ConfigError, UserConfig, load_config, save_config, with_value

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


config_app = typer.Typer(help="Show or change an engine's saved settings.")
models_app.add_typer(config_app, name="config")


def _groups(
    engine: Engine, config: UserConfig
) -> list[tuple[str, tuple[Setting, ...], Mapping[str, object]]]:
    return [
        (engine.name, engine.settings, config.engine(engine.name)),
        ("all engines", READ_SETTINGS, config.read),
    ]


@config_app.callback(invoke_without_command=True)
def config(ctx: typer.Context, name: EngineName) -> None:
    """Show every setting with its current value, default and allowed values."""
    engine = _engine(name)
    ctx.obj = engine
    if ctx.invoked_subcommand is None:
        typer.echo(render_settings(_groups(engine, _load()), config_file()))


def _load() -> UserConfig:
    try:
        return load_config()
    except ConfigError as exc:
        raise _fail(str(exc)) from exc


def _section(engine: Engine, key: str) -> tuple[str, Setting]:
    for section, settings in ((engine.name, engine.settings), ("read", READ_SETTINGS)):
        setting = find_setting(settings, key)
        if setting is not None:
            return section, setting
    names = [s.name for s in (*engine.settings, *READ_SETTINGS)]
    raise _fail(f"unknown setting {key!r}; available: {', '.join(names)}")


@config_app.command("set")
def set_value(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument(help="Setting name.")],
    value: Annotated[str, typer.Argument(help="New value.")],
) -> None:
    """Save a setting, validated by the engine."""
    engine: Engine = ctx.obj
    section, setting = _section(engine, key)
    try:
        parsed = setting.coerce(value)
        if section == engine.name:
            engine.check_voice({key: parsed})
    except EngineError as exc:
        raise _fail(str(exc)) from exc
    path = save_config(with_value(_load(), section, key, parsed))
    typer.echo(f"{engine.name} {key} = {value} (saved to {short_path(path)})")


@config_app.command("unset")
def unset_value(
    ctx: typer.Context, key: Annotated[str, typer.Argument(help="Setting name.")]
) -> None:
    """Remove a saved setting so its default applies."""
    engine: Engine = ctx.obj
    section, setting = _section(engine, key)
    path = save_config(with_value(_load(), section, key, None))
    typer.echo(
        f"{engine.name} {key} reset to default {show_value(setting.default)}"
        f" (saved to {short_path(path)})"
    )
