import typer

from velora.inspect.command import inspect_book

app = typer.Typer(no_args_is_help=True)
app.command("inspect")(inspect_book)


@app.callback()
def _root() -> None:
    """Turn EPUB chapters into audio."""
