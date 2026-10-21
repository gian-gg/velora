import typer

from velora.inspect.command import inspect_book
from velora.models.command import models_app

app = typer.Typer(no_args_is_help=True)
app.command("inspect")(inspect_book)
app.add_typer(models_app, name="models")


@app.callback()
def _root() -> None:
    """Turn EPUB chapters into audio."""
