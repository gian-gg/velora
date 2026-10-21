import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def hello(name: str = typer.Argument("world", help="Who to greet.")) -> None:
    """Print a greeting."""
    typer.echo(f"Hello, {name}!")


@app.callback()
def _root() -> None:
    """Turn EPUB and PDF chapters into audio."""
