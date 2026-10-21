"""`velora inspect`: list a book's chapters or preview one."""

from pathlib import Path
from typing import Annotated

import typer

from velora.config import DEFAULT_WPM
from velora.inspect.render import render_book, render_chapter
from velora.readers.epub import EpubError, load_book


def inspect_book(
    book_path: Annotated[Path, typer.Argument(help="EPUB file to inspect.", metavar="BOOK")],
    wpm: Annotated[
        int, typer.Option(min=1, help="Words per minute for duration estimates.")
    ] = DEFAULT_WPM,
    show: Annotated[
        int | None,
        typer.Option(min=1, metavar="N", help="Preview the opening text of chapter N."),
    ] = None,
) -> None:
    """List a book's chapters with word counts and estimated minutes."""
    try:
        book = load_book(book_path)
    except EpubError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if show is None:
        typer.echo(render_book(book, wpm))
        return

    if show > len(book.chapters):
        raise typer.BadParameter(
            f"chapter {show} does not exist; the book has {len(book.chapters)}",
            param_hint="--show",
        )
    typer.echo(render_chapter(book.chapters[show - 1]))
