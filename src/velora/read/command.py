"""`velora read`: synthesize chapters of a book into tagged audio files."""

import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from velora.audio import AudioError, Tags, encode, with_pauses
from velora.book import Book, Chapter
from velora.config import DEFAULT_ENGINE, DEFAULT_OUTPUT_SUFFIX
from velora.engines.base import Engine, EngineError, ensure_ready
from velora.engines.registry import get_engine
from velora.models.render import short_path
from velora.read.pipeline import chapter_chunks, synthesize
from velora.read.selection import SelectionError, default_name, select_chapters
from velora.readers.epub import EpubError, load_book


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(1)


def _clock(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _render(
    engine: Engine, book: Book, chapters: tuple[Chapter, ...], voice: dict[str, Any], out: Path
) -> None:
    chunks = [c for chapter in chapters for c in chapter_chunks(chapter, engine.max_chunk_chars)]
    first, last = chapters[0], chapters[-1]
    title = first.title if first is last else f"{first.title} to {last.title}"
    tags = Tags(title=title, album=book.title, artist=book.author, track=first.index)

    columns = (
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("left"),
        TimeRemainingColumn(),
    )
    started = time.monotonic()
    with Progress(*columns) as bar:
        task = bar.add_task(out.name, total=sum(len(c.text) for c in chunks))
        segments = synthesize(engine, chunks, voice, lambda c: bar.advance(task, len(c.text)))
        seconds = encode(with_pauses(segments, engine.sample_rate), engine.sample_rate, out, tags)
    elapsed = time.monotonic() - started
    typer.echo(
        f"wrote {short_path(out)}  {_clock(seconds)} of audio in {_clock(elapsed)}"
        f" ({elapsed / max(seconds, 1e-9):.2f}x real time)"
    )


def _jobs(
    book_path: Path, selected: tuple[Chapter, ...], output: Path | None, split: bool
) -> list[tuple[tuple[Chapter, ...], Path]]:
    """Pair each output file with the chapters it holds."""
    if split:
        folder = output or Path.cwd()
        folder.mkdir(parents=True, exist_ok=True)
        return [
            ((c,), folder / default_name(book_path, (c,), DEFAULT_OUTPUT_SUFFIX)) for c in selected
        ]
    out = output or Path.cwd() / default_name(book_path, selected, DEFAULT_OUTPUT_SUFFIX)
    if not out.parent.is_dir():
        raise _fail(f"output folder does not exist: {out.parent}")
    return [(selected, out)]


def read_book(  # noqa: PLR0913, PLR0917
    book_path: Annotated[Path, typer.Argument(help="EPUB file to read.", metavar="BOOK")],
    chapter: Annotated[
        int | None, typer.Option(metavar="N", help="Chapter number, as shown by velora inspect.")
    ] = None,
    chapters: Annotated[
        str | None, typer.Option(metavar="A-B", help="Range of chapters, e.g. 3-5.")
    ] = None,
    engine_name: Annotated[str, typer.Option("--engine", help="TTS engine.")] = DEFAULT_ENGINE,
    voice: Annotated[
        str | None, typer.Option(help="Voice name or blend, e.g. af_heart:0.6,bf_emma:0.4.")
    ] = None,
    speed: Annotated[float | None, typer.Option(help="Speaking speed, 1.0 is normal.")] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Output file (.m4a, .mp3, .wav), or a folder with --split."
        ),
    ] = None,
    split: Annotated[
        bool,
        typer.Option("--split", help="Write one file per chapter instead of one combined file."),
    ] = False,
) -> None:
    """Synthesize chapters into an audio file."""
    try:
        book = load_book(book_path)
        selected = select_chapters(book, chapter, chapters)
        engine = get_engine(engine_name)
        ensure_ready(engine)
    except (EpubError, SelectionError, EngineError) as exc:
        raise _fail(str(exc)) from exc

    settings: dict[str, Any] = {
        k: v for k, v in {"voice": voice, "speed": speed}.items() if v is not None
    }
    try:
        for job_chapters, out in _jobs(book_path, selected, output, split):
            _render(engine, book, job_chapters, settings, out)
    except (EngineError, AudioError) as exc:
        raise _fail(str(exc)) from exc
    except KeyboardInterrupt as exc:
        typer.echo("cancelled; no partial file was kept.", err=True)
        raise typer.Exit(130) from exc
