"""`velora read`: synthesize chapters of a book into tagged audio files."""

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

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
from velora.config import DEFAULT_ENGINE, DEFAULT_OUTPUT_SUFFIX, config_file
from velora.engines.base import Engine, EngineError, ensure_ready
from velora.engines.registry import get_engine
from velora.engines.settings import Value, resolve
from velora.models.render import short_path
from velora.read.pipeline import chapter_chunks, synthesize
from velora.read.selection import SelectionError, default_name, select_chapters
from velora.read.settings import READ_SETTINGS
from velora.readers.epub import EpubError, load_book
from velora.user_config import ConfigError, load_config


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(1)


def _clock(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _render(  # noqa: PLR0913, PLR0917
    engine: Engine,
    book: Book,
    chapters: tuple[Chapter, ...],
    voice: dict[str, Value],
    pauses: dict[str, Value],
    out: Path,
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
        blocks = with_pauses(
            segments,
            engine.sample_rate,
            chunk_pause=float(pauses["chunk_pause"]),
            paragraph_pause=float(pauses["paragraph_pause"]),
        )
        seconds = encode(blocks, engine.sample_rate, out, tags)
    elapsed = time.monotonic() - started
    typer.echo(
        f"wrote {short_path(out)}  {_clock(seconds)} of audio in {_clock(elapsed)}"
        f" ({elapsed / max(seconds, 1e-9):.2f}x real time)"
    )


def _settings(
    engine: Engine, overrides: Mapping[str, object]
) -> tuple[dict[str, Value], dict[str, Value]]:
    """Layer engine defaults, saved config and command-line flags; validate before synthesis."""
    config = load_config()
    saved = short_path(config_file())
    voice = resolve(
        engine.settings,
        [
            (f"{saved} [engines.{engine.name}]", config.engine(engine.name)),
            ("command line", overrides),
        ],
    )
    engine.check_voice(voice)
    pauses = resolve(READ_SETTINGS, [(f"{saved} [read]", config.read)])
    return voice, pauses


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
        str | None,
        typer.Option(help="Override the configured voice, e.g. af_heart:0.6,bf_emma:0.4."),
    ] = None,
    speed: Annotated[
        float | None, typer.Option(help="Override the configured speed; 1.0 is normal.")
    ] = None,
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
    overrides = {k: v for k, v in {"voice": voice, "speed": speed}.items() if v is not None}
    try:
        book = load_book(book_path)
        selected = select_chapters(book, chapter, chapters)
        engine = get_engine(engine_name)
        ensure_ready(engine)
        voice_settings, pauses = _settings(engine, overrides)
    except (EpubError, SelectionError, EngineError, ConfigError) as exc:
        raise _fail(str(exc)) from exc

    try:
        for job_chapters, out in _jobs(book_path, selected, output, split):
            _render(engine, book, job_chapters, voice_settings, pauses, out)
    except (EngineError, AudioError) as exc:
        raise _fail(str(exc)) from exc
    except KeyboardInterrupt as exc:
        typer.echo("cancelled; no partial file was kept.", err=True)
        raise typer.Exit(130) from exc
