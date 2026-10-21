"""Resolve --chapter / --chapters into the chapters to read and a default output name."""

from pathlib import Path

from velora.book import Book, Chapter


class SelectionError(Exception):
    """The requested chapters are missing, malformed, or out of range."""


def parse_range(spec: str) -> tuple[int, int]:
    first, _, last = spec.partition("-")
    try:
        start, end = int(first), int(last or first)
    except ValueError as exc:
        raise SelectionError(f"invalid chapter range {spec!r}; use A-B, e.g. 3-5") from exc
    if start > end:
        raise SelectionError(f"invalid chapter range {spec!r}; start is after end")
    return start, end


def select_chapters(book: Book, chapter: int | None, chapters: str | None) -> tuple[Chapter, ...]:
    if (chapter is None) == (chapters is None):
        raise SelectionError("pass exactly one of --chapter N or --chapters A-B")
    start, end = (chapter, chapter) if chapter is not None else parse_range(str(chapters))
    count = len(book.chapters)
    if start < 1 or end > count:
        raise SelectionError(f"chapters must be between 1 and {count}; see velora inspect")
    return book.chapters[start - 1 : end]


def default_name(book_path: Path, selected: tuple[Chapter, ...], suffix: str) -> str:
    first, last = selected[0].index, selected[-1].index
    span = f"{first:02d}" if first == last else f"{first:02d}-{last:02d}"
    return f"{book_path.stem}-ch{span}{suffix}"
