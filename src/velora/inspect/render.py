"""Plain-text rendering of a book's chapter table and chapter previews."""

from collections.abc import Callable

from velora.book import Book, Chapter
from velora.config import (
    INSPECT_MAX_TITLE_WIDTH,
    INSPECT_MIN_SHOWN_MINUTES,
    INSPECT_PREVIEW_PARAGRAPHS,
)


def chapter_minutes(minutes: float) -> str:
    return "<1" if minutes < INSPECT_MIN_SHOWN_MINUTES else str(max(1, round(minutes)))


def total_duration(minutes: float) -> str:
    hours, mins = divmod(round(minutes), 60)
    return f"{hours}h {mins:02d}m" if hours else f"{mins}m"


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def render_book(book: Book, wpm: int) -> str:
    header = book.title if book.author is None else f"{book.title} — {book.author}"
    if not book.chapters:
        return f"{header}\n\nNo readable chapters found."

    num_w = len(str(len(book.chapters)))
    title_w = min(
        INSPECT_MAX_TITLE_WIDTH, max(len("title"), *(len(c.title) for c in book.chapters))
    )
    words_w = max(len("words"), *(len(f"{c.word_count:,}") for c in book.chapters))

    rows = [
        f"  {c.index:>{num_w}}  {_truncate(c.title, title_w):<{title_w}}"
        f"  {c.word_count:>{words_w},}  {chapter_minutes(c.minutes(wpm)):>5}"
        for c in book.chapters
    ]
    summary = (
        f"{len(book.chapters)} chapters · {book.word_count:,} words"
        f" · ~{total_duration(book.minutes(wpm))} at {wpm} wpm"
    )
    return "\n".join(
        [
            header,
            "",
            f"  {'#':>{num_w}}  {'title':<{title_w}}  {'words':>{words_w}}   ~min",
            *rows,
            "",
            summary,
        ]
    )


def render_chapter(
    chapter: Chapter,
    transform: Callable[[str], str] | None = None,
    paragraphs: int = INSPECT_PREVIEW_PARAGRAPHS,
) -> str:
    opening = chapter.paragraphs[:paragraphs]
    remaining = len(chapter.paragraphs) - len(opening)
    shown = [t for t in map(transform, opening) if t] if transform else list(opening)
    blocks = [f"{chapter.index}. {chapter.title}  ({chapter.word_count:,} words)", *shown]
    if remaining:
        blocks.append(f"… {remaining} more paragraphs")
    return "\n\n".join(blocks)
