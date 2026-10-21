"""Turn chapters into speakable chunks and lazily synthesize them."""

import re
from collections.abc import Callable, Iterable, Iterator

from velora.book import Chapter
from velora.engines.base import Engine, Samples, Voice
from velora.text.chunk import Chunk, chunk_paragraphs
from velora.text.normalize import normalize


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text).lower().split()


def _title_is_spoken(title: str, paragraphs: list[str]) -> bool:
    """True when the chapter text already opens with its title, e.g. a "Chapter 1" heading."""
    first = next((p for p in paragraphs if p), "")
    title_words = _words(title)
    return bool(title_words) and _words(first)[: len(title_words)] == title_words


def chapter_chunks(chapter: Chapter, max_chars: int) -> tuple[Chunk, ...]:
    paragraphs = [normalize(p) for p in chapter.paragraphs]
    title = normalize(chapter.title)
    if not _title_is_spoken(title, paragraphs):
        paragraphs = [title, *paragraphs]
    return chunk_paragraphs(paragraphs, max_chars)


def synthesize(
    engine: Engine,
    chunks: Iterable[Chunk],
    voice: Voice,
    on_chunk: Callable[[Chunk], None],
) -> Iterator[tuple[Samples, bool]]:
    for chunk in chunks:
        samples = engine.synthesize(chunk.text, voice)
        on_chunk(chunk)
        yield samples, chunk.paragraph_end
