"""Format-independent book and chapter models."""

from dataclasses import dataclass

from velora.config import DEFAULT_WPM


def _minutes(words: int, wpm: int) -> float:
    if wpm <= 0:
        raise ValueError(f"wpm must be positive, got {wpm}")
    return words / wpm


@dataclass(frozen=True)
class Chapter:
    index: int
    title: str
    paragraphs: tuple[str, ...]

    @property
    def word_count(self) -> int:
        return sum(len(p.split()) for p in self.paragraphs)

    def minutes(self, wpm: int = DEFAULT_WPM) -> float:
        return _minutes(self.word_count, wpm)


@dataclass(frozen=True)
class Book:
    title: str
    author: str | None
    chapters: tuple[Chapter, ...]

    @property
    def word_count(self) -> int:
        return sum(c.word_count for c in self.chapters)

    def minutes(self, wpm: int = DEFAULT_WPM) -> float:
        return _minutes(self.word_count, wpm)
