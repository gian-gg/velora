import pytest

from velora.book import Book, Chapter
from velora.inspect.render import (
    chapter_minutes,
    render_book,
    render_chapter,
    total_duration,
)


def chapter(index: int, title: str, words: int) -> Chapter:
    return Chapter(index=index, title=title, paragraphs=(" ".join(["w"] * words),))


@pytest.mark.parametrize(
    ("minutes", "expected"), [(0.2, "<1"), (0.5, "1"), (1.4, "1"), (42.6, "43")]
)
def test_chapter_minutes(minutes: float, expected: str) -> None:
    assert chapter_minutes(minutes) == expected


@pytest.mark.parametrize(("minutes", "expected"), [(43, "43m"), (60, "1h 00m"), (521.4, "8h 41m")])
def test_total_duration(minutes: float, expected: str) -> None:
    assert total_duration(minutes) == expected


def test_render_book_lists_chapters_and_totals() -> None:
    book = Book(
        title="Dune",
        author="Frank Herbert",
        chapters=(chapter(1, "Copyright", 30), chapter(2, "Chapter 1", 6000)),
    )
    out = render_book(book, wpm=150)
    lines = out.splitlines()
    assert lines[0] == "Dune — Frank Herbert"
    assert lines[2].split() == ["#", "title", "words", "~min"]
    assert lines[3].split() == ["1", "Copyright", "30", "<1"]
    assert lines[4].split() == ["2", "Chapter", "1", "6,000", "40"]
    assert lines[-1] == "2 chapters · 6,030 words · ~40m at 150 wpm"


def test_render_book_without_author_or_chapters() -> None:
    out = render_book(Book(title="Empty", author=None, chapters=()), wpm=150)
    assert out == "Empty\n\nNo readable chapters found."


def test_render_book_truncates_long_titles() -> None:
    book = Book(title="T", author=None, chapters=(chapter(1, "x" * 100, 10),))
    row = render_book(book, wpm=150).splitlines()[3]
    assert "x" * 47 + "…" in row
    assert "x" * 48 not in row


def test_render_chapter_shows_opening_paragraphs() -> None:
    ch = Chapter(index=3, title="The Gate", paragraphs=("one", "two", "three"))
    assert render_chapter(ch, paragraphs=2) == (
        "3. The Gate  (3 words)\n\none\n\ntwo\n\n… 1 more paragraphs"
    )
    assert render_chapter(ch) == "3. The Gate  (3 words)\n\none\n\ntwo\n\nthree"


def test_render_chapter_applies_transform_and_drops_empty_results() -> None:
    ch = Chapter(index=1, title="T", paragraphs=("a", "skip", "b"))
    out = render_chapter(ch, transform=lambda p: "" if p == "skip" else p.upper())
    assert out == "1. T  (3 words)\n\nA\n\nB"
