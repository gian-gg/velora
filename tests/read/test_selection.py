from pathlib import Path

import pytest

from velora.book import Book, Chapter
from velora.read.selection import SelectionError, default_name, parse_range, select_chapters

BOOK = Book(
    title="B",
    author=None,
    chapters=tuple(Chapter(index=i, title=f"C{i}", paragraphs=("x",)) for i in range(1, 6)),
)


@pytest.mark.parametrize(("spec", "expected"), [("3-5", (3, 5)), ("4", (4, 4)), ("2-2", (2, 2))])
def test_parse_range(spec: str, expected: tuple[int, int]) -> None:
    assert parse_range(spec) == expected


@pytest.mark.parametrize("spec", ["a-b", "5-3", "-", "1-x"])
def test_parse_range_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(SelectionError):
        parse_range(spec)


def test_select_single_and_range() -> None:
    assert [c.index for c in select_chapters(BOOK, 2, None)] == [2]
    assert [c.index for c in select_chapters(BOOK, None, "3-5")] == [3, 4, 5]


@pytest.mark.parametrize(
    ("chapter", "chapters", "message"),
    [
        (None, None, "exactly one"),
        (1, "1-2", "exactly one"),
        (0, None, "between 1 and 5"),
        (None, "4-9", "between 1 and 5"),
    ],
)
def test_select_rejects_bad_requests(
    chapter: int | None, chapters: str | None, message: str
) -> None:
    with pytest.raises(SelectionError, match=message):
        select_chapters(BOOK, chapter, chapters)


def test_default_name() -> None:
    path = Path("/books/artemis.epub")
    assert default_name(path, BOOK.chapters[3:4], ".m4a") == "artemis-ch04.m4a"
    assert default_name(path, BOOK.chapters[2:5], ".mp3") == "artemis-ch03-05.mp3"
