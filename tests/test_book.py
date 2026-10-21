import pytest

from velora.book import Book, Chapter


def make_chapter(index: int, *paragraphs: str) -> Chapter:
    return Chapter(index=index, title=f"Ch {index}", paragraphs=paragraphs)


def test_chapter_counts_words_across_paragraphs() -> None:
    chapter = make_chapter(1, "one two three", "four  five")
    assert chapter.word_count == 5


def test_chapter_minutes_uses_wpm() -> None:
    chapter = make_chapter(1, " ".join(["word"] * 300))
    assert chapter.minutes() == 2
    assert chapter.minutes(wpm=100) == 3


def test_book_totals_all_chapters() -> None:
    book = Book(
        title="T",
        author=None,
        chapters=(make_chapter(1, "a b"), make_chapter(2, "c d e")),
    )
    assert book.word_count == 5
    assert book.minutes(wpm=5) == 1


@pytest.mark.parametrize("wpm", [0, -10])
def test_minutes_rejects_non_positive_wpm(wpm: int) -> None:
    with pytest.raises(ValueError, match="wpm must be positive"):
        make_chapter(1, "a").minutes(wpm=wpm)
