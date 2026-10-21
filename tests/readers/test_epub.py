from pathlib import Path

import pytest

from tests.epub_factory import Doc, MakeEpub
from velora.readers.epub import EpubError, load_book


def test_reads_title_and_author(make_epub: MakeEpub) -> None:
    book = load_book(make_epub([Doc("<p>Hi.</p>")], title="Dune", author="Frank Herbert"))
    assert book.title == "Dune"
    assert book.author == "Frank Herbert"


def test_missing_metadata_falls_back_to_file_stem_and_no_author(make_epub: MakeEpub) -> None:
    path = make_epub([Doc("<p>Hi.</p>")], title=None, author=None, name="my-book.epub")
    book = load_book(path)
    assert book.title == "my-book"
    assert book.author is None


def test_chapters_follow_spine_order_with_toc_titles(make_epub: MakeEpub) -> None:
    book = load_book(
        make_epub(
            [
                Doc("<p>First.</p>", toc_title="Prologue"),
                Doc("<p>Second.</p>", toc_title="Chapter One"),
            ]
        )
    )
    assert [(c.index, c.title) for c in book.chapters] == [(1, "Prologue"), (2, "Chapter One")]
    assert book.chapters[0].paragraphs == ("First.",)


def test_nav_document_is_not_a_chapter(make_epub: MakeEpub) -> None:
    book = load_book(make_epub([Doc("<p>Only.</p>", toc_title="Only")]))
    assert [c.title for c in book.chapters] == ["Only"]


def test_nested_toc_titles_are_found(make_epub: MakeEpub) -> None:
    path = make_epub([Doc("<p>Text.</p>", toc_title="Chapter 1")], section="Part One")
    assert load_book(path).chapters[0].title == "Chapter 1"


def test_non_linear_and_empty_documents_are_skipped(make_epub: MakeEpub) -> None:
    book = load_book(
        make_epub(
            [
                Doc('<img src="cover.jpg" alt=""/>'),
                Doc("<p>Story.</p>", toc_title="Story"),
                Doc("<p>Footnote.</p>", toc_title="Notes", linear=False),
                Doc("<p>End.</p>", toc_title="End"),
            ]
        )
    )
    assert [(c.index, c.title) for c in book.chapters] == [(1, "Story"), (2, "End")]


def test_title_falls_back_to_heading_then_number(make_epub: MakeEpub) -> None:
    book = load_book(
        make_epub(
            [
                Doc("<h2>The  Gate</h2><p>Text.</p>"),
                Doc("<p>No heading here.</p>"),
            ]
        )
    )
    assert [c.title for c in book.chapters] == ["The Gate", "Chapter 2"]


def test_paragraph_text_is_cleaned(make_epub: MakeEpub) -> None:
    body = (
        "<h1>Title</h1>"
        "<p>Line one<br/>line   two <em>stressed</em>.</p>"
        "<script>var x = 1;</script><style>p {}</style>"
        "<ul><li><p>Nested item.</p></li><li>Plain item.</li></ul>"
        "<div><div>Div only text.</div></div>"
        "<p>   </p>"
    )
    book = load_book(make_epub([Doc(body)]))
    assert book.chapters[0].paragraphs == (
        "Title",
        "Line one line two stressed.",
        "Nested item.",
        "Plain item.",
        "Div only text.",
    )


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(EpubError, match="file not found"):
        load_book(tmp_path / "nope.epub")


def test_non_epub_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "fake.epub"
    path.write_text("not a zip")
    with pytest.raises(EpubError, match="not a readable EPUB"):
        load_book(path)
