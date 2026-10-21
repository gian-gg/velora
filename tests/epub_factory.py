from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ebooklib import epub


@dataclass(frozen=True)
class Doc:
    """One spine document for a generated test EPUB."""

    body: str
    toc_title: str | None = None
    linear: bool = True


MakeEpub = Callable[..., Path]


def epub_builder(tmp_path: Path) -> MakeEpub:
    """Return a function that builds a small EPUB in tmp_path from a list of Docs."""

    def build(
        docs: Sequence[Doc],
        *,
        title: str | None = "Test Book",
        author: str | None = "Ann Author",
        section: str | None = None,
        name: str = "book.epub",
    ) -> Path:
        book = epub.EpubBook()
        book.set_identifier("velora-test")
        book.set_language("en")
        if title is not None:
            book.set_title(title)
        if author is not None:
            book.add_author(author)

        items = []
        for i, doc in enumerate(docs, 1):
            item = epub.EpubHtml(title=doc.toc_title or "", file_name=f"text/ch{i}.xhtml")
            item.content = f"<html><head><title>x</title></head><body>{doc.body}</body></html>"
            book.add_item(item)
            items.append(item)

        links = [
            epub.Link(item.file_name, doc.toc_title, f"toc{i}")
            for i, (item, doc) in enumerate(zip(items, docs, strict=True), 1)
            if doc.toc_title
        ]
        book.toc = [(epub.Section(section), links)] if section else links
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = [
            "nav",
            *[(item, "yes" if doc.linear else "no") for item, doc in zip(items, docs, strict=True)],
        ]

        path = tmp_path / name
        epub.write_epub(str(path), book, {"raise_exceptions": True})
        return path

    return build
