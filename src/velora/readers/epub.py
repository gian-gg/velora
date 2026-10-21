"""Load an EPUB into a Book: spine order, TOC titles, paragraph text."""

import posixpath
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import ebooklib
from ebooklib import epub
from selectolax.lexbor import LexborHTMLParser

from velora.book import Book, Chapter
from velora.config import UNTITLED_SNIPPET_CHARS

BLOCKS = "p, h1, h2, h3, h4, h5, h6, li, dt, dd, blockquote, pre, td, div"
HEADINGS = "h1, h2, h3"


class EpubError(Exception):
    """The file is missing or is not a readable EPUB."""


def load_book(path: Path) -> Book:
    if not path.is_file():
        raise EpubError(f"file not found: {path}")
    try:
        source = epub.read_epub(str(path), options={"ignore_ncx": False})
    except Exception as exc:
        raise EpubError(f"not a readable EPUB: {path} ({exc})") from exc

    toc = _toc_titles(source.toc)
    chapters: list[Chapter] = []
    for idref, linear in source.spine:
        item = source.get_item_with_id(idref)
        if linear == "no" or not _is_readable(item):
            continue
        paragraphs, heading = _extract_text(item.get_content())
        if not paragraphs:
            continue
        title = _lookup(toc, item.get_name()) or heading or _snippet(paragraphs[0])
        chapters.append(Chapter(index=len(chapters) + 1, title=title, paragraphs=paragraphs))

    return Book(
        title=_metadata(source, "title") or path.stem,
        author=_metadata(source, "creator"),
        chapters=tuple(chapters),
    )


def _is_readable(item: Any) -> bool:
    return (
        item is not None
        and not isinstance(item, epub.EpubNav)
        and item.get_type() == ebooklib.ITEM_DOCUMENT
    )


def _clean(text: str) -> str:
    return " ".join(text.split())


def _snippet(text: str) -> str:
    """Quoted opening words, used as the title of a chapter with no TOC entry or heading."""
    if len(text) <= UNTITLED_SNIPPET_CHARS:
        return f"“{text}”"
    cut = text[:UNTITLED_SNIPPET_CHARS].rsplit(" ", 1)[0]
    return f"“{cut}…”"


def _metadata(source: Any, name: str) -> str | None:
    for value, _attrs in source.get_metadata("DC", name):
        if isinstance(value, str) and value.strip():
            return _clean(value)
    return None


def _iter_toc(entries: Iterable[Any] | Any) -> Iterator[tuple[str, str]]:
    """Yield (href, title) pairs, children before their section so the deeper title wins."""
    if not isinstance(entries, list | tuple):
        entries = [entries]
    for entry in entries:
        if isinstance(entry, tuple):
            section, children = entry
            yield from _iter_toc(children)
            yield section.href or "", section.title or ""
        elif isinstance(entry, epub.Link):
            yield entry.href or "", entry.title or ""
        elif isinstance(entry, epub.EpubHtml):
            yield entry.get_name(), entry.title or ""


def _href_key(href: str) -> str:
    return posixpath.normpath(unquote(href.split("#", 1)[0]))


def _toc_titles(entries: Iterable[Any]) -> dict[str, str]:
    titles: dict[str, str] = {}
    for href, title in _iter_toc(entries):
        if href and title.strip():
            titles.setdefault(_href_key(href), _clean(title))
    return titles


def _lookup(toc: dict[str, str], name: str) -> str | None:
    key = _href_key(name)
    if key in toc:
        return toc[key]
    base = posixpath.basename(key)
    return next((title for href, title in toc.items() if posixpath.basename(href) == base), None)


def _extract_text(content: bytes) -> tuple[tuple[str, ...], str | None]:
    tree = LexborHTMLParser(content)
    body = tree.body
    if body is None:
        return (), None
    tree.strip_tags(["script", "style"])
    for br in body.css("br"):
        br.replace_with(" ")

    paragraphs: list[str] = []
    heading: str | None = None
    for node in body.css(BLOCKS):
        if len(node.css(BLOCKS)) > 1:
            continue
        text = _clean(node.text())
        if not text:
            continue
        paragraphs.append(text)
        if heading is None and node.tag in HEADINGS.split(", "):
            heading = text
    return tuple(paragraphs), heading
