"""Split normalized paragraphs into sentence-aware chunks no longer than an engine's limit."""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from velora.config import DEFAULT_CHUNK_CHARS

SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s+[\"'(\[]?[A-Z0-9])")
INITIAL = re.compile(r"(?:^|\s)[A-Z]\.$")
CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+")


@dataclass(frozen=True)
class Chunk:
    text: str
    paragraph_end: bool


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in SENTENCE_END.finditer(text):
        if INITIAL.search(text[start : match.end()]):
            continue
        sentences.append(text[start : match.end()].strip())
        start = match.end()
    sentences.append(text[start:].strip())
    return [s for s in sentences if s]


def _pack(pieces: Iterable[str], max_chars: int) -> Iterator[str]:
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_chars:
            yield current
            current = piece
        else:
            current = f"{current} {piece}" if current else piece
    if current:
        yield current


def _hard_cut(word: str, max_chars: int) -> list[str]:
    return [word[i : i + max_chars] for i in range(0, len(word), max_chars)]


def _fit(sentence: str, max_chars: int) -> list[str]:
    """Break one over-long sentence at clause punctuation, then spaces, then mid-word."""
    if len(sentence) <= max_chars:
        return [sentence]
    clauses = CLAUSE_BREAK.split(sentence)
    if len(clauses) > 1:
        return [part for clause in clauses for part in _fit(clause, max_chars)]
    words = [part for word in sentence.split() for part in _hard_cut(word, max_chars)]
    return list(_pack(words, max_chars))


def chunk_paragraphs(
    paragraphs: Iterable[str], max_chars: int = DEFAULT_CHUNK_CHARS
) -> tuple[Chunk, ...]:
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    chunks: list[Chunk] = []
    for paragraph in paragraphs:
        pieces = [part for s in split_sentences(paragraph) for part in _fit(s, max_chars)]
        texts = list(_pack(pieces, max_chars))
        chunks += [Chunk(text, paragraph_end=i == len(texts) - 1) for i, text in enumerate(texts)]
    return tuple(chunks)
