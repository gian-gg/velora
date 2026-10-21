import numpy as np

from velora.book import Chapter
from velora.engines.fake import FakeEngine
from velora.read.pipeline import chapter_chunks, synthesize
from velora.text.chunk import Chunk


def chapter(title: str, *paragraphs: str) -> Chapter:
    return Chapter(index=1, title=title, paragraphs=paragraphs)


def test_title_is_spoken_first_when_text_lacks_it() -> None:
    chunks = chapter_chunks(chapter("Dedication", "For Michael."), 300)
    assert [c.text for c in chunks] == ["Dedication", "For Michael."]
    assert chunks[0].paragraph_end


def test_title_is_not_repeated_when_text_opens_with_it() -> None:
    chunks = chapter_chunks(chapter("Chapter 1", "Chapter 1", "It began."), 300)
    assert [c.text for c in chunks] == ["Chapter one", "It began."]


def test_snippet_title_is_not_repeated() -> None:
    chunks = chapter_chunks(chapter("“Dear Kelvin Otieno,”", "Dear Kelvin Otieno,", "Hi."), 300)
    assert [c.text for c in chunks] == ["Dear Kelvin Otieno", "Hi."]


def test_text_is_normalized_and_chunked_to_limit() -> None:
    chunks = chapter_chunks(chapter("Part IV", "Dr. Lee paid $5. " * 10), 40)
    assert chunks[0].text == "Part four"
    assert all(len(c.text) <= 40 for c in chunks)
    assert "Doctor Lee paid five dollars." in chunks[1].text


def test_synthesize_is_lazy_and_reports_each_chunk() -> None:
    seen: list[str] = []
    chunks = [Chunk("one", paragraph_end=False), Chunk("two", paragraph_end=True)]
    segments = synthesize(FakeEngine(), chunks, {}, lambda c: seen.append(c.text))
    assert seen == []
    out = list(segments)
    assert seen == ["one", "two"]
    assert [end for _, end in out] == [False, True]
    assert all(s.dtype == np.float32 for s, _ in out)
