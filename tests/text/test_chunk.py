import pytest

from velora.text.chunk import Chunk, chunk_paragraphs, split_sentences


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two! Three?", ["One.", "Two!", "Three?"]),
        ('"Hi," she said. "Bye."', ['"Hi," she said.', '"Bye."']),
        ("He left (quickly.) Then rain.", ["He left (quickly.)", "Then rain."]),
        ("Wait... what? Really?!", ["Wait... what?", "Really?!"]),
        ("ends with lowercase. then this", ["ends with lowercase. then this"]),
        ("J. Worthalot paid. Fine.", ["J. Worthalot paid.", "Fine."]),
        ("It was 1984. Then 2001.", ["It was 1984.", "Then 2001."]),
        ("No terminal punctuation", ["No terminal punctuation"]),
        ("   ", []),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


def test_packs_sentences_up_to_limit() -> None:
    chunks = chunk_paragraphs(["Aaa bb. Ccc dd. Eee ff."], max_chars=16)
    assert chunks == (
        Chunk("Aaa bb. Ccc dd.", paragraph_end=False),
        Chunk("Eee ff.", paragraph_end=True),
    )


def test_marks_last_chunk_of_each_paragraph() -> None:
    chunks = chunk_paragraphs(["First.", "Second. Third."], max_chars=100)
    assert chunks == (
        Chunk("First.", paragraph_end=True),
        Chunk("Second. Third.", paragraph_end=True),
    )


def test_skips_empty_paragraphs() -> None:
    assert chunk_paragraphs(["", "  ", "Real."]) == (Chunk("Real.", paragraph_end=True),)


def test_long_sentence_breaks_at_clauses() -> None:
    sentence = "one two three, four five six; seven eight nine."
    chunks = chunk_paragraphs([sentence], max_chars=20)
    assert [c.text for c in chunks] == ["one two three,", "four five six;", "seven eight nine."]


def test_long_clause_breaks_at_words() -> None:
    chunks = chunk_paragraphs(["alpha beta gamma delta epsilon"], max_chars=12)
    assert [c.text for c in chunks] == ["alpha beta", "gamma delta", "epsilon"]


def test_long_word_is_cut() -> None:
    chunks = chunk_paragraphs(["abcdefghij"], max_chars=4)
    assert [c.text for c in chunks] == ["abcd", "efgh", "ij"]


def test_no_chunk_exceeds_limit() -> None:
    text = " ".join(f"Sentence number {i}, with a clause; and more words here." for i in range(50))
    chunks = chunk_paragraphs([text], max_chars=40)
    assert all(len(c.text) <= 40 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == text.split()


@pytest.mark.parametrize("max_chars", [0, -1])
def test_rejects_non_positive_limit(max_chars: int) -> None:
    with pytest.raises(ValueError, match="max_chars must be positive"):
        chunk_paragraphs(["x"], max_chars=max_chars)
