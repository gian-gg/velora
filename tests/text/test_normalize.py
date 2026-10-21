import pytest

from velora.text.normalize import normalize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mr. Smith met Dr. Jones.", "Mister Smith met Doctor Jones."),
        (
            "Ms. Teller\u2019s class, e.g. today, i.e. now, etc.",
            "Miz Teller's class, for example today, that is now, et cetera.",
        ),
        ("Room No. 7.", "Room number seven."),
        ("Pens, ink, etc. Then more.", "Pens, ink, et cetera. Then more."),
        ("ink, etc. and more", "ink, et cetera and more"),
        ("It's Dr.Who.", "It's Dr.Who."),
    ],
)
def test_abbreviations(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12 cats", "twelve cats"),
        ("1,500 soldiers", "one thousand five hundred soldiers"),
        ("3.14 pies", "three point one four pies"),
        ("the 21st and 3rd", "the twenty-first and third"),
        ("50% off", "fifty percent off"),
        ("12.5 % more", "twelve point five percent more"),
        (
            "In 1847 and 1905 and 2024.",
            "In eighteen forty-seven and nineteen oh-five and twenty twenty-four.",
        ),
        ("from 1914\u20131918", "from nineteen fourteen to nineteen eighteen"),
        ("4-5 years", "four to five years"),
        ("in the 1990s and 1800s", "in the nineteen nineties and eighteen hundreds"),
        (
            "ISBN 9780553448139",
            "ISBN nine seven eight zero five five three four four eight one three nine",
        ),
        ("Kovalto1 and v5.1 and MP3", "Kovalto one and v five point one and MP three"),
        ("the 4th of 10am", "the fourth of ten am"),
    ],
)
def test_numbers(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$5.50", "five dollars and fifty cents"),
        ("$1", "one dollar"),
        ("$0.01", "zero dollars and one cent"),
        ("£3 million", "three million pounds"),
        ("$1.5 billion", "one point five billion dollars"),
        ("€2.00", "two euros"),
        ("$2.5", "two point five dollars"),
    ],
)
def test_currency(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("at 10:30", "at ten thirty"),
        ("at 9:05", "at nine oh five"),
        ("at 7:00", "at seven o'clock"),
    ],
)
def test_times(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Chapter XII", "Chapter twelve"),
        ("Part IV. Book ii", "Part four. Book ii"),
        ("XIV", "fourteen"),
        ("IV.", "four"),
        ("Henry I went home. I think so.", "Henry I went home. I think so."),
        ("Chapter IIII", "Chapter IIII"),
        (
            "Richbastard III and Henry V and Louis XIV",
            "Richbastard the third and Henry the fifth and Louis the fourteenth",
        ),
        ("Plan IIII", "Plan IIII"),
    ],
)
def test_roman_numerals(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\u201cWait\u2014\u201d she said\u2026", '"Wait," she said...'),
        ("He paused\u2014then left.", "He paused, then left."),
        ("is\u201d\u2014he smiled\u2014\u201cJazz!", 'is", he smiled, "Jazz!'),
        ("\u2014", ""),
        ("See footnote¹ and [12] here.", "See footnote and here."),
        ("Visit https://example.com/x now.", "Visit now."),
        ("salt & pepper", "salt and pepper"),
        ("Copyright \u00a9 Acme\u2122", "Copyright Acme"),
        ("  lots \n of   space ", "lots of space"),
        ("non\u00a0breaking soft\u00adhyphen", "non breaking softhyphen"),
    ],
)
def test_cleanup(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_unsupported_language_raises() -> None:
    with pytest.raises(ValueError, match="unsupported language 'fr'"):
        normalize("Bonjour", lang="fr")
