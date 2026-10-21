"""Rewrite chapter text into plain words a TTS model can read aloud (English)."""

import re
from collections.abc import Callable

from num2words import num2words

SUPPORTED_LANGUAGES = frozenset({"en"})

ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
ROMAN_KEYWORDS = r"Chapter|Part|Book|Volume|Act|Scene|Section|Canto|Letter"

ABBREVIATIONS = {
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miz",
    "Dr.": "Doctor",
    "Prof.": "Professor",
    "Jr.": "Junior",
    "Sr.": "Senior",
    "Mt.": "Mount",
    "e.g.": "for example",
    "i.e.": "that is",
    "etc.": "et cetera",
    "vs.": "versus",
}

CURRENCIES = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "€": ("euro", "euros", "cent", "cents"),
}

CHARACTER_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2026": "...",
        "&": " and ",
        "\u00a0": " ",
        "\u00ad": "",
        "\u00a9": "",
        "\u00ae": "",
        "\u2122": "",
    }
)

Rule = tuple[re.Pattern[str], Callable[[re.Match[str]], str] | str]


def _words(value: int | float, **kwargs: str) -> str:
    return str(num2words(value, lang="en", **kwargs)).replace(",", "")


def _int(text: str) -> int:
    return int(text.replace(",", ""))


def _plural(count: int, one: str, many: str) -> str:
    return f"{_words(count)} {one if count == 1 else many}"


def _currency(match: re.Match[str]) -> str:
    major_one, major_many, minor_one, minor_many = CURRENCIES[match["symbol"]]
    major, minor, scale = _int(match["major"]), match["minor"], match["scale"]
    if scale:
        amount = _words(float(f"{major}.{minor}")) if minor else _words(major)
        return f"{amount} {scale} {major_many}"
    if minor and len(minor) != 2:  # noqa: PLR2004
        return f"{_words(float(f'{major}.{minor}'))} {major_many}"
    spoken = _plural(major, major_one, major_many)
    if minor and int(minor):
        spoken += f" and {_plural(int(minor), minor_one, minor_many)}"
    return spoken


def _time(match: re.Match[str]) -> str:
    hour, minute = int(match["hour"]), int(match["minute"])
    if minute == 0:
        return f"{_words(hour)} o'clock"
    tail = f"oh {_words(minute)}" if minute < 10 else _words(minute)  # noqa: PLR2004
    return f"{_words(hour)} {tail}"


def _abbreviation(match: re.Match[str]) -> str:
    """Expand an abbreviation, keeping the full stop when "etc." ends a sentence."""
    word = ABBREVIATIONS[match[0]]
    rest = match.string[match.end() :]
    ends_sentence = match[0] == "etc." and (not rest.strip() or re.match(r"\s+[\"A-Z]", rest))
    return f"{word}." if ends_sentence else word


def _regnal(match: re.Match[str]) -> str:
    value = _roman_to_int(match["numeral"])
    if value is None:
        return match[0]
    return f"{match['name']} the {_words(value, to='ordinal')}"


def _decade(match: re.Match[str]) -> str:
    spoken = _words(int(match[1]), to="year")
    return spoken[:-1] + "ies" if spoken.endswith("y") else spoken + "s"


def _roman_to_int(numeral: str) -> int | None:
    values = [ROMAN_VALUES[c] for c in numeral.upper()]
    total = sum(-v if v < nxt else v for v, nxt in zip(values, [*values[1:], 0], strict=True))
    return total if 0 < total < 4000 and _int_to_roman(total) == numeral.upper() else None  # noqa: PLR2004


def _int_to_roman(value: int) -> str:
    pairs = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
        (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]  # fmt: skip
    out = ""
    for number, letters in pairs:
        count, value = divmod(value, number)
        out += letters * count
    return out


def _roman(match: re.Match[str]) -> str:
    value = _roman_to_int(match["numeral"])
    if value is None:
        return match[0]
    prefix = f"{match['keyword']} " if match["keyword"] else ""
    return f"{prefix}{_words(value)}"


RULES: list[Rule] = [
    (re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+|\[\d{1,3}\]"), ""),
    (re.compile(r"\b(?:https?://|www\.)\S+"), ""),
    (re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])(?!(?:st|nd|rd|th|s)\b)"), " "),
    (
        re.compile("|".join(rf"(?<![\w.]){re.escape(a)}(?=\s|$|[,;:])" for a in ABBREVIATIONS)),
        _abbreviation,
    ),
    (re.compile(r"\bNo\.\s*(?=\d)"), "number "),
    (
        re.compile(
            r"(?P<symbol>[$£€])(?P<major>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<minor>\d+))?"
            r"(?:\s+(?P<scale>thousand|million|billion|trillion)\b)?"
        ),
        _currency,
    ),
    (re.compile(r"(\d+(?:\.\d+)?)\s?%"), lambda m: f"{m[1]} percent"),
    (
        re.compile(r"\b(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)\b(?!:)"),
        _time,
    ),
    (re.compile(r"(?<=\d)\s?[-\u2013]\s?(?=\d)"), " to "),
    (re.compile(r"\s*[\u2014\u2013]+(?=[\"')\]](?:\s|$|[,.;:!?]))"), ","),
    (re.compile(r"\s*[\u2014\u2013]+\s*"), ", "),
    (
        re.compile(rf"\b(?:(?P<keyword>{ROMAN_KEYWORDS})\s+)(?P<numeral>[IVXLCDM]+)\b"),
        _roman,
    ),
    (
        re.compile(r"\b(?P<name>[A-Z][a-z]+)\s+(?P<numeral>(?=[IVX]{2,}\b|V\b|X\b)[IVX]+)\b"),
        _regnal,
    ),
    (re.compile(r"^(?P<keyword>)(?P<numeral>[IVXLCDM]+)\.?$"), _roman),
    (
        re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE),
        lambda m: _words(int(m[1]), to="ordinal"),
    ),
    (re.compile(r"\b\d{7,}\b"), lambda m: " ".join(_words(int(d)) for d in m[0])),
    (re.compile(r"\b(1[1-9]\d0|20\d0)s\b"), _decade),
    (
        re.compile(r"(?<![\d,.])\b(1[1-9]\d\d|20\d\d)\b(?![\d,]|\.\d)"),
        lambda m: _words(int(m[1]), to="year"),
    ),
    (
        re.compile(r"(?<![\d,])(\d{1,3}(?:,\d{3})+|\d+)\.(\d+)\b"),
        lambda m: _words(float(f"{_int(m[1])}.{m[2]}")),
    ),
    (
        re.compile(r"(?<![\d,])(\d{1,3}(?:,\d{3})+|\d+)\b"),
        lambda m: _words(_int(m[1])),
    ),
]


def normalize(text: str, lang: str = "en") -> str:
    """Return text with numbers, abbreviations, symbols and markers spelled out for speech."""
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported language {lang!r}; supported: {sorted(SUPPORTED_LANGUAGES)}")
    out = " ".join(text.translate(CHARACTER_MAP).split())
    for pattern, replacement in RULES:
        out = pattern.sub(replacement, out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r",\s*,", ",", out)
    return " ".join(out.split()).strip(", ")
