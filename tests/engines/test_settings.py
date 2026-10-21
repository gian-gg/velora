import pytest

from velora.engines.base import EngineError
from velora.engines.settings import Setting, find_setting, resolve

SPEED = Setting("speed", 1.0, "Rate", minimum=0.5, maximum=2.0)
LANG = Setting("lang", "auto", "Accent", choices=("auto", "en-us"))
VOICE = Setting("voice", "a", "Preset or blend")
ALL = (SPEED, LANG, VOICE)


@pytest.mark.parametrize(("raw", "expected"), [("0.9", 0.9), (2, 2.0), (0.5, 0.5)])
def test_numeric_coerce(raw: object, expected: float) -> None:
    assert SPEED.coerce(raw) == expected


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("fast", "must be a number, got 'fast'"),
        (True, "must be a number"),
        (["1"], "must be a number"),
        (5, "between 0.5 and 2, got 5"),
    ],
)
def test_numeric_coerce_rejects(raw: object, message: str) -> None:
    with pytest.raises(EngineError, match=message):
        SPEED.coerce(raw)


def test_choice_and_text_coerce() -> None:
    assert LANG.coerce("en-us") == "en-us"
    assert VOICE.coerce("  b ") == "b"
    with pytest.raises(EngineError, match="one of auto, en-us"):
        LANG.coerce("fr")
    with pytest.raises(EngineError, match="must be text"):
        VOICE.coerce(" ")


def test_allowed_describes_each_kind() -> None:
    assert [s.allowed() for s in ALL] == ["0.5 to 2", "auto, en-us", "Preset or blend"]


def test_unbounded_number_accepts_anything() -> None:
    assert Setting("n", 1.0, "any").coerce("-50") == -50


def test_resolve_layers_later_wins() -> None:
    values = resolve(ALL, [("file", {"speed": 0.8, "voice": "b"}), ("cli", {"speed": "1.5"})])
    assert values == {"speed": 1.5, "lang": "auto", "voice": "b"}


def test_resolve_labels_errors_with_their_source() -> None:
    with pytest.raises(EngineError, match=r"file: unknown setting\(s\) pitch; allowed: speed"):
        resolve(ALL, [("file", {"pitch": 1})])
    with pytest.raises(EngineError, match="cli: speed must be between"):
        resolve(ALL, [("file", {}), ("cli", {"speed": 9})])


def test_find_setting() -> None:
    assert find_setting(ALL, "lang") is LANG
    assert find_setting(ALL, "nope") is None
