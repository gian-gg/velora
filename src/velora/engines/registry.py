"""Engine lookup by name; engine modules import their heavy packages lazily."""

from collections.abc import Callable

from velora.engines.base import Engine, EngineError
from velora.engines.fake import FakeEngine

ENGINES: dict[str, Callable[[], Engine]] = {
    "fake": FakeEngine,
}
HIDDEN_ENGINES = frozenset({"fake"})


def get_engine(name: str) -> Engine:
    if name not in ENGINES:
        raise EngineError(f"unknown engine {name!r}; available: {', '.join(engine_names())}")
    return ENGINES[name]()


def engine_names(include_hidden: bool = False) -> list[str]:
    return [n for n in ENGINES if include_hidden or n not in HIDDEN_ENGINES]
