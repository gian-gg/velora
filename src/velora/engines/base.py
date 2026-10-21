"""The contract every TTS engine implements, plus readiness checks."""

import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

Samples = NDArray[np.float32]
Voice = Mapping[str, Any]
Progress = Callable[[str, int, int], None]


class EngineError(Exception):
    """An engine cannot be found, is not ready, or rejected its input."""


@dataclass(frozen=True)
class WeightsStatus:
    downloaded: bool
    path: Path | None = None
    size_bytes: int = 0


class Engine(Protocol):
    name: str
    extra: str
    requires: tuple[str, ...]
    max_chunk_chars: int
    sample_rate: int

    def weights_status(self) -> WeightsStatus: ...

    def download(self, progress: Progress) -> None: ...

    def synthesize(self, text: str, voice: Voice) -> Samples:
        """Return mono float32 samples at `sample_rate`."""
        ...


def package_installed(engine: Engine) -> bool:
    return all(importlib.util.find_spec(module) is not None for module in engine.requires)


def install_hint(engine: Engine) -> str:
    return f'uv tool install "velora[{engine.extra}]"'


def ensure_ready(engine: Engine) -> None:
    if not package_installed(engine):
        raise EngineError(f"{engine.name} is not installed. Run: {install_hint(engine)}")
    if not engine.weights_status().downloaded:
        raise EngineError(
            f"{engine.name} weights are not downloaded. Run: velora models download {engine.name}"
        )


def reject_unknown_keys(engine: Engine, voice: Voice, allowed: set[str]) -> None:
    unknown = sorted(set(voice) - allowed)
    if unknown:
        raise EngineError(
            f"{engine.name} does not support voice setting(s) {', '.join(unknown)};"
            f" allowed: {', '.join(sorted(allowed))}"
        )
