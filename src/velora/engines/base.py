"""The contract every TTS engine implements, plus readiness checks."""

import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from velora.engines.settings import Setting

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
    settings: tuple["Setting", ...]

    def weights_status(self) -> WeightsStatus: ...

    def download(self, progress: Progress) -> None: ...

    def remove_weights(self) -> None: ...

    def voices(self) -> list[str]: ...

    def check_voice(self, voice: Voice) -> None:
        """Raise EngineError if voice settings are invalid, without loading the model."""
        ...

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
