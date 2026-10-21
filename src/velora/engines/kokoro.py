"""Kokoro-82M via kokoro-onnx: preset and blended voices, CPU-friendly."""

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from velora.config import KOKORO_DEFAULT_VOICE, models_dir
from velora.engines.base import (
    EngineError,
    Progress,
    Samples,
    Voice,
    WeightsStatus,
    reject_unknown_keys,
)
from velora.engines.download import RemoteFile, fetch

if TYPE_CHECKING:
    from kokoro_onnx import Kokoro

RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
MODEL = RemoteFile(
    name="kokoro-v1.0.onnx",
    url=f"{RELEASE}/kokoro-v1.0.onnx",
    sha256="7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    size=325_532_387,
)
VOICES = RemoteFile(
    name="voices-v1.0.bin",
    url=f"{RELEASE}/voices-v1.0.bin",
    sha256="bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
    size=28_214_398,
)
FILES = (MODEL, VOICES)
MIN_SPEED, MAX_SPEED = 0.5, 2.0


def parse_blend(spec: str) -> list[tuple[str, float]]:
    """Parse "af_heart" or "af_heart:0.6,bf_emma:0.4" into weights that sum to 1."""
    parts: list[tuple[str, float]] = []
    for item in spec.split(","):
        name, _, weight = item.strip().partition(":")
        try:
            value = float(weight) if weight else 1.0
        except ValueError as exc:
            raise EngineError(f"invalid weight {weight!r} for voice {name!r}") from exc
        if not name or value <= 0:
            raise EngineError(f"invalid voice blend {spec!r}")
        parts.append((name, value))
    total = sum(w for _, w in parts)
    return [(name, w / total) for name, w in parts]


class KokoroEngine:
    name = "kokoro"
    extra = "kokoro"
    requires: tuple[str, ...] = ("kokoro_onnx",)
    max_chunk_chars = 300
    sample_rate = 24_000

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or models_dir() / "kokoro"

    def weights_status(self) -> WeightsStatus:
        return WeightsStatus(
            downloaded=all(f.present(self.directory) for f in FILES),
            path=self.directory,
            size_bytes=sum(f.size for f in FILES),
        )

    def download(self, progress: Progress) -> None:
        for remote in FILES:
            if not remote.present(self.directory):
                fetch(remote, self.directory, progress)

    def voices(self) -> list[str]:
        with np.load(VOICES.path(self.directory)) as data:
            return sorted(data.files)

    @cached_property
    def _voice_names(self) -> frozenset[str]:
        return frozenset(self.voices())

    @cached_property
    def _model(self) -> "Kokoro":
        from kokoro_onnx import Kokoro  # noqa: PLC0415

        return Kokoro(str(MODEL.path(self.directory)), str(VOICES.path(self.directory)))

    def _style(self, spec: str) -> Samples:
        blend = parse_blend(spec)
        missing = [name for name, _ in blend if name not in self._voice_names]
        if missing:
            raise EngineError(f"unknown kokoro voice(s) {', '.join(missing)}; see velora models")
        style = sum(w * self._model.get_voice_style(name) for name, w in blend)
        return np.asarray(style, dtype=np.float32)

    def synthesize(self, text: str, voice: Voice) -> Samples:
        reject_unknown_keys(self, voice, {"voice", "speed"})
        speed = float(voice.get("speed", 1.0))
        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise EngineError(f"kokoro speed must be between {MIN_SPEED} and {MAX_SPEED}")
        style = self._style(str(voice.get("voice", KOKORO_DEFAULT_VOICE)))
        samples, _ = self._model.create(text, voice=style, speed=speed, lang="en-us")
        return np.asarray(samples, dtype=np.float32)
