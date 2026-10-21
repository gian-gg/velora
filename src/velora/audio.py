"""Join synthesized chunks with pauses and encode them to a tagged audio file via ffmpeg."""

import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from velora.config import (
    AAC_BITRATE,
    MP3_QUALITY,
    PAUSE_CHUNK_SECONDS,
    PAUSE_PARAGRAPH_SECONDS,
)
from velora.engines.base import Samples

CODECS: dict[str, list[str]] = {
    ".m4a": ["-c:a", "aac", "-b:a", AAC_BITRATE],
    ".mp3": ["-c:a", "libmp3lame", "-q:a", MP3_QUALITY],
    ".wav": ["-c:a", "pcm_s16le"],
}


class AudioError(Exception):
    """ffmpeg is missing, the format is unsupported, or encoding failed."""


@dataclass(frozen=True)
class Tags:
    title: str
    album: str | None = None
    artist: str | None = None
    track: int | None = None

    def ffmpeg_args(self) -> list[str]:
        values = {
            "title": self.title,
            "album": self.album,
            "artist": self.artist,
            "track": None if self.track is None else str(self.track),
        }
        return [
            arg for key, value in values.items() if value for arg in ("-metadata", f"{key}={value}")
        ]


def silence(seconds: float, sample_rate: int) -> Samples:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def assemble(
    segments: Iterable[tuple[Samples, bool]],
    sample_rate: int,
    chunk_pause: float = PAUSE_CHUNK_SECONDS,
    paragraph_pause: float = PAUSE_PARAGRAPH_SECONDS,
) -> Samples:
    """Concatenate (samples, paragraph_end) segments, with a pause after every one but the last."""
    parts: list[Samples] = []
    for samples, paragraph_end in segments:
        parts += [samples, silence(paragraph_pause if paragraph_end else chunk_pause, sample_rate)]
    return np.concatenate(parts[:-1]) if parts else np.zeros(0, dtype=np.float32)


def encode(samples: Samples, sample_rate: int, path: Path, tags: Tags) -> None:
    codec = CODECS.get(path.suffix.lower())
    if codec is None:
        raise AudioError(
            f"unsupported output format {path.suffix!r}; use one of {', '.join(CODECS)}"
        )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioError("ffmpeg not found on PATH; install it to write audio files")

    partial = path.with_name(f".{path.stem}.part{path.suffix}")
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        *tags.ffmpeg_args(), *codec, str(partial),
    ]  # fmt: skip
    pcm = np.clip(samples, -1.0, 1.0).astype("<f4").tobytes()
    result = subprocess.run(command, input=pcm, capture_output=True, check=False)
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise AudioError(f"ffmpeg failed: {result.stderr.decode(errors='replace').strip()}")
    partial.replace(path)
