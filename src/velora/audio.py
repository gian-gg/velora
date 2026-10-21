"""Join synthesized chunks with pauses and encode them to a tagged audio file via ffmpeg."""

import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
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


def with_pauses(
    segments: Iterable[tuple[Samples, bool]],
    sample_rate: int,
    chunk_pause: float = PAUSE_CHUNK_SECONDS,
    paragraph_pause: float = PAUSE_PARAGRAPH_SECONDS,
) -> Iterator[Samples]:
    """Yield (samples, paragraph_end) segments lazily, with a pause between consecutive ones."""
    pause: Samples | None = None
    for samples, paragraph_end in segments:
        if pause is not None:
            yield pause
        yield samples
        pause = silence(paragraph_pause if paragraph_end else chunk_pause, sample_rate)


def _feed(process: subprocess.Popen[bytes], blocks: Iterable[Samples]) -> int:
    """Stream blocks to ffmpeg's stdin; return samples written, stopping if ffmpeg exits early."""
    assert process.stdin is not None
    written = 0
    try:
        for block in blocks:
            process.stdin.write(np.clip(block, -1.0, 1.0).astype("<f4").tobytes())
            written += len(block)
        process.stdin.close()
    except BrokenPipeError:
        pass
    return written


def encode(blocks: Iterable[Samples], sample_rate: int, path: Path, tags: Tags) -> float:
    """Stream sample blocks through ffmpeg into path; return the seconds of audio written."""
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
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
        )
        try:
            written = _feed(process, blocks)
        except BaseException:
            process.kill()
            process.wait()
            partial.unlink(missing_ok=True)
            raise
        if process.wait() != 0:
            partial.unlink(missing_ok=True)
            errors.seek(0)
            raise AudioError(f"ffmpeg failed: {errors.read().decode(errors='replace').strip()}")
    partial.replace(path)
    return written / sample_rate
