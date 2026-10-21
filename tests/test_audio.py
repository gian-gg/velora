import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from velora.audio import AudioError, Tags, assemble, encode, silence

RATE = 1_000
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def tone(seconds: float) -> np.ndarray:
    return np.full(int(seconds * RATE), 0.5, dtype=np.float32)


def probe(path: Path) -> dict[str, str]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:format_tags",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    fmt = json.loads(out)["format"]
    return {"duration": fmt["duration"], **{k.lower(): v for k, v in fmt.get("tags", {}).items()}}


def test_silence_length() -> None:
    assert len(silence(0.5, RATE)) == 500


def test_assemble_inserts_chunk_and_paragraph_pauses() -> None:
    out = assemble(
        [(tone(1), False), (tone(1), True), (tone(1), True)],
        RATE,
        chunk_pause=0.2,
        paragraph_pause=0.5,
    )
    assert len(out) == 1000 + 200 + 1000 + 500 + 1000
    assert out.dtype == np.float32
    assert not out[1000:1200].any()


def test_assemble_of_nothing_is_empty() -> None:
    assert len(assemble([], RATE)) == 0


def test_tags_skip_missing_values() -> None:
    assert Tags(title="Ch 1", track=3).ffmpeg_args() == [
        "-metadata",
        "title=Ch 1",
        "-metadata",
        "track=3",
    ]


@needs_ffmpeg
def test_encode_wav_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "out.wav"
    encode(tone(2), RATE, path, Tags(title="T"))
    with wave.open(str(path)) as wav:
        assert wav.getframerate() == RATE
        assert wav.getnframes() == 2 * RATE
    assert not list(tmp_path.glob(".*part*"))


@needs_ffmpeg
@pytest.mark.parametrize("suffix", [".m4a", ".mp3"])
def test_encode_compressed_with_tags(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"chapter{suffix}"
    tags = Tags(title="Chapter 1", album="Artemis", artist="Andy Weir", track=4)
    encode(tone(2), 16_000 if suffix == ".mp3" else RATE * 8, path, tags)
    info = probe(path)
    assert info["title"] == "Chapter 1"
    assert info["album"] == "Artemis"
    assert info["artist"] == "Andy Weir"
    assert info["track"].split("/")[0] == "4"


def test_encode_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(AudioError, match=r"unsupported output format '\.ogg'"):
        encode(tone(1), RATE, tmp_path / "x.ogg", Tags(title="T"))


def test_encode_requires_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(AudioError, match="ffmpeg not found"):
        encode(tone(1), RATE, tmp_path / "x.wav", Tags(title="T"))


@needs_ffmpeg
def test_encode_reports_ffmpeg_failure(tmp_path: Path) -> None:
    path = tmp_path / "missing-dir" / "x.wav"
    with pytest.raises(AudioError, match="ffmpeg failed"):
        encode(tone(1), RATE, path, Tags(title="T"))
    assert not path.exists()
