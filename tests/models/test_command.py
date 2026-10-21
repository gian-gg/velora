import hashlib
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from velora.cli import app
from velora.engines import kokoro
from velora.engines.download import RemoteFile

runner = CliRunner()


@pytest.fixture
def fake_kokoro_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point kokoro at small local files and an isolated cache; return the cache dir."""
    release = tmp_path / "release"
    release.mkdir()
    np.savez(release / "voices.npz", af_heart=np.zeros(1), bf_emma=np.zeros(1), solo=np.zeros(1))
    (release / "model.onnx").write_bytes(b"model")
    files = tuple(
        RemoteFile(
            name, path.as_uri(), hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size
        )
        for name, path in [
            ("kokoro-v1.0.onnx", release / "model.onnx"),
            ("voices-v1.0.bin", release / "voices.npz"),
        ]
    )
    monkeypatch.setattr(kokoro, "FILES", files)
    monkeypatch.setattr(kokoro, "VOICES", files[1])
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    return tmp_path / "xdg" / "velora" / "models" / "kokoro"


def test_lists_engines_with_download_hint(fake_kokoro_release: Path) -> None:
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "kokoro  installed" in result.stdout
    assert "not downloaded → velora models download kokoro" in result.stdout
    assert "fake" not in result.stdout


def test_download_then_list_voices_then_remove(fake_kokoro_release: Path) -> None:
    result = runner.invoke(app, ["models", "download", "kokoro"])
    assert result.exit_code == 0, result.output
    assert "kokoro weights downloaded." in result.stdout
    assert (fake_kokoro_release / "kokoro-v1.0.onnx").read_bytes() == b"model"

    assert "already downloaded" in runner.invoke(app, ["models", "download", "kokoro"]).stdout
    assert "downloaded  " in runner.invoke(app, ["models"]).stdout

    voices = runner.invoke(app, ["models", "voices", "kokoro"]).stdout
    assert "kokoro voices (3)" in voices
    assert "af  heart" in voices.splitlines()
    assert "bf  emma" in voices.splitlines()
    assert "    solo" in voices.splitlines()

    assert "kokoro weights removed." in runner.invoke(app, ["models", "remove", "kokoro"]).stdout
    assert not fake_kokoro_release.exists()
    assert "had no downloaded weights" in runner.invoke(app, ["models", "remove", "kokoro"]).stdout


def test_voices_requires_download(fake_kokoro_release: Path) -> None:
    result = runner.invoke(app, ["models", "voices", "kokoro"])
    assert result.exit_code == 1
    assert "Run: velora models download kokoro" in result.stderr


def test_download_reports_corrupt_file(
    fake_kokoro_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tuple(RemoteFile(f.name, f.url, "0" * 64, f.size) for f in kokoro.FILES)
    monkeypatch.setattr(kokoro, "FILES", bad)
    result = runner.invoke(app, ["models", "download", "kokoro"])
    assert result.exit_code == 1
    assert "corrupt" in result.stderr


@pytest.mark.parametrize("sub", ["download", "remove", "voices"])
def test_unknown_engine_fails(sub: str) -> None:
    result = runner.invoke(app, ["models", sub, "nope"])
    assert result.exit_code == 1
    assert "unknown engine 'nope'" in result.stderr
