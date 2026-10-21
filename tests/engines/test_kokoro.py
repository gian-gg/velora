import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from velora.engines import kokoro
from velora.engines.base import EngineError, package_installed
from velora.engines.download import RemoteFile
from velora.engines.kokoro import KokoroEngine, parse_blend
from velora.engines.registry import get_engine

REAL_WEIGHTS = os.environ.get("VELORA_KOKORO_DIR")
needs_weights = pytest.mark.skipif(not REAL_WEIGHTS, reason="set VELORA_KOKORO_DIR to run")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("af_heart", [("af_heart", 1.0)]),
        ("af_heart:3, bf_emma:1", [("af_heart", 0.75), ("bf_emma", 0.25)]),
    ],
)
def test_parse_blend(spec: str, expected: list[tuple[str, float]]) -> None:
    assert parse_blend(spec) == expected


@pytest.mark.parametrize("spec", ["", "af_heart:0", "af_heart:-1", "af_heart:abc", ":0.5"])
def test_parse_blend_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(EngineError):
        parse_blend(spec)


def test_registered_and_installed() -> None:
    engine = get_engine("kokoro")
    assert engine.sample_rate == 24_000
    assert package_installed(engine)


def test_weights_status_and_download_use_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "m.bin"
    source.write_bytes(b"model")
    remote = RemoteFile("m.bin", source.as_uri(), hashlib.sha256(b"model").hexdigest(), 5)
    monkeypatch.setattr(kokoro, "FILES", (remote,))
    engine = KokoroEngine(tmp_path / "cache")
    assert not engine.weights_status().downloaded
    engine.download(lambda *_: None)
    status = engine.weights_status()
    assert status.downloaded
    assert status.path == tmp_path / "cache"
    assert status.size_bytes == 5


def test_default_directory_follows_xdg_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg")
    assert KokoroEngine().directory == Path("/tmp/xdg/velora/models/kokoro")


def test_synthesize_validates_settings_before_loading_model(tmp_path: Path) -> None:
    engine = KokoroEngine(tmp_path)
    with pytest.raises(EngineError, match="does not support"):
        engine.synthesize("hi", {"pitch": 2})
    with pytest.raises(EngineError, match="speed must be between"):
        engine.synthesize("hi", {"speed": 5})


@needs_weights
def test_real_synthesis_with_blend_and_unknown_voice() -> None:
    engine = KokoroEngine(Path(str(REAL_WEIGHTS)))
    assert "af_heart" in engine.voices()
    samples = engine.synthesize("Hello there.", {"voice": "af_heart:0.6,bf_emma:0.4"})
    assert samples.dtype == np.float32
    assert len(samples) > engine.sample_rate // 2
    with pytest.raises(EngineError, match="unknown kokoro voice"):
        engine.synthesize("Hi.", {"voice": "zz_nobody"})
