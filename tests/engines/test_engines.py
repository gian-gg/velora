from dataclasses import dataclass, field

import numpy as np
import pytest

from velora.engines.base import (
    EngineError,
    Progress,
    Samples,
    Voice,
    WeightsStatus,
    ensure_ready,
    install_hint,
    package_installed,
)
from velora.engines.fake import FakeEngine
from velora.engines.registry import engine_names, get_engine
from velora.engines.settings import Setting


@dataclass
class StubEngine:
    name: str = "stub"
    extra: str = "stub"
    requires: tuple[str, ...] = ()
    max_chunk_chars: int = 100
    sample_rate: int = 8_000
    downloaded: bool = True
    calls: list[str] = field(default_factory=list)
    settings: tuple[Setting, ...] = ()

    def check_voice(self, voice: Voice) -> None:
        self.calls.append("check")

    def weights_status(self) -> WeightsStatus:
        return WeightsStatus(downloaded=self.downloaded)

    def download(self, progress: Progress) -> None:
        self.calls.append("download")

    def remove_weights(self) -> None:
        self.calls.append("remove")

    def voices(self) -> list[str]:
        return ["a_one", "a_two", "b_three"]

    def synthesize(self, text: str, voice: Voice) -> Samples:
        return np.zeros(1, dtype=np.float32)


def test_fake_engine_renders_tone_proportional_to_text() -> None:
    engine = FakeEngine()
    samples = engine.synthesize("x" * 100, {})
    assert samples.dtype == np.float32
    assert len(samples) == engine.sample_rate
    assert 0 < np.abs(samples).max() <= 0.1


def test_fake_engine_rejects_unknown_voice_settings() -> None:
    with pytest.raises(EngineError, match=r"unknown setting\(s\) speed; allowed: frequency"):
        FakeEngine().synthesize("hi", {"speed": 1.2})


def test_registry_returns_engine_by_name() -> None:
    assert get_engine("fake").name == "fake"


def test_registry_rejects_unknown_engine() -> None:
    with pytest.raises(EngineError, match="unknown engine 'nope'"):
        get_engine("nope")


def test_hidden_engines_are_listed_only_on_request() -> None:
    assert "fake" not in engine_names()
    assert "fake" in engine_names(include_hidden=True)


def test_package_installed_checks_required_modules() -> None:
    assert package_installed(StubEngine(requires=("json",)))
    assert not package_installed(StubEngine(requires=("json", "no_such_module_xyz")))


def test_ensure_ready_passes_when_installed_and_downloaded() -> None:
    ensure_ready(StubEngine())


def test_ensure_ready_explains_missing_package() -> None:
    engine = StubEngine(requires=("no_such_module_xyz",))
    with pytest.raises(
        EngineError, match=r'stub is not installed\. Run: uv tool install "velora\[stub\]"'
    ):
        ensure_ready(engine)
    assert install_hint(engine) == 'uv tool install "velora[stub]"'


def test_ensure_ready_explains_missing_weights() -> None:
    with pytest.raises(EngineError, match="Run: velora models download stub"):
        ensure_ready(StubEngine(downloaded=False))


def test_fake_engine_is_always_ready() -> None:
    engine = FakeEngine()
    seen: list[str] = []
    engine.download(lambda name, done, total: seen.append(name))
    assert engine.weights_status().downloaded
    assert seen == ["fake"]
    engine.remove_weights()
    assert engine.voices() == []
    ensure_ready(engine)
