from pathlib import Path

from velora.engines.base import WeightsStatus
from velora.models.render import EngineRow, human_size, render_engines, render_voices, short_path


def test_render_engines_shows_install_and_download_hints() -> None:
    rows = [
        EngineRow("kokoro", True, WeightsStatus(True, Path.home() / ".cache/k", 354_000_000), "x"),
        EngineRow("big", False, WeightsStatus(False), 'uv tool install "velora[big]"'),
    ]
    lines = render_engines(rows).splitlines()
    assert lines[0].split() == ["engine", "package", "weights"]
    assert "installed" in lines[1]
    assert "downloaded  354 MB  ~/.cache/k" in lines[1]
    assert 'missing → uv tool install "velora[big]"' in lines[2]
    assert "not downloaded → velora models download big" in lines[2]


def test_short_path_outside_home_is_absolute() -> None:
    assert short_path(Path("/opt/models")) == "/opt/models"


def test_human_size() -> None:
    assert human_size(28_214_398) == "28 MB"


def test_render_voices_groups_by_prefix() -> None:
    out = render_voices("kokoro", ["bf_emma", "af_heart", "af_bella", "plain"])
    assert out.splitlines()[2:] == ["    plain", "af  bella heart", "bf  emma"]


def test_render_voices_without_presets() -> None:
    assert render_voices("clone", []) == "clone has no preset voices."
