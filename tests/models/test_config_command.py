from pathlib import Path

import pytest
from typer.testing import CliRunner

from velora.cli import app

runner = CliRunner()


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cfg" / "velora" / "config.toml"


def run(*args: str) -> tuple[int, str, str]:
    result = runner.invoke(app, ["models", "config", *args])
    return result.exit_code, result.stdout, result.stderr


def test_show_lists_every_setting_with_defaults(config_path: Path) -> None:
    code, out, _ = run("kokoro")
    assert code == 0
    for name in ("voice", "speed", "lang", "sentence_pause", "clause_pause", "chunk_pause"):
        assert name in out
    assert "auto, en-us, en-gb" in out
    assert "all engines" in out
    assert not config_path.exists()


def test_set_saves_and_marks_values_then_unset_restores(config_path: Path) -> None:
    assert run("kokoro", "set", "speed", "0.9")[0] == 0
    assert run("kokoro", "set", "paragraph_pause", "1.2")[0] == 0
    assert "[engines.kokoro]\nspeed = 0.9" in config_path.read_text()
    assert "[read]\nparagraph_pause = 1.2" in config_path.read_text()

    out = run("kokoro")[1]
    assert "0.9 *" in out
    assert "1.2 *" in out

    code, out, _ = run("kokoro", "unset", "speed")
    assert code == 0
    assert "reset to default 1" in out
    assert "speed" not in config_path.read_text()


def test_set_accepts_blend_without_downloaded_weights(config_path: Path) -> None:
    code, _, _ = run("kokoro", "set", "voice", "bf_emma:0.7,bf_isabella:0.3")
    assert code == 0
    assert 'voice = "bf_emma:0.7,bf_isabella:0.3"' in config_path.read_text()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("set", "speed", "5"), "speed must be between 0.5 and 2, got 5"),
        (("set", "lang", "fr"), "lang must be one of auto, en-us, en-gb"),
        (("set", "voice", "af_heart:0"), "invalid voice blend"),
        (("set", "pitch", "2"), "unknown setting 'pitch'"),
        (("unset", "pitch"), "unknown setting 'pitch'"),
    ],
)
def test_invalid_changes_are_rejected_and_not_saved(
    config_path: Path, args: tuple[str, ...], message: str
) -> None:
    code, _, err = run("kokoro", *args)
    assert code == 1
    assert message in err
    assert not config_path.exists()


def test_unknown_engine(config_path: Path) -> None:
    code, _, err = run("nope")
    assert code == 1
    assert "unknown engine 'nope'" in err


def test_broken_config_file_is_reported(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True)
    config_path.write_text("oops = 1")
    code, _, err = run("kokoro")
    assert code == 1
    assert "unknown section(s) oops" in err
