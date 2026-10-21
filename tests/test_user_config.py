from pathlib import Path

import pytest

from velora.config import config_file
from velora.user_config import ConfigError, UserConfig, load_config, save_config, with_value


def test_missing_file_is_empty_config(tmp_path: Path) -> None:
    assert load_config(tmp_path / "none.toml") == UserConfig()


def test_round_trip_and_immutable_updates(tmp_path: Path) -> None:
    path = tmp_path / "velora" / "config.toml"
    original = UserConfig()
    updated = with_value(
        with_value(original, "kokoro", "voice", "bf_emma"), "read", "chunk_pause", 0.5
    )
    assert original == UserConfig()
    save_config(updated, path)
    assert path.read_text() == (
        '[read]\nchunk_pause = 0.5\n\n[engines.kokoro]\nvoice = "bf_emma"\n'
    )
    assert load_config(path) == updated


def test_unsetting_last_value_drops_the_table(tmp_path: Path) -> None:
    config = with_value(UserConfig(), "kokoro", "speed", 0.9)
    emptied = with_value(config, "kokoro", "speed", None)
    assert emptied.engines == {}
    path = save_config(emptied, tmp_path / "c.toml")
    assert path.read_text() == ""


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("this is = = not toml", "c.toml"),
        ("oops = 1", "unknown section"),
        ("read = 1", r"\[read\] must be a table"),
        ("[engines]\nkokoro = 3", r"\[engines.kokoro\] must be a table"),
    ],
)
def test_bad_files_raise(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "c.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_config_file_follows_xdg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/cfg")
    assert config_file() == Path("/tmp/cfg/velora/config.toml")
