"""The user's saved settings in $XDG_CONFIG_HOME/velora/config.toml."""

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from velora.config import config_file

SECTIONS = ("read", "engines")


class ConfigError(Exception):
    """The config file is unreadable or has the wrong shape."""


@dataclass(frozen=True)
class UserConfig:
    read: Mapping[str, object] = field(default_factory=dict)
    engines: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def engine(self, name: str) -> Mapping[str, object]:
        return self.engines.get(name, {})


def _table(value: object, where: str, path: Path) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: [{where}] must be a table")
    return dict(value)


def load_config(path: Path | None = None) -> UserConfig:
    path = path or config_file()
    if not path.is_file():
        return UserConfig()
    try:
        data = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    unknown = sorted(set(data) - set(SECTIONS))
    if unknown:
        raise ConfigError(f"{path}: unknown section(s) {', '.join(unknown)}")
    engines = _table(data.get("engines", {}), "engines", path)
    return UserConfig(
        read=_table(data.get("read", {}), "read", path),
        engines={name: _table(t, f"engines.{name}", path) for name, t in engines.items()},
    )


def with_value(config: UserConfig, section: str, key: str, value: object | None) -> UserConfig:
    """Return a copy with section ("read" or an engine name) key set, or removed when None."""
    table = dict(config.read if section == "read" else config.engine(section))
    if value is None:
        table.pop(key, None)
    else:
        table[key] = value
    if section == "read":
        return UserConfig(read=table, engines=config.engines)
    engines = {**config.engines, section: table}
    return UserConfig(read=config.read, engines={k: v for k, v in engines.items() if v})


def _toml_table(header: str, values: Mapping[str, object]) -> str:
    lines = [f"[{header}]", *(f"{k} = {json.dumps(v)}" for k, v in sorted(values.items()))]
    return "\n".join(lines)


def save_config(config: UserConfig, path: Path | None = None) -> Path:
    path = path or config_file()
    tables = [_toml_table("read", config.read)] if config.read else []
    tables += [_toml_table(f"engines.{n}", t) for n, t in sorted(config.engines.items()) if t]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(tables) + "\n" if tables else "")
    return path
