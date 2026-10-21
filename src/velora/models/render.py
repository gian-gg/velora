"""Plain-text rendering of engine readiness and voice lists."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

from velora.engines.base import Engine, WeightsStatus, install_hint, package_installed
from velora.engines.settings import Setting

HEADERS = ("setting", "value", "default", "allowed")


@dataclass(frozen=True)
class EngineRow:
    name: str
    installed: bool
    weights: WeightsStatus
    hint: str


def engine_row(engine: Engine) -> EngineRow:
    return EngineRow(
        engine.name, package_installed(engine), engine.weights_status(), install_hint(engine)
    )


def human_size(size_bytes: int) -> str:
    return f"{size_bytes / 1_000_000:.0f} MB"


def short_path(path: Path) -> str:
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def _weights_cell(row: EngineRow) -> str:
    if not row.weights.downloaded:
        return f"not downloaded → velora models download {row.name}"
    where = f"  {short_path(row.weights.path)}" if row.weights.path else ""
    return f"downloaded  {human_size(row.weights.size_bytes)}{where}"


def render_engines(rows: list[EngineRow]) -> str:
    package_cells = ["installed" if r.installed else f"missing → {r.hint}" for r in rows]
    name_w = max(len("engine"), *(len(r.name) for r in rows))
    pkg_w = max(len("package"), *(len(c) for c in package_cells))
    lines = [f"{'engine':<{name_w}}  {'package':<{pkg_w}}  weights"]
    lines += [
        f"{r.name:<{name_w}}  {cell:<{pkg_w}}  {_weights_cell(r)}"
        for r, cell in zip(rows, package_cells, strict=True)
    ]
    return "\n".join(lines)


def _prefix(voice: str) -> str:
    return voice.partition("_")[0] if "_" in voice else ""


def render_voices(engine_name: str, voices: list[str]) -> str:
    if not voices:
        return f"{engine_name} has no preset voices."
    ordered = sorted(voices, key=lambda v: (_prefix(v), v))
    width = max(len(_prefix(v)) for v in voices)
    lines = [
        f"{prefix:<{width}}  " + " ".join(v.partition("_")[2] if prefix else v for v in names)
        for prefix, names in groupby(ordered, key=_prefix)
    ]
    return "\n".join([f"{engine_name} voices ({len(voices)}), as PREFIX_NAME:", "", *lines])


def show_value(value: object) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _setting_row(setting: Setting, saved: Mapping[str, object]) -> tuple[str, ...]:
    value = show_value(saved.get(setting.name, setting.default))
    marker = " *" if setting.name in saved else ""
    return (setting.name, value + marker, show_value(setting.default), setting.allowed())


def render_settings(
    groups: list[tuple[str, tuple[Setting, ...], Mapping[str, object]]], source: Path
) -> str:
    """Render (heading, settings, saved values) groups as one aligned table."""
    blocks = [
        (heading, [_setting_row(s, saved) for s in settings]) for heading, settings, saved in groups
    ]
    rows = [row for _, block in blocks for row in block]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(HEADERS)]

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)).rstrip()

    lines = [line(HEADERS)]
    for heading, block in blocks:
        lines += ["", heading, *(line(r) for r in block)]
    lines += ["", f"* saved in {short_path(source)}"]
    return "\n".join(lines)
