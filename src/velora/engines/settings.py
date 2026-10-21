"""Declared, validated settings for engines and for reading, plus layered resolution."""

from collections.abc import Mapping
from dataclasses import dataclass

from velora.engines.base import EngineError

Value = str | float


@dataclass(frozen=True)
class Setting:
    name: str
    default: Value
    help: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()

    @property
    def numeric(self) -> bool:
        return isinstance(self.default, float)

    def allowed(self) -> str:
        if self.choices:
            return ", ".join(self.choices)
        if self.numeric:
            return f"{self.minimum:g} to {self.maximum:g}"
        return self.help

    def coerce(self, raw: object) -> Value:
        """Convert a CLI string or TOML value to this setting's type, enforcing its limits."""
        if self.numeric:
            return self._number(raw)
        if not isinstance(raw, str) or not raw.strip():
            raise EngineError(f"{self.name} must be text")
        value = raw.strip()
        if self.choices and value not in self.choices:
            raise EngineError(f"{self.name} must be one of {self.allowed()}, got {value!r}")
        return value

    def _number(self, raw: object) -> float:
        if isinstance(raw, bool) or not isinstance(raw, int | float | str):
            raise EngineError(f"{self.name} must be a number")
        try:
            value = float(raw)
        except ValueError as exc:
            raise EngineError(f"{self.name} must be a number, got {raw!r}") from exc
        low = self.minimum if self.minimum is not None else value
        high = self.maximum if self.maximum is not None else value
        if not low <= value <= high:
            raise EngineError(f"{self.name} must be between {low:g} and {high:g}, got {value:g}")
        return value


def find_setting(settings: tuple[Setting, ...], name: str) -> Setting | None:
    return next((s for s in settings if s.name == name), None)


def resolve(
    settings: tuple[Setting, ...],
    layers: list[tuple[str, Mapping[str, object]]],
) -> dict[str, Value]:
    """Merge (label, values) layers over the defaults; later layers win."""
    names = [s.name for s in settings]
    resolved: dict[str, Value] = {s.name: s.default for s in settings}
    for label, values in layers:
        unknown = sorted(set(values) - set(names))
        if unknown:
            raise EngineError(
                f"{label}: unknown setting(s) {', '.join(unknown)}; allowed: {', '.join(names)}"
            )
        for setting in settings:
            if setting.name in values:
                try:
                    resolved[setting.name] = setting.coerce(values[setting.name])
                except EngineError as exc:
                    raise EngineError(f"{label}: {exc}") from exc
    return resolved
