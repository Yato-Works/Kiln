"""Explicit runner configuration and runtime evidence parsing for ``kiln link``."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

__all__ = (
    "RunnerConfig",
    "RunnerError",
    "RuntimeMetrics",
    "load_runner_config",
    "parse_runtime_metrics",
    "run_runner_config",
)


class RunnerError(RuntimeError):
    """A malformed runner configuration or failed flash/run command."""


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Commands for flashing and capturing output from a real board.

    Commands are argv arrays rather than shell strings. ``{exe}`` is replaced
    with the linked artifact path, keeping board automation explicit and
    avoiding shell injection through a model or output directory name.
    """

    name: str
    flash: tuple[str, ...]
    run: tuple[str, ...]
    timeout_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    """Measurements optionally emitted by a hardware firmware harness."""

    cycles: int | None = None
    peak_ram_bytes: int | None = None
    flash_bytes: int | None = None
    stack_bytes: int | None = None

    @property
    def emitted(self) -> bool:
        return any(
            value is not None
            for value in (self.cycles, self.peak_ram_bytes, self.flash_bytes, self.stack_bytes)
        )


def _as_command(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise RunnerError(f"runner config '{field}' must be a non-empty string array")
    if any("{" in item and item.count("{exe}") != item.count("{") for item in value):
        raise RunnerError(f"runner config '{field}' contains an unsupported placeholder")
    return tuple(value)


def load_runner_config(path: Path) -> RunnerConfig:
    """Loads and validates a board runner JSON document."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RunnerError(f"cannot read runner config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RunnerError(f"runner config {path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise RunnerError("runner config must be a JSON object")
    name = payload.get("name", path.stem)
    if not isinstance(name, str) or not name.strip():
        raise RunnerError("runner config 'name' must be a non-empty string")
    timeout = payload.get("timeout_seconds", 30.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise RunnerError("runner config 'timeout_seconds' must be positive")
    return RunnerConfig(
        name=name,
        flash=_as_command(payload.get("flash"), "flash"),
        run=_as_command(payload.get("run"), "run"),
        timeout_seconds=float(timeout),
    )


def _expand(command: Sequence[str], executable: Path) -> list[str]:
    expanded = [item.replace("{exe}", str(executable)) for item in command]
    if not any("{exe}" in item for item in command):
        expanded.append(str(executable))
    return expanded


def _run_command(command: Sequence[str], *, timeout: float, what: str) -> str:
    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise RunnerError(f"{what} timed out after {timeout:g}s: {' '.join(command)}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RunnerError(f"{what} failed:\n  {' '.join(command)}\n{detail}")
    return result.stdout


def run_runner_config(config: RunnerConfig, executable: Path) -> str:
    """Flashes, runs, and captures stdout from a configured hardware target."""
    _run_command(_expand(config.flash, executable), timeout=config.timeout_seconds, what="flashing")
    return _run_command(
        _expand(config.run, executable), timeout=config.timeout_seconds, what="running hardware"
    )


_METRIC_KEYS = {"cycles", "peak_ram_bytes", "flash_bytes", "stack_bytes"}


def parse_runtime_metrics(stdout: str) -> RuntimeMetrics:
    """Parses strict ``KILN_METRIC key=value`` lines from runner output."""
    values: dict[str, int] = {}
    for line in stdout.splitlines():
        if not line.startswith("KILN_METRIC "):
            continue
        fields = line[len("KILN_METRIC ") :].split()
        for field in fields:
            if "=" not in field:
                raise RunnerError(f"malformed runtime metric: {line!r}")
            key, raw_value = field.split("=", 1)
            if key not in _METRIC_KEYS:
                raise RunnerError(f"unknown runtime metric '{key}'")
            if key in values:
                raise RunnerError(f"runtime metric '{key}' was emitted more than once")
            try:
                value = int(raw_value, 10)
            except ValueError as exc:
                raise RunnerError(
                    f"runtime metric '{key}' is not an integer: {raw_value!r}"
                ) from exc
            if value < 0:
                raise RunnerError(f"runtime metric '{key}' must not be negative")
            values[key] = value
    return RuntimeMetrics(**values)
