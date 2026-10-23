from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

import kiln.link.runner as runner_module
from kiln.link.runner import (
    RunnerError,
    load_runner_config,
    parse_runtime_metrics,
    run_runner_config,
)


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "name": "test-board",
                "flash": ["flash-tool", "{exe}"],
                "run": ["run-tool", "--capture", "{exe}"],
                "timeout_seconds": 4,
            }
        ),
        encoding="utf-8",
    )


def test_runner_config_flashes_then_runs_without_a_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "runner.json"
    _write_config(config_path)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout="0 1\nKILN_METRIC cycles=123 peak_ram_bytes=456\n",
            stderr="",
        )

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    output = run_runner_config(load_runner_config(config_path), tmp_path / "model.elf")

    assert "KILN_METRIC cycles=123" in output
    assert [call[0] for call in calls] == [
        ["flash-tool", str(tmp_path / "model.elf")],
        ["run-tool", "--capture", str(tmp_path / "model.elf")],
    ]
    assert all(call[1]["timeout"] == 4.0 for call in calls)
    assert all(call[1]["check"] is False for call in calls)


def test_runner_config_rejects_unknown_placeholders(tmp_path: Path) -> None:
    path = tmp_path / "runner.json"
    path.write_text(
        json.dumps({"flash": ["flash", "{port}"], "run": ["run", "{exe}"]}),
        encoding="utf-8",
    )

    with pytest.raises(RunnerError, match="unsupported placeholder"):
        load_runner_config(path)


def test_runtime_metrics_are_strict_and_structured() -> None:
    metrics = parse_runtime_metrics(
        "0 4\nKILN_METRIC cycles=123 peak_ram_bytes=456 flash_bytes=789 stack_bytes=12\n"
    )

    assert metrics.cycles == 123
    assert metrics.peak_ram_bytes == 456
    assert metrics.flash_bytes == 789
    assert metrics.stack_bytes == 12
    assert metrics.emitted


def test_runtime_metrics_reject_bad_values() -> None:
    with pytest.raises(RunnerError, match="not an integer"):
        parse_runtime_metrics("KILN_METRIC cycles=fast\n")
    with pytest.raises(RunnerError, match="must not be negative"):
        parse_runtime_metrics("KILN_METRIC cycles=-1\n")
    with pytest.raises(RunnerError, match="unknown runtime metric"):
        parse_runtime_metrics("KILN_METRIC voltage=3\n")
