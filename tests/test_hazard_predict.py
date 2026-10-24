"""Tests for the torch-free Hazard text-to-VM user path."""

from __future__ import annotations

import json
from io import StringIO

from examples.hazard.predict import main


def test_predict_text_returns_a_named_label_and_vm_metrics(capsys) -> None:
    assert main(["--text", "print(eval(user_input))", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "ok"
    assert payload["label_id"] == 1
    assert payload["label"] == "dynamic_exec"
    assert payload["token_count"] == 96
    assert payload["metrics"]["allocation_bytes"] == 0


def test_predict_stdin_is_deterministic(capsys, monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", StringIO("print(eval(user_input))"))

    assert main(["--stdin"]) == 0
    first = capsys.readouterr().out

    monkeypatch.setattr("sys.stdin", StringIO("print(eval(user_input))"))
    assert main(["--stdin"]) == 0
    second = capsys.readouterr().out

    assert first == second
    assert "prediction: dynamic_exec (1)" in first
