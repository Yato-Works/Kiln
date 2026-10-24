from __future__ import annotations

import json

from kiln.cli import main


def test_doctor_reports_core_workflow(capsys) -> None:
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "Python" in output
    assert "Kiln core and VM workflow are available" in output


def test_doctor_json_distinguishes_required_and_optional_tools(capsys) -> None:
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "ready"
    assert {check["category"] for check in payload["checks"]} >= {
        "required",
        "frontend",
        "target",
    }
    assert all("available" in check for check in payload["checks"])


def test_selftest_is_board_free_and_machine_readable(capsys) -> None:
    assert main(["selftest", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "passed"
    assert payload["outputs"] == {"label": [1]}
    assert payload["metrics"]["allocation_bytes"] == 0
    assert "reference parity" in payload["checks"]
