from __future__ import annotations

from kiln.cli import main


def test_doctor_reports_core_workflow(capsys) -> None:
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "Python" in output
    assert "Kiln core workflow is available" in output
