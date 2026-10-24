from __future__ import annotations

import json
import re
from pathlib import Path

from kiln.link.runner import load_runner_config


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "examples" / "hardware" / "rp2040"


def test_rp2040_adapter_has_a_fail_closed_build_contract() -> None:
    cmake = (ADAPTER / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "PICO_SDK_PATH must point to a pinned Pico SDK checkout" in cmake
    assert "KILN_GENERATED_DIR must contain model.c and budget_certificate.txt" in cmake
    assert "KILN_ARENA_BYTES" in cmake
    assert "pico_add_extra_outputs(kiln_rp2040)" in cmake

    main = (ADAPTER / "main.c").read_text(encoding="utf-8")
    assert "KILN_METRIC cycles=" in main
    assert "__flash_binary_end" in main
    assert "[0 ..." not in main  # keep the harness valid under -std=c11 -pedantic


def test_rp2040_runner_config_is_a_valid_argv_contract() -> None:
    config_path = ADAPTER / "runner.example.json"
    config = load_runner_config(config_path)
    assert config.name == "rp2040-pico-usb-example"
    assert "{exe}" in config.flash
    assert config.run[:3] == ("python", "scripts/capture_serial.py", "--port")


def test_rp2040_cmake_parser_matches_the_checked_in_certificate() -> None:
    certificate = (ROOT / "examples" / "hazard" / "build" / "budget_certificate.txt").read_text(
        encoding="utf-8"
    )
    match = re.search(r"Arena \(\.bss\) RAM\s*:\s*([0-9,]+) bytes", certificate)
    assert match is not None
    assert int(match.group(1).replace(",", "")) == 1344

    payload = json.loads((ADAPTER / "runner.example.json").read_text(encoding="utf-8"))
    assert payload["timeout_seconds"] > 0
