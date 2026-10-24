# Hardware runner examples

## Reference board direction

Kiln's first board-level reference target is the **Raspberry Pi Pico
(RP2040)**. It is inexpensive, widely available to students, and uses the
Cortex-M0+ class already represented by the `cortex-m0plus` cross-compile
profile.

The repository currently contains the runner contract and the evidence format,
but it does **not** claim that an RP2040 image boots on real hardware yet. The
generic Cortex-M compile profile intentionally has no vendor startup, flash
layout, USB/UART transport, or Pico SDK dependency. Those pieces belong in the
board adapter and must be verified with a physical board before this target is
promoted to Stable.

The board acceptance gate is:

1. build from a clean checkout with a pinned Pico SDK and
   `arm-none-eabi-gcc`;
2. flash the exact linked artifact with the documented command;
3. capture all five deterministic harness predictions after reset;
4. emit at least `KILN_METRIC cycles=... peak_ram_bytes=...` from firmware;
5. write `execution_certificate.txt` and retain the toolchain, board revision,
   SDK revision, clock, and capture command next to the evidence.

Until all five gates pass, the board remains Experimental and the static cycle
estimate must not be described as a measured latency.

The concrete adapter workflow is documented in
[`rp2040/README.md`](rp2040/README.md).

Copy `runner.example.json` and replace the commands with the flashing and
capture tool used by your board. The commands must be argv arrays and must
contain `{exe}` where the linked ELF should be passed.

For a UART-connected board, Kiln includes a small capture helper. Install the
optional dependency with `python -m pip install -e ".[hardware]"` and use it
as the `run` command after adapting the flash command to your board SDK:

```json
{
  "name": "rp2040-uart-example",
  "flash": ["your-board-flash-tool", "{exe}"],
  "run": ["python", "scripts/capture_serial.py", "--port", "COM7", "--baud", "115200"],
  "timeout_seconds": 30
}
```

The helper is transport-only: it does not claim that an ELF is valid for a
particular MCU, and it does not synthesize missing metrics.

The board firmware must print the same five `index prediction` lines as the
generated harness. Optional measurements use the `KILN_METRIC key=value`
protocol documented in [TARGET_VALIDATION.md](../../TARGET_VALIDATION.md).
