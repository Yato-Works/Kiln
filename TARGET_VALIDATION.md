# Target validation and runtime evidence

Kiln separates three kinds of evidence. A green host build is not evidence of
execution on a microcontroller, and a QEMU result is not evidence of the timing
or peripheral behaviour of a particular board.

| Evidence class | What it proves | What it does not prove |
| :--- | :--- | :--- |
| `host-process` | C semantics, harness parity, linked image accounting | MCU ABI, startup, peripheral behaviour, timing |
| `system-emulator` | target ISA execution, reset/startup image, linker map, deterministic output | flash wait states, clock tree, interrupts under load, board peripherals, power |
| `hardware-runner` | the flashed board produced the captured output and optional measurements | another board revision, another clock/configuration, unmeasured properties |

## QEMU system-emulator path

The `cortex-m3-qemu` profile boots a freestanding image for QEMU's
`mps2-an385` Cortex-M3 machine. Kiln supplies the vector table, reset handler,
memory map, and ARM semihosting output. It is intentionally separate from the
compile-only `cortex-m0plus`, `cortex-m4`, and `riscv32` profiles.

```bash
python -m kiln link examples/hazard/kiln.yaml \
  --target cortex-m3-qemu --run
```

The CI job installs `gcc-arm-none-eabi` and `qemu-system-arm` and runs this
command. QEMU output is used for deterministic harness replay; it does not
invent a cycle count.

## Hardware runner contract

Hardware is deliberately configured outside the compiler. A runner document is
JSON with argv arrays, never a shell command:

```json
{
  "name": "my-board-probe",
  "flash": ["probe-rs", "download", "{exe}", "--chip", "RP2040"],
  "run": ["probe-rs", "run", "--chip", "RP2040", "{exe}"],
  "timeout_seconds": 30
}
```

The `{exe}` placeholder is replaced with the linked ELF path. Kiln executes the
flash command first, then the run command, captures stdout, and parses the
normal harness lines (`index prediction`). This leaves the board tool, UART,
RTT, or debugger transport to the project owner while keeping the build
contract reproducible and auditable. The selected board profile/toolchain must
also provide a real reset/startup implementation and a `printf`/stdout path;
the runner is transport orchestration, not a pretend universal UART driver.

```bash
python -m kiln link examples/hazard/kiln.yaml \
  --target cortex-m4 \
  --runner-config examples/hardware/runner.example.json \
  --run
```

## Runtime measurement protocol

A hardware firmware may additionally emit one line using the following strict
protocol:

```text
KILN_METRIC cycles=12345 peak_ram_bytes=4096 flash_bytes=6912 stack_bytes=512
```

All fields are optional, non-negative decimal integers. Kiln writes the parsed
observations to `execution_certificate.txt`. Missing values remain
`not emitted`; static estimates are never relabelled as measurements.
