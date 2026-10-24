# RP2040 reference-board adapter

This directory is the first board adapter for the Kiln hardware evidence
contract. It uses the Raspberry Pi Pico C/C++ SDK for startup, flash layout,
USB CDC stdio, and firmware linking. The generated Kiln model remains a plain
C translation unit and is not modified for the board.

The adapter is deliberately separate from `kiln link`: `kiln link` validates a
generic target image, while this project produces a board-bootable ELF/UF2
using the vendor SDK. Promotion to Stable requires the resulting image to be
flashed and replayed on a physical Pico.

## Build

Use a pinned Pico SDK checkout and a working ARM GNU toolchain. First generate
the model artifacts:

```bash
python -m kiln build examples/hazard/kiln.yaml \
  --output examples/hardware/rp2040/generated \
  --runtime-header None
```

Configure and build with CMake/Ninja:

```bash
cmake -S examples/hardware/rp2040 -B build/rp2040 -G Ninja \
  -DPICO_SDK_PATH=/path/to/pico-sdk \
  -DKILN_GENERATED_DIR="$PWD/examples/hardware/rp2040/generated"
cmake --build build/rp2040
```

Windows PowerShell uses the same commands with an absolute
`-DPICO_SDK_PATH=C:/path/to/pico-sdk` value.

The build emits `kiln_rp2040.elf` and `kiln_rp2040.uf2`. The CMake configure
step reads the audited arena size from `budget_certificate.txt` and passes it
to the firmware; it fails closed if the certificate is absent or malformed.

## Flash and capture

With `picotool` installed and a Pico connected:

```bash
picotool load -f build/rp2040/kiln_rp2040.elf
python scripts/capture_serial.py --port /dev/ttyACM0 --baud 115200 --seconds 10
```

On Windows replace the port with the assigned `COM` port. `picotool` supports
explicit ELF loading; the generated UF2 remains available for drag-and-drop
BOOTSEL workflows.

The output must contain five `index prediction` lines followed by one strict
metric line. `peak_ram_bytes` is the audited Kiln arena plus the measured stack
high-water mark; `flash_bytes` is the linked image endpoint relative to the
RP2040 XIP base. These definitions are intentionally narrower than a whole
system memory budget and are recorded as such in the evidence.

## Evidence gate

Run the host reference first, save the board output, and record:

- Pico board revision and SDK commit;
- ARM compiler version and CMake generator;
- system clock frequency;
- exact ELF SHA-256;
- flash command and serial port;
- captured predictions and all `KILN_METRIC` fields.

The adapter is not considered a passing target until the predictions match the
host `kiln.ref` vectors and the captured evidence is attached to an
`execution_certificate.txt` produced by the runner workflow.
