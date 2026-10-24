"""Capture a Kiln board harness over a serial port.

The board runner invokes this script as an argv command. It deliberately does
not flash firmware: flashing belongs to the preceding runner command, while
this process only captures the board's reset-time output and preserves metric
lines for ``kiln link`` to parse.
"""

from __future__ import annotations

import argparse
import sys
import time


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="serial device, e.g. COM7 or /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=10.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.baud <= 0 or args.seconds <= 0:
        _parser().error("--baud and --seconds must be positive")
    try:
        import serial
    except ImportError:
        print(
            "serial capture requires pyserial; install with python -m pip install -e '.[hardware]'",
            file=sys.stderr,
        )
        return 2

    deadline = time.monotonic() + args.seconds
    try:
        with serial.Serial(args.port, args.baud, timeout=0.25) as device:
            while time.monotonic() < deadline:
                line = device.readline()
                if line:
                    sys.stdout.write(line.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
    except serial.SerialException as exc:
        print(f"serial capture failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
