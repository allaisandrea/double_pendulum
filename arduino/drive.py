# /// script
# requires-python = ">=3.11"
# dependencies = ["pyserial>=3.5"]
# ///
"""Applies one duty command to the motor for a while, then stops it.

    uv run arduino/drive.py 30 3        # +30 for 3 s: counter-clockwise
    uv run arduino/drive.py -30 3       # clockwise
    uv run arduino/drive.py 0 1         # moves nothing: checks the link

The duty is the sketch's int8 command, in 1/255 of full scale; the board
clips it to its own limit. The byte is resent at --rate, because the sketch
zeroes the duty shortly after the last one, and zeros are streamed at the
end.
Ctrl-C stops the motor at once.
"""
import argparse
import sys
import time

import serial
from serial.tools import list_ports

ARDUINO_VID = 0x2341
READY = b"ready mdd10"


def find_port():
    """The Uno's port. Its name changes with the USB socket, so it is found
    by Arduino's vendor id rather than hard-coded."""
    found = [p.device for p in list_ports.comports() if p.vid == ARDUINO_VID]
    if not found:
        sys.exit("no Arduino found; plug it in, or pass --port")
    if len(found) > 1:
        sys.exit(f"several Arduinos found ({', '.join(found)}); pass --port")
    return found[0]


def wait_ready(port, timeout=5.0):
    """Opening the port resets the Uno; bytes sent before its ready line go
    to the bootloader, so nothing is written until the line arrives."""
    start, seen = time.monotonic(), b""
    while READY not in seen:
        if time.monotonic() - start > timeout:
            sys.exit(f"no '{READY.decode()}' after {timeout:.0f} s; got {seen!r}. "
                     "Is the arduino/ sketch flashed?")
        seen += port.read(64)
    return time.monotonic() - start


def stream(port, duty, seconds, period):
    """Writes the duty every period for the given time."""
    byte = duty.to_bytes(1, "big", signed=True)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        port.write(byte)
        time.sleep(period)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("duty", type=int, help="signed duty, -128..127 (the board clips it to its limit)")
    ap.add_argument("seconds", type=float, help="how long to apply it")
    ap.add_argument("--port", help="serial port [default: the Arduino found by USB vendor id]")
    ap.add_argument("--rate", type=float, default=50.0, help="resend rate in Hz [default: 50]")
    args = ap.parse_args()

    if not -128 <= args.duty <= 127:
        ap.error("duty must fit in an int8: -128..127")

    device = args.port or find_port()
    with serial.Serial(device, 115200, timeout=0.05) as port:
        print(f"{device}: ready after {wait_ready(port):.2f} s")
        interrupted = False
        try:
            stream(port, args.duty, args.seconds, 1 / args.rate)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            # Explicit zeros rather than waiting out the watchdog.
            stream(port, 0, 0.1, 0.02)
            port.flush()
        extra = port.read(256)
    what = "interrupted" if interrupted else f"applied {args.duty:+d} for {args.seconds:g} s"
    print(f"{what}; motor stopped")
    if extra:
        print(f"unexpected bytes from the board: {extra!r}")


if __name__ == "__main__":
    main()
