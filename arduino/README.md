# Arduino motor control

An Arduino Uno drives the pendulum's DC motor through a BTS7960 H-bridge,
taking commands from this laptop over USB serial. Eventually the tag poses
from `apriltag-cam` will feed a control loop that sends those commands.

## Sketches

| sketch | purpose |
| --- | --- |
| `motor/` | the real thing: serial commands to duty cycle, with a watchdog |
| `hello/` | serial-only sanity check; touches no pins, so it cannot move the motor |
| `probe/` | reports the state of the four control lines, to find wiring faults |

## Wiring

| Uno pin | BTS7960 | notes |
| --- | --- | --- |
| 9 | RPWM | PWM, Timer1 |
| 10 | LPWM | PWM, Timer1 |
| 7 | R_EN | |
| 8 | L_EN | |
| GND | GND | required; without it the driver has no reference for the signals |

**Sign convention, measured 2026-09-11:** positive duty drives RPWM and turns
the shaft **counter-clockwise**.

## Building and uploading

`arduino-cli` lives in `~/.local/bin` — Homebrew could not build it, because
the formula wants a newer Command Line Tools than macOS 13 has, so this is
Arduino's official prebuilt binary.

```sh
export PATH="$HOME/.local/bin:$PATH"
arduino-cli compile --fqbn arduino:avr:uno arduino/motor
arduino-cli upload -p /dev/cu.usbmodem143101 --fqbn arduino:avr:uno arduino/motor
arduino-cli monitor -p /dev/cu.usbmodem143101 -c baudrate=115200
```

The port name can change; `arduino-cli board list` shows it.

## Serial protocol

115200 baud, one command per line. Every command answers with a line starting
`ok`, `err` or `watchdog`, carrying `en`, `duty`, `target`, `limit` and the
board's millisecond clock.

| command | meaning |
| --- | --- |
| `e 1` / `e 0` | enable / disable both halves of the bridge |
| `v <-255..255>` | signed duty, clamped to the current limit |
| `s` | duty 0, bridge still enabled (shorts the motor, so it stops quickly) |
| `l <0..255>` | cap on \|duty\| |
| `t 1` / `t 0` | stream status at 10 Hz |
| `?` | one status line; also serves as a heartbeat |

## Rules a host program must follow

- **Send something at least every 300 ms.** After that silence the sketch
  stops the motor and disables the bridge. A crashed control loop, an
  unplugged cable and a wedged laptop all look the same from the board, and
  all of them should stop the motor. `?` is the cheapest heartbeat.
- **Wait about two seconds after opening the port.** Opening it resets the
  Uno, and the bootloader swallows input while it runs.
- **`v` is refused unless enabled.** Send `e 1` first.

## Safety behaviour in the sketch

- Duty is capped at 40 of 255 by default; 60 was already violent on this rig.
- Duty slews about one count per millisecond, so no step lands on the gearbox.
- A direction change passes through zero and pauses 5 ms there.
- Enables come up low and the PWM pins are written low before they become
  outputs, so a reset cannot leave a level on the driver's inputs.
- PWM runs at ~3.9 kHz (Timer1 prescaler /8): above most of the audible
  whine, and well under the BTS7960's 25 kHz ceiling. Timer0 is untouched, so
  `millis()` and `delay()` still work.

## Known hardware fault, 2026-09-11

**The driver's LPWM input does not work, so the motor turns one way only.**
Tested by swapping the pin 9 and pin 10 wires at the Arduino header and
repeating a gentle 30-duty run in each direction:

| test | signal path | result |
| --- | --- | --- |
| pin 9 → RPWM | original wiring | turns counter-clockwise |
| pin 10 → LPWM | original wiring | nothing |
| pin 9 → LPWM | wires swapped | nothing |
| pin 10 → RPWM | wires swapped | turns counter-clockwise |

The motor turns whenever a signal reaches RPWM and never when it reaches
LPWM, whichever Uno pin sends it, so both outputs are good. `probe` found all
four lines identical: pulled low through the driver's input pull-downs, and
free of shorts. `L_EN` is proven good too, because forward current returns
through that chip's low side — the motor could not turn forward at all if
that chip were disabled.

That leaves the LPWM connection at the module end, or the left chip's high
side. Reseat the LPWM wire at the module first; if it is definitely on the
right pin and still dead, the module has a failed half and needs replacing.
Keep the same pin-out and nothing here changes.

**The wires may still be swapped** from that test. Put pin 9 back on RPWM and
pin 10 back on LPWM before trusting the sign convention above.
