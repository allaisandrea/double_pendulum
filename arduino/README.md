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
| `f <100..25000>` | PWM frequency in Hz; 10000 at startup (see below) |
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
- PWM comes from Timer1 in phase-correct mode with `ICR1` as the top value, so
  `f` can pick any frequency from 100 Hz to 25 kHz, the driver's ceiling. It
  starts at 10 kHz. Timer0 is untouched, so `millis()` and `delay()` still
  work. Do not call `analogWrite()` on pins 9 or 10: it assumes an 8-bit top
  and would fight this configuration.

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

Reseating the LPWM wire at the module changed nothing. One more clue: the
motor whines audibly whenever current flows, and reverse is **silent**. A
stalled motor is louder than a turning one, not quieter, so no current reaches
the motor in reverse at all.

Two explanations survive, and on a new module the cheaper one is likelier:

1. **The LPWM wire is one pin off on the module's header.** The header usually
   runs `RPWM · LPWM · R_EN · L_EN · R_IS · L_IS · VCC · GND`. If that wire
   sits on `R_IS` or `L_IS` — current-sense outputs, each tied to ground
   through a resistor — every observation above follows from a *healthy*
   module: `probe` sees the sense resistor pulling the line low and the
   Arduino easily driving it high, the real LPWM input floats low on its
   internal pull-down, forward needs exactly that low to return its current,
   and reverse is never driven at all. **Check this first**, counting pin
   positions against the silkscreen rather than trusting wire colours.
2. **The left chip's high side is damaged.** Possible, but a half-dead module
   straight out of the packet is only a few percent likely.

Keep the same pin-out either way and nothing here changes.

## Load behaviour, 2026-09-11

At duty 30 the arm starts moving, then stops partway while still humming, and
afterwards falls back to rest on its own — so it is not jammed against
anything. Duty 60 moves it briskly, and felt violent on the rig.

**Why it stops is unresolved.** A simple torque limit does not fit: a motor
short of torque creeps rather than stopping dead, and the arm swings back
freely. Candidates worth testing, in order:

- the supply sagging under load — the BTS7960 cuts out below about 5 V
- the surviving driver half tripping its own current or thermal protection
- a dead spot in the motor: an open winding or a burnt commutator segment
  would stop the rotor at one particular angle

To separate them, watch the supply's voltage and current during a run, and
note whether the arm stops at the same angle every time. The same angle points
at the motor; a different angle each run points at the supply or the driver.

The hum itself is not a fault and not an alarm: the windings vibrate at the
PWM switching frequency, so its pitch follows `f`. A stall makes the same
sound while the shaft stays put, and heats the motor and the bridge, so do
not hold one.

### PWM frequency, measured at duty 30

| frequency | pulse width | result |
| --- | --- | --- |
| 490 Hz | ~240 µs | moves; painfully loud, a coarse buzz where the ear is sharpest |
| 3.9 kHz | ~30 µs | moves; a clear whine |
| **10 kHz** | ~12 µs | **moves; higher pitched and noticeably quieter — the default** |
| 20 kHz | ~6 µs | **nothing happens at all** |

The pattern is the BTS7960's switching delays, which are a few microseconds.
Once the pulse approaches them the output never fully turns on, which is why
20 kHz does nothing. Lower frequencies chop more coarsely, so the current
ripple is larger — that is what makes 490 Hz the best bet for breaking a
stubborn load free, and also what makes it so loud.

**The trade-off to watch:** pulse width is duty × period, so a *small* duty at
10 kHz is a short pulse too. Duty 15 at 10 kHz is already down at ~6 µs, where
20 kHz failed. Expect a dead zone near zero duty, and if a control loop needs
fine authority around zero, drop the frequency rather than fight it.

When 20 kHz first failed it looked as though a Timer1 rewrite had broken
something. Flashing the committed pre-rewrite sketch moved the arm, and the
rewritten one at the same frequency and duty moved it identically — the
frequency was the cause, not the rewrite.

With only one working direction, the arm cannot be driven back electrically
after a stall — reposition it by hand before the next run.
