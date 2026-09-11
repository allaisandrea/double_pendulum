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
| `f 1` / `f 2` / `f 3` | PWM frequency: ~3.9 kHz (default), ~490 Hz, ~20 kHz (see below) |
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
- PWM comes from Timer1 in phase-correct mode with `ICR1` as the top value,
  so `f` can pick the frequency: ~3.9 kHz by default, ~490 Hz for a coarser
  chop with more current ripple, or ~20 kHz to get above hearing while staying
  under the BTS7960's 25 kHz ceiling. Timer0 is untouched, so `millis()` and
  `delay()` still work. Do not call `analogWrite()` on pins 9 or 10: it
  assumes an 8-bit top and would fight this configuration.

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

Reseating the LPWM wire at the module changed nothing: with the wires back in
their documented positions, a gentle run turned the shaft counter-clockwise
as before and reverse stayed dead. A further clue settles it — the motor
whines audibly whenever current flows, and reverse is **silent**. A stalled
motor is louder than a turning one, not quieter, so no current reaches the
motor in reverse at all. The module's left half has failed and wants
replacing; keep the same pin-out and nothing here changes.

## Load behaviour, 2026-09-11

At duty 30 the arm starts moving and then stalls partway, humming. Duty 60
moves it briskly, and felt violent on the rig. So the usable duty for this
load sits between the two, and the figure is worth measuring properly once the
driver is replaced and both directions work.

The hum is not a fault and not an alarm: the windings vibrate at the PWM
switching frequency, 3.9 kHz, which the ear hears easily. A stall makes the
same sound while the shaft stays put, and heats the motor and the bridge,
so do not hold one.

### PWM frequency, measured

- **~3.9 kHz (`f 1`, default):** works. Audible whine, and the arm moves at
  duty 30.
- **~490 Hz (`f 2`):** works, and is very loud — a coarse buzz where the ear
  is sharpest. Its bigger current ripple is the best bet for breaking a
  stubborn load free, if you can stand the noise.
- **~20 kHz (`f 3`):** inaudible, but **moved nothing at duty 30**. The
  BTS7960's switching delays are several microseconds, and duty 30 at 20 kHz
  is a pulse of roughly 6 µs, so the output never fully turns on. Expect it to
  need a much higher duty before it does anything, and test it before relying
  on it.

Confirmed by flashing the committed pre-frequency sketch, which moved the arm
at duty 30, then the rewritten one at the same frequency and duty, which moved
it identically. The frequency is what mattered, not the rewrite.

With only one working direction, the arm cannot be driven back electrically
after a stall — reposition it by hand before the next run.
