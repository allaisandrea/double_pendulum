# Arduino motor control

An Arduino Uno drives the pendulum's DC motor through a Cytron SHIELD-MDD10,
taking a stream of duty commands from this laptop over USB serial. The
sketch is `arduino.ino`; Arduino requires it to share its folder's name.

## The shield

A shield plugs straight onto the Uno, so there is no signal wiring. Only
channel 1 is used, jumpered to **PWM1 = D9** and **DIR1 = D8**. Motor to
M1A/M1B, the 12 V supply to VB+/VB-.

- **No reverse-polarity protection** — the silkscreen says so, and the
  datasheet says a reversed supply destroys the board instantly. Check VB+
  and VB- before powering up.
- **The test buttons run the motor at full speed.** Duty 60 of 255 was
  already violent on this rig; do not press them with the arm attached.
- **It cannot coast.** PWM low shorts the motor terminals (brake), and there
  is no enable pin to float them. Idle, the motor damps the pendulum.
- **Its 5 V regulator feeds the Uno's 5V pin** by default, alongside USB.
  Cutting the 5V solder jumper on the underside separates them.
- **Rated to 20 kHz PWM** at full current (40 kHz derated). High frequencies
  are quiet where lower ones whine; `PWM_HZ` sets what the sketch uses.
- **Two diagnostic LEDs:** ERR lights on undervoltage shutdown, OC when the
  shield is limiting current.

**Measured 2026-09-21, duty 30 at 20 kHz:** positive duty turns the shaft
**counter-clockwise**, negative **clockwise**. Both directions work and both
are very quiet.

## Building and uploading

`arduino-cli` lives in `~/.local/bin` — Homebrew could not build it, because
the formula wants a newer Command Line Tools than macOS 13 has, so this is
Arduino's official prebuilt binary.

```sh
export PATH="$HOME/.local/bin:$PATH"
arduino-cli compile --fqbn arduino:avr:uno arduino
arduino-cli upload -p /dev/cu.usbmodem143201 --fqbn arduino:avr:uno arduino
```

The port name can change; `arduino-cli board list` shows it.

## Testing

`drive.py` applies one duty for a while, then stops the motor. It finds the
Uno by USB vendor id, waits for the ready line, resends the byte at 50 Hz to
stay ahead of the watchdog, and streams zeros at the end; Ctrl-C stops it at
once.

```sh
uv run arduino/drive.py 30 3      # +30 for 3 s: counter-clockwise
uv run arduino/drive.py -30 3     # clockwise
uv run arduino/drive.py 0 1       # moves nothing: checks the link
```

**Do not open a serial monitor on this sketch.** Every byte is a motor
command, so typing moves the motor: `?` is byte 63, which is duty 63.

## Protocol

115200 baud. **Each byte the host sends is one command**: a signed 8-bit
duty (`int8_t`), in units of 1/255 of full scale. The sketch clips it to
±`LIMIT` and applies it at once. Positive turns the shaft counter-clockwise.

- **No framing.** Every byte is a complete, valid command, so the two ends
  cannot fall out of step.
- **No ramp and no pause at reversal.** What is sent is what the motor gets.
  The shield's current limiting absorbs the step.
- **No replies**, with one exception: `ready mdd10` once at startup.
- **Watchdog.** `WATCHDOG_MS` after the last byte, the duty goes to zero.
  To keep the motor driven, resend at least that often, repeating the same
  value if it has not changed.
- **Frequency, cap and watchdog are constants** in the sketch, `PWM_HZ`,
  `LIMIT` and `WATCHDOG_MS`, and `arduino.ino` is the one place their values
  are set. Changing any of them means reflashing.

The byte range reaches ±127, i.e. 50% duty, so `LIMIT` cannot exceed that.

### Timing

A byte is applied as soon as the serial interrupt receives it, and the new
duty reaches the pin at the top of the next PWM period, at most 1/`PWM_HZ`
later.
The Uno's clock is not involved, so **an action takes effect at the host's
write time plus the one-way transfer**: 87 µs for the byte on the wire, plus
USB scheduling of up to about a millisecond. Timestamp writes on the host's
monotonic clock and nothing else needs synchronising.

That is why the protocol has no replies. An earlier text protocol answered
every command with a 57-byte status line, which measured a round trip of
5.9–10.5 ms (median 8.3) — mostly that reply — and the Uno's clock ran at
−422 ppm against the laptop's, 25 ms per minute.

### Rules for the host

- **Wait for `ready mdd10` after opening the port.** Opening it resets the
  Uno. Bytes sent during the ~1.5 s before that line go to the bootloader,
  which may act on them.
- **Only the controlling program may talk to this port.** Any byte moves the
  motor, including text; there is no arm step to guard against it.
- **Keep sending.** Silence longer than `WATCHDOG_MS` stops the motor, by
  design: a crashed program, an unplugged cable and a wedged laptop all look
  the same from the board.

## Safety behaviour in the sketch

- Duty is capped at ±`LIMIT`. Duty 60 felt violent with the previous
  driver, so approach the cap gradually.
- The PWM, DIR and channel-2 inputs are written low before they become
  outputs. The pins still float during the ~2 s reset whenever the port is
  opened, and no sketch can prevent that; keep clear of the arm then.
- PWM comes from Timer1 in phase-correct mode with `ICR1` as the top value:
  f = 16 MHz / (2 × top), so top = 8 MHz / `PWM_HZ`. Timer0 is untouched, so
  `millis()` still works. Do not call `analogWrite()` on pin 9.

## Open questions

- **Where is the dead zone?** Pulse width is (duty / 255) / `PWM_HZ`, so
  small duties make short pulses. Duty 30 at 20 kHz, a ~6 µs pulse, worked in
  the 2026-09-21 test, but below some duty the shield will not switch fully —
  and below some other, the motor's static friction wins anyway. A slow sweep
  up from 1 would find both.
- **Does the arm still stall?** With the previous driver, duty 30 moved the
  arm partway and then stopped it, humming, after which it fell back on its
  own. Supply sag, driver protection and a dead spot in the motor were all
  candidates, and none was ruled out. If it recurs, the ERR and OC LEDs
  separate the first two; stopping at the same angle every time would point
  at the motor.

The previous driver, a BTS7960, turned the motor one way only and whined at
every frequency it could follow. It and its sketches are in git history.
