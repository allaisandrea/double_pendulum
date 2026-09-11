// Drives a brushed DC motor through a BTS7960 H-bridge, taking commands over
// USB serial. Written for the pendulum rig: a laptop-side control loop sends
// a signed duty cycle at a steady rate, and the motor must coast to a stop by
// itself the moment that loop stops talking.
//
// Commands, one per line, at 115200 baud:
//   e 1 | e 0     enable / disable both halves of the bridge
//   v <-255..255> signed duty; negative is the other direction
//   s             duty 0, bridge left enabled (an active brake: both low
//                 sides on, so the motor is shorted and stops quickly)
//   l <0..255>    cap on |duty|, applied to this and later commands
//   t 1 | t 0     stream status at 10 Hz
//   f 1 | f 2 | f 3  PWM frequency: ~3.9 kHz (default), ~490 Hz, ~20 kHz
//   ?             one status line
// Every command answers with a line beginning "ok" or "err".

const uint8_t PIN_RPWM = 9;   // forward PWM, Timer1
const uint8_t PIN_LPWM = 10;  // reverse PWM, Timer1
const uint8_t PIN_R_EN = 7;
const uint8_t PIN_L_EN = 8;

// With no command for this long, stop and disable. The laptop crashing, the
// USB cable coming loose and the control loop stalling all look the same from
// here, and all of them should stop the motor.
const unsigned long WATCHDOG_MS = 300;

// Ramp limit, in duty counts per millisecond: full scale takes ~0.25 s. Keeps
// the current surge of a step command off the supply and the gearbox.
const int SLEW_PER_MS = 1;

// Direction reversals wait this long at zero duty. The BTS7960 has its own
// shoot-through protection, but crossing zero slowly also spares the motor.
const unsigned long REVERSE_PAUSE_MS = 5;

int target = 0;          // duty asked for, after clamping
int applied = 0;         // duty actually on the pins, ramped towards target
int limit = 40;          // 60 was already violent on this rig; raise with 'l'
bool enabled = false;
bool streaming = false;
unsigned long last_command_ms = 0;
unsigned long last_update_ms = 0;
unsigned long last_status_ms = 0;
unsigned long reverse_until_ms = 0;
bool watchdog_tripped = false;

uint8_t pwm_opt = 1;
uint16_t pwm_top = 256;  // Timer1 counts to this; sets frequency and resolution

void set_pins(int duty) {
  // Only ever one side driven; the other is held at zero. Duty arrives as
  // 0..255 on the wire and is scaled to whatever top the frequency uses.
  OCR1A = duty > 0 ? ((uint32_t)duty * pwm_top) / 255 : 0;   // pin 9, RPWM
  OCR1B = duty < 0 ? ((uint32_t)-duty * pwm_top) / 255 : 0;  // pin 10, LPWM
}

// Timer1 runs in phase-correct PWM with ICR1 as the top value, so both the
// frequency and the resolution are ours: f = 16 MHz / (2 * prescaler * top).
//   1  ~3.9 kHz  default; an audible whine, but easy on the driver
//   2   ~490 Hz  coarse chopping, so more current ripple. A bigger ripple can
//                break a load free that a smooth current only hums against,
//                at the cost of a loud buzz right where the ear is sharpest
//   3   ~20 kHz  above hearing and inside the BTS7960's 25 kHz ceiling, but
//                its switching delays are microseconds long, so the short
//                pulses of a low duty never fully turn it on: duty 30 moved
//                nothing at all here. Only useful at high duty
// Timer0 is untouched, so millis() and delay() still work. analogWrite() must
// not be used on pins 9 and 10 once this runs: it assumes an 8-bit top.
void set_pwm_option(uint8_t opt) {
  uint8_t prescale;
  switch (opt) {
    case 2:  prescale = 0b011; pwm_top = 255; break;  // /64
    case 3:  prescale = 0b001; pwm_top = 400; break;  // /1
    default: opt = 1; prescale = 0b010; pwm_top = 256; break;  // /8
  }
  TCCR1A = _BV(COM1A1) | _BV(COM1B1) | _BV(WGM11);  // non-inverting, mode 10
  TCCR1B = _BV(WGM13) | prescale;
  ICR1 = pwm_top;
  pwm_opt = opt;
  set_pins(applied);  // rescale whatever is currently driven to the new top
}

const char *pwm_name() {
  return pwm_opt == 2 ? "490Hz" : pwm_opt == 3 ? "20kHz" : "3.9kHz";
}

void set_enabled(bool on) {
  if (!on) {
    // Order matters: stop driving before removing the enables, so the bridge
    // is never disabled mid-pulse.
    applied = target = 0;
    set_pins(0);
  }
  enabled = on;
  digitalWrite(PIN_R_EN, on ? HIGH : LOW);
  digitalWrite(PIN_L_EN, on ? HIGH : LOW);
}

void status(const char *tag) {
  Serial.print(tag);
  Serial.print(" en=");
  Serial.print(enabled ? 1 : 0);
  Serial.print(" duty=");
  Serial.print(applied);
  Serial.print(" target=");
  Serial.print(target);
  Serial.print(" limit=");
  Serial.print(limit);
  Serial.print(" pwm=");
  Serial.print(pwm_name());
  Serial.print(" ms=");
  Serial.println(millis());
}

void handle(char *line) {
  last_command_ms = millis();
  watchdog_tripped = false;
  char cmd = line[0];
  int arg = atoi(line + 1);  // 0 when the line carries no number
  switch (cmd) {
    case 'e':
      set_enabled(arg != 0);
      status("ok");
      break;
    case 'v':
      if (!enabled) {
        Serial.println("err not enabled: send 'e 1' first");
        break;
      }
      arg = constrain(arg, -limit, limit);
      // Reversing: pass through zero and pause there.
      if ((arg > 0 && applied < 0) || (arg < 0 && applied > 0)) {
        applied = 0;
        set_pins(0);
        reverse_until_ms = millis() + REVERSE_PAUSE_MS;
      }
      target = arg;
      status("ok");
      break;
    case 's':
      target = 0;
      applied = 0;
      set_pins(0);
      status("ok");
      break;
    case 'l':
      limit = constrain(arg, 0, 255);
      target = constrain(target, -limit, limit);
      status("ok");
      break;
    case 't':
      streaming = arg != 0;
      status("ok");
      break;
    case 'f':
      if (arg < 1 || arg > 3) {
        Serial.println("err pwm option must be 1, 2 or 3");
        break;
      }
      set_pwm_option(arg);
      status("ok");
      break;
    case '?':
      status("ok");
      break;
    default:
      Serial.println("err unknown command");
  }
}

void setup() {
  // Enables low and PWM at zero before the pins become outputs, so the bridge
  // cannot see a stray level as the Uno comes up.
  digitalWrite(PIN_R_EN, LOW);
  digitalWrite(PIN_L_EN, LOW);
  pinMode(PIN_R_EN, OUTPUT);
  pinMode(PIN_L_EN, OUTPUT);
  digitalWrite(PIN_RPWM, LOW);
  digitalWrite(PIN_LPWM, LOW);
  pinMode(PIN_RPWM, OUTPUT);
  pinMode(PIN_LPWM, OUTPUT);

  set_pwm_option(1);

  Serial.begin(115200);
  last_command_ms = millis();
  Serial.println("ready bts7960 rpwm=9 lpwm=10 r_en=7 l_en=8");
}

void loop() {
  static char line[32];
  static uint8_t len = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (len) {
        line[len] = '\0';
        handle(line);
        len = 0;
      }
    } else if (len < sizeof(line) - 1) {
      line[len++] = c;
    }
  }

  unsigned long now = millis();

  if (enabled && !watchdog_tripped && now - last_command_ms > WATCHDOG_MS) {
    set_enabled(false);
    watchdog_tripped = true;
    status("watchdog");
  }

  // Ramp towards the target, a millisecond at a time.
  if (now != last_update_ms && now >= reverse_until_ms) {
    int step = (int)min(now - last_update_ms, 255UL) * SLEW_PER_MS;
    last_update_ms = now;
    if (applied != target) {
      applied += constrain(target - applied, -step, step);
      set_pins(applied);
    }
  }

  if (streaming && now - last_status_ms >= 100) {
    last_status_ms = now;
    status("st");
  }
}
