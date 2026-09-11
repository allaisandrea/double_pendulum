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
//   f <100..25000>  PWM frequency in Hz; 10000 at startup
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

uint16_t pwm_hz = 10000;  // as requested; the achieved rate is within a few Hz
uint16_t pwm_top = 800;   // Timer1 counts to this: sets frequency and resolution

void set_pins(int duty) {
  // Only ever one side driven; the other is held at zero. Duty arrives as
  // 0..255 on the wire and is scaled to whatever top the frequency uses.
  OCR1A = duty > 0 ? ((uint32_t)duty * pwm_top) / 255 : 0;   // pin 9, RPWM
  OCR1B = duty < 0 ? ((uint32_t)-duty * pwm_top) / 255 : 0;  // pin 10, LPWM
}

// Timer1 runs in phase-correct PWM with ICR1 as the top value, so both the
// frequency and the resolution are ours: f = 16 MHz / (2 * prescaler * top).
//
// The choice is a compromise. Low frequencies chop coarsely, which means more
// current ripple: that can break a stubborn load free, but 490 Hz buzzes right
// where the ear is sharpest. High frequencies are quieter and smoother, until
// the pulses get shorter than the BTS7960's switching delays of a few
// microseconds, at which point its output never fully turns on - 20 kHz at
// duty 30 is a ~6 us pulse, and moved nothing at all here.
//
// Timer0 is untouched, so millis() and delay() still work. analogWrite() must
// not be used on pins 9 and 10 once this runs: it assumes an 8-bit top.
void set_pwm_frequency(uint16_t hz) {
  // Prefer the smallest prescaler whose top still fits in 16 bits: a bigger
  // top is finer duty resolution.
  uint8_t prescale = 0b001;          // /1
  uint32_t top = 8000000UL / hz;
  if (top > 65535) {
    prescale = 0b010;                // /8
    top /= 8;
  }
  if (top > 65535) {
    prescale = 0b011;                // /64
    top /= 8;
  }
  pwm_top = top;
  pwm_hz = hz;
  TCCR1A = _BV(COM1A1) | _BV(COM1B1) | _BV(WGM11);  // non-inverting, mode 10
  TCCR1B = _BV(WGM13) | prescale;
  ICR1 = pwm_top;
  set_pins(applied);  // rescale whatever is currently driven to the new top
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
  Serial.print(pwm_hz);
  Serial.print("Hz");
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
      if (arg < 100 || arg > 25000) {
        Serial.println("err frequency must be 100..25000 Hz");
        break;
      }
      set_pwm_frequency(arg);
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

  // 10 kHz: quieter than 3.9 kHz and still long enough a pulse for the driver
  // to switch properly at the duties this rig uses. Watch the low end though -
  // the smaller the duty, the shorter the pulse, and below a few microseconds
  // the BTS7960 stops responding at all.
  set_pwm_frequency(10000);

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
