// Drives the pendulum motor through a Cytron SHIELD-MDD10, taking commands
// over USB serial. Replaces the BTS7960 sketch in motor/ and keeps its
// protocol, so a host program written for one works with the other.
//
// The shield is sign-magnitude: one PWM pin sets the drive, one DIR pin sets
// the direction. Its truth table (datasheet, table 4):
//   PWM low            -> both outputs low: brake
//   PWM high, DIR low  -> A high, B low:    forward
//   PWM high, DIR high -> A low,  B high:   backward
// So during each PWM off-phase the motor is shorted, not left floating.
//
// There is no enable pin, and therefore no way to let the motor coast: with
// PWM held low the motor terminals are shorted together, so an idle motor
// brakes the arm through its own back-EMF. Expect the pendulum to swing
// noticeably more damped with this shield attached than with it unplugged.
//
// Commands, one per line, at 115200 baud:
//   e 1 | e 0     arm / disarm. Disarmed holds PWM low (a brake, see above)
//                 and refuses 'v'
//   v <-255..255> signed duty; negative is the other direction
//   s             duty 0, still armed
//   l <0..255>    cap on |duty|, applied to this and later commands
//   t 1 | t 0     stream status at 10 Hz
//   f <100..20000>  PWM frequency in Hz; 20000 at startup
//   ?             one status line
// Every command answers with a line beginning "ok" or "err".

const uint8_t PIN_PWM = 9;  // PWM1 jumper on D9: Timer1 channel A
const uint8_t PIN_DIR = 8;  // DIR1 jumper on D8

// The shield's channel 2 inputs can be jumpered to any of these. Nothing is
// connected to motor 2, but a floating input could still switch its bridge,
// so all of them are held low.
const uint8_t CHANNEL2_PWM_PINS[] = {6, 10, 11};

// With no command for this long, stop and disarm. The laptop crashing, the
// USB cable coming loose and the control loop stalling all look the same from
// here, and all of them should stop the motor.
const unsigned long WATCHDOG_MS = 300;

// Ramp limit, in duty counts per millisecond: full scale takes ~0.25 s.
const int SLEW_PER_MS = 1;

// Direction reversals wait this long at zero duty, so DIR never changes
// while the bridge is driving.
const unsigned long REVERSE_PAUSE_MS = 5;

int target = 0;          // duty asked for, after clamping
int applied = 0;         // duty actually on the pins, ramped towards target
int limit = 40;          // 60 was already violent on this rig; raise with 'l'
bool armed = false;
bool streaming = false;
unsigned long last_command_ms = 0;
unsigned long last_update_ms = 0;
unsigned long last_status_ms = 0;
unsigned long reverse_until_ms = 0;
bool watchdog_tripped = false;

uint16_t pwm_hz = 20000;
uint16_t pwm_top = 400;  // Timer1 counts to this: sets frequency and resolution

void set_pins(int duty) {
  // DIR only changes at zero duty: the reversal pause in 'v' guarantees
  // that applied passes through 0 before its sign flips.
  if (duty > 0) digitalWrite(PIN_DIR, LOW);        // forward
  else if (duty < 0) digitalWrite(PIN_DIR, HIGH);  // backward
  uint16_t mag = duty < 0 ? -duty : duty;
  OCR1A = ((uint32_t)mag * pwm_top) / 255;
}

// Timer1 in phase-correct PWM with ICR1 as the top value, so frequency and
// resolution are both ours: f = 16 MHz / (2 * prescaler * top). Only
// channel A is connected to a pin; pin 10 (channel B) stays a plain output.
//
// The shield is rated to 20 kHz at full current, which is at the edge of
// hearing, so that is the default. The BTS7960 could not follow 20 kHz at
// small duties - its switching delays were longer than the pulses - and the
// same can happen here: at 20 kHz, duty 30 is a ~6 us pulse. If small duties
// do nothing, lower the frequency with 'f' before suspecting anything else.
void set_pwm_frequency(uint16_t hz) {
  uint8_t prescale = 0b001;  // /1
  uint32_t top = 8000000UL / hz;
  if (top > 65535) {
    prescale = 0b010;        // /8
    top /= 8;
  }
  if (top > 65535) {
    prescale = 0b011;        // /64
    top /= 8;
  }
  pwm_top = top;
  pwm_hz = hz;
  TCCR1A = _BV(COM1A1) | _BV(WGM11);  // channel A non-inverting, mode 10
  TCCR1B = _BV(WGM13) | prescale;
  ICR1 = pwm_top;
  set_pins(applied);
}

void disarm() {
  applied = target = 0;
  set_pins(0);
  armed = false;
}

void status(const char *tag) {
  Serial.print(tag);
  Serial.print(armed ? " armed" : " disarmed");
  Serial.print(" duty=");
  Serial.print(applied);
  Serial.print(" target=");
  Serial.print(target);
  Serial.print(" limit=");
  Serial.print(limit);
  Serial.print(" pwm=");
  Serial.print(pwm_hz);
  Serial.print("Hz ms=");
  Serial.println(millis());
}

void handle(char *line) {
  last_command_ms = millis();
  watchdog_tripped = false;
  char cmd = line[0];
  long arg = atol(line + 1);  // 0 when the line carries no number
  switch (cmd) {
    case 'e':
      if (arg) armed = true;
      else disarm();
      status("ok");
      break;
    case 'v':
      if (!armed) {
        Serial.println("err disarmed: send 'e 1' first");
        break;
      }
      arg = constrain(arg, -limit, limit);
      if ((arg > 0 && applied < 0) || (arg < 0 && applied > 0)) {
        applied = 0;
        set_pins(0);
        reverse_until_ms = millis() + REVERSE_PAUSE_MS;
      }
      target = arg;
      status("ok");
      break;
    case 's':
      target = applied = 0;
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
      if (arg < 100 || arg > 20000) {
        Serial.println("err frequency must be 100..20000 Hz");
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
  // Everything low before it becomes an output, so the shield cannot see a
  // stray level as the Uno comes up. The Uno's pins float during reset and
  // the bootloader, which is roughly two seconds.
  for (uint8_t pin : CHANNEL2_PWM_PINS) {
    digitalWrite(pin, LOW);
    pinMode(pin, OUTPUT);
  }
  digitalWrite(PIN_DIR, LOW);
  pinMode(PIN_DIR, OUTPUT);
  digitalWrite(PIN_PWM, LOW);
  pinMode(PIN_PWM, OUTPUT);

  set_pwm_frequency(20000);

  Serial.begin(115200);
  last_command_ms = millis();
  Serial.println("ready mdd10 pwm=9 dir=8");
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

  if (armed && !watchdog_tripped && now - last_command_ms > WATCHDOG_MS) {
    disarm();
    watchdog_tripped = true;
    status("watchdog");
  }

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
