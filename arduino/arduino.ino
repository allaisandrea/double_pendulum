// Drives the pendulum motor through a Cytron SHIELD-MDD10 from a stream of
// bytes over USB serial.
//
// The protocol is one signed byte per command, nothing else: each byte read
// is an int8_t duty in units of 1/255 of full scale, clipped to +/-LIMIT and
// applied at once. Positive turns the shaft counter-clockwise. There is no
// framing, because every byte is a complete command; no ramp, so what is
// sent is what the motor gets; and no replies. The one exception is a
// single line at startup, "ready mdd10", after which the board is silent.
//
// Timing, for anyone reconstructing when an action took effect: a byte is
// applied as soon as the serial interrupt has it, and the new duty reaches
// the pin at the top of the next PWM period, at most 1/PWM_HZ later. The
// Uno's own clock never enters into it, so a host that timestamps its
// writes has the action time to within the USB transfer.
//
// The watchdog: WATCHDOG_MS after the last byte, the duty goes to zero. The
// host must resend at least that often to keep the motor driven.
//
// Every byte moves the motor, text included - '?' is 63 - so nothing but
// the controlling program should ever open this port. Opening it resets
// the Uno, and bytes sent before "ready" go to the bootloader.
//
// The shield is sign-magnitude (datasheet, table 4):
//   PWM low            -> both outputs low: brake
//   PWM high, DIR low  -> forward
//   PWM high, DIR high -> backward
// With no enable pin it cannot coast: zero duty shorts the motor, which
// then damps the pendulum through its own back-EMF.

const uint8_t PIN_PWM = 9;  // PWM1 jumper on D9: Timer1 channel A
const uint8_t PIN_DIR = 8;  // DIR1 jumper on D8

// The shield's channel 2 inputs can be jumpered to any of these. Nothing is
// connected to motor 2, but a floating input could still switch its bridge.
const uint8_t CHANNEL2_PWM_PINS[] = {6, 10, 11};

// Cap on |duty|, in the same 1/255 units the bytes use. Duty 60 felt violent
// with the previous driver, so approach this cap gradually.
const int8_t LIMIT = 80;

// Rated for 20 kHz at full current, and at the edge of hearing. Duty 30 is
// then a ~6 us pulse, which this shield follows; the BTS7960 did not.
const uint16_t PWM_HZ = 20000;
const uint16_t PWM_TOP = 8000000UL / PWM_HZ;  // Timer1 /1: f = 16 MHz / (2 * top)

const unsigned long WATCHDOG_MS = 300;

unsigned long last_byte_ms = 0;
bool driving = false;

void apply(int8_t duty) {
  duty = constrain(duty, -LIMIT, LIMIT);
  // The compare value is double-buffered and takes effect at the top of the
  // PWM period, while DIR changes at once. Across a reversal, that can leave
  // the old magnitude in the new direction for up to one period: harmless.
  uint16_t mag = duty < 0 ? -duty : duty;
  OCR1A = ((uint32_t)mag * PWM_TOP) / 255;
  digitalWrite(PIN_DIR, duty < 0 ? HIGH : LOW);
  driving = duty != 0;
}

void setup() {
  // Everything low before it becomes an output, so the shield cannot see a
  // stray level as the Uno comes up. The pins float during reset and the
  // bootloader, roughly two seconds, and nothing here can help with that.
  for (uint8_t pin : CHANNEL2_PWM_PINS) {
    digitalWrite(pin, LOW);
    pinMode(pin, OUTPUT);
  }
  digitalWrite(PIN_DIR, LOW);
  pinMode(PIN_DIR, OUTPUT);
  digitalWrite(PIN_PWM, LOW);
  pinMode(PIN_PWM, OUTPUT);

  // Phase-correct PWM with ICR1 as top (mode 10), channel A non-inverting.
  // Timer0 is untouched, so millis() still works.
  TCCR1A = _BV(COM1A1) | _BV(WGM11);
  TCCR1B = _BV(WGM13) | _BV(CS10);
  ICR1 = PWM_TOP;
  OCR1A = 0;

  Serial.begin(115200);
  Serial.println("ready mdd10");
}

void loop() {
  while (Serial.available()) {
    apply((int8_t)Serial.read());
    last_byte_ms = millis();
  }
  if (driving && millis() - last_byte_ms > WATCHDOG_MS) {
    apply(0);
  }
}
