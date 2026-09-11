// Diagnoses the four control lines to the BTS7960 without a multimeter, by
// comparing the known-good side against the dead one.
//
// Two tests per pin:
//   pullup  - pin as INPUT_PULLUP. The driver's inputs have internal
//             pull-downs, so a connected wire usually reads LOW. HIGH
//             suggests the wire is off, broken, or not seated.
//   driven  - pin as OUTPUT HIGH, read back. LOW means something external is
//             holding the line down: a short, or a failed driver input.
//
// Safe with the motor attached: both enables are held low for the PWM tests,
// and the PWM pins are held low for the enable tests, so the bridge never
// drives the motor.

const uint8_t PIN_RPWM = 9;
const uint8_t PIN_LPWM = 10;
const uint8_t PIN_R_EN = 7;
const uint8_t PIN_L_EN = 8;

void hold_low(uint8_t pin) {
  digitalWrite(pin, LOW);
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);
}

void probe(const char *name, uint8_t pin) {
  pinMode(pin, INPUT_PULLUP);
  delay(20);
  int pulled = digitalRead(pin);

  digitalWrite(pin, LOW);
  pinMode(pin, OUTPUT);
  digitalWrite(pin, HIGH);
  delay(20);
  int driven = digitalRead(pin);
  hold_low(pin);

  Serial.print(name);
  Serial.print(" pin=");
  Serial.print(pin);
  Serial.print(" pullup=");
  Serial.print(pulled ? "HIGH" : "LOW");
  Serial.print(" driven=");
  Serial.println(driven ? "HIGH" : "LOW");
}

void setup() {
  Serial.begin(115200);
  hold_low(PIN_R_EN);
  hold_low(PIN_L_EN);
  hold_low(PIN_RPWM);
  hold_low(PIN_LPWM);
  delay(300);

  Serial.println("--- pwm lines (enables held low) ---");
  probe("RPWM", PIN_RPWM);
  probe("LPWM", PIN_LPWM);
  Serial.println("--- enable lines (pwm held low) ---");
  probe("R_EN", PIN_R_EN);
  probe("L_EN", PIN_L_EN);
  Serial.println("--- done; compare the working side against the dead one ---");
}

void loop() {}
