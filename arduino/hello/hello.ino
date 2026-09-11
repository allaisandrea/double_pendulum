// Proves the upload path and the serial link work, before any motor code.
// Deliberately touches no I/O pins: every pin stays an input, as it is after
// a reset, so nothing here can drive the motor driver's inputs.

void setup() {
  Serial.begin(115200);
}

void loop() {
  Serial.print("hello from the uno, millis=");
  Serial.println(millis());
  delay(500);
}
