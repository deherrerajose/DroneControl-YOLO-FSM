#include <Wire.h>
#include <SoftwareSerial.h>

SoftwareSerial xbeeSerial(0, 1);

const int takeOff = 7;
const int target = 8;
const int Abort = 9;

String control;


void setup() {
  Serial.begin(9600);
  pinMode(takeOff, INPUT);
  pinMode(target, INPUT);
  pinMode(Abort, INPUT);
}

void loop() {
  if (digitalRead(takeOff) == HIGH) {
    Serial.println("tl");
    delay(700);
  }

  if (digitalRead(target) == HIGH) {
    Serial.println("T");
    delay(700);
  }

  if (digitalRead(Abort) == HIGH) {
    Serial.println("$");
    delay(700);
  }
}