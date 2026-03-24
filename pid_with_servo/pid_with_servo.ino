#include <Arduino.h>
#include <QTRSensors.h>
#include <Servo.h>
#include <Adafruit_NeoPixel.h>

// ================= MOTORS =================
#define PWM_A   16
#define FWD_A   18
#define REV_A   17

#define PWM_B   21
#define FWD_B   20
#define REV_B   19

// ================= SERVOS =================
#define SERVO_LR_PIN 27
#define SERVO_FB_PIN 28

Servo servoLR;
Servo servoFB;

// ================= LED ====================
#define LED_PIN    15
#define LED_COUNT  8

Adafruit_NeoPixel leds(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

// ================= QTR ====================
#define QTR_LEDON 1
#define BLACK_THRESHOLD 800

const uint8_t SensorCount = 8;
uint8_t qtrPins[SensorCount] = {2, 3, 4, 5, 9, 6, 7, 8};

QTRSensors qtr;
uint16_t sensorValues[SensorCount];

// ================= PHYSICAL MODEL =========
float maxSpeed = 1.3;      // m/s at PWM 255 (MEASURE)
float maxLength = 0.065;    // meters across sensor
float baseSpeed_mps = 0.5; // forward speed

// ================= PID ====================
float Kp = 2.0;
float Kd = 0.02;

float lastError = 0;
float prevDerivative = 0;

// derivative smoothing
float alpha = 0.1;

// ================= TIMESTEP ===============
const float dt = 0.002;                // 2 ms
const unsigned long LOOP_TIME_US = 2000;

// ================= HELPERS =================
int mpsToPwm(float speed_mps) {
  return constrain((speed_mps / maxSpeed) * 255.0, -255, 255);
}

void setMotor(int pwmA, int pwmB) {
  pwmA = constrain(pwmA, -255, 255);
  pwmB = constrain(pwmB, -255, 255);

  digitalWrite(FWD_A, pwmA >= 0);
  digitalWrite(REV_A, pwmA < 0);
  analogWrite(PWM_A, abs(pwmA));

  digitalWrite(FWD_B, pwmB >= 0);
  digitalWrite(REV_B, pwmB < 0);
  analogWrite(PWM_B, abs(pwmB));
}

// ================= CALIBRATION ============
void autoCalibrate() {
  // LED ON (10%)
  leds.setBrightness(25);
  leds.fill(leds.Color(255, 150, 0));
  leds.show();

  delay(3000); // move robot manually

  for (int i = 0; i < 120; i++) {
    qtr.calibrate();
    delay(10);
  }

  // LED OFF
  leds.clear();
  leds.show();

  delay(300);
}

// ================= SETUP ==================
void setup() {
  pinMode(FWD_A, OUTPUT);
  pinMode(REV_A, OUTPUT);
  pinMode(PWM_A, OUTPUT);

  pinMode(FWD_B, OUTPUT);
  pinMode(REV_B, OUTPUT);
  pinMode(PWM_B, OUTPUT);

  // Servos fixed
  servoLR.attach(SERVO_LR_PIN);
  servoFB.attach(SERVO_FB_PIN);
  servoLR.write(90);
  servoFB.write(90);

  // LED init
  leds.begin();
  leds.clear();
  leds.show();

  // QTR setup
  qtr.setTypeRC();
  qtr.setSensorPins(qtrPins, SensorCount);
  qtr.setEmitterPin(QTR_LEDON);

  delay(2000);
  autoCalibrate();
}

// ================= LOOP ===================
void loop() {
  unsigned long start = micros();

  // ---- READ SENSORS ----
  qtr.read(sensorValues);

  // ---- ALL BLACK ----
  bool allBlack = true;
  for (int i = 0; i < SensorCount; i++) {
    if (sensorValues[i] < BLACK_THRESHOLD) {
      allBlack = false;
      break;
    }
  }

  if (allBlack) {
    setMotor(mpsToPwm(baseSpeed_mps), mpsToPwm(baseSpeed_mps));
  } else {
    uint16_t position = qtr.readLineBlack(sensorValues);

    // ---- SENSOR → METERS ----
    float position_m = (position / 7000.0) * maxLength;
    float error_m = position_m - (maxLength / 2.0);

    // ---- DERIVATIVE (with dt) ----
    float rawDerivative = (error_m - lastError) / dt;

    // ---- FILTER ----
    float derivative = alpha * rawDerivative + (1 - alpha) * prevDerivative;
    prevDerivative = derivative;

    // ---- PD CONTROL ----
    float correction_mps = Kp * error_m + Kd * derivative;

    float left_mps  = baseSpeed_mps - correction_mps;
    float right_mps = baseSpeed_mps + correction_mps;

    setMotor(mpsToPwm(left_mps), mpsToPwm(right_mps));

    lastError = error_m;
  }

  // ---- FIXED TIMESTEP ----
  while (micros() - start < LOOP_TIME_US);
}