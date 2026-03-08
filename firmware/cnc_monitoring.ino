#include <WiFi.h>
#include <Firebase_ESP_Client.h>

#include <OneWire.h>
#include <DallasTemperature.h>

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>

#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"

#define WIFI_SSID "iPhone"
#define WIFI_PASSWORD "vishal123"

#define API_KEY "AIzaSyCG3xnHN-_7-QnYKlivs7TSxPy1_DLU-sQ"
#define DATABASE_URL "https://cnc-machine-8664d-default-rtdb.asia-southeast1.firebasedatabase.app/"

#define IN1 14
#define IN2 12
#define ENA 13

#define TEMP_PIN 4
#define SOUND_PIN 34
#define CURRENT_PIN 35
#define HALL_SENSOR 26

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

OneWire oneWire(TEMP_PIN);
DallasTemperature tempSensor(&oneWire);

Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);

volatile int pulseCount = 0;
int rpm = 0;
unsigned long lastRPMTime = 0;

void IRAM_ATTR countPulse()
{
  pulseCount++;
}

float readTemperature()
{
  tempSensor.requestTemperatures();
  return tempSensor.getTempCByIndex(0);
}

int readSound()
{
  return analogRead(SOUND_PIN);
}

float readCurrent()
{
  int rawValue = analogRead(CURRENT_PIN);
  float voltage = rawValue * (3.3 / 4095.0);
  float current = (voltage - 2.5) / 0.185;
  return current;
}

float readVibration()
{
  sensors_event_t event;
  accel.getEvent(&event);

  float x = event.acceleration.x;
  float y = event.acceleration.y;
  float z = event.acceleration.z;

  float vibration = sqrt(x*x + y*y + z*z);

  return vibration;
}

void calculateRPM()
{
  if (millis() - lastRPMTime >= 1000)
  {
    rpm = pulseCount * 60;
    pulseCount = 0;
    lastRPMTime = millis();
  }
}

void sendToFirebase(float temp, float vibration, int sound, float current, int rpmValue)
{
  if (Firebase.ready())
  {
    Firebase.RTDB.setFloat(&fbdo, "cnc_machine/sensors/temperature", temp);
    Firebase.RTDB.setFloat(&fbdo, "cnc_machine/sensors/vibration", vibration);
    Firebase.RTDB.setInt(&fbdo, "cnc_machine/sensors/sound", sound);
    Firebase.RTDB.setFloat(&fbdo, "cnc_machine/sensors/current", current);
    Firebase.RTDB.setInt(&fbdo, "cnc_machine/sensors/rpm", rpmValue);
    Firebase.RTDB.setInt(&fbdo, "cnc_machine/sensors/timestamp", millis());
  }
}

void setup()
{
  Serial.begin(115200);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);

  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(ENA, HIGH);

  pinMode(SOUND_PIN, INPUT);
  pinMode(HALL_SENSOR, INPUT_PULLUP);

  analogReadResolution(12);

  attachInterrupt(digitalPinToInterrupt(HALL_SENSOR), countPulse, FALLING);

  tempSensor.begin();

  if (!accel.begin())
  {
    Serial.println("ADXL345 not detected");
    while (1);
  }

  accel.setRange(ADXL345_RANGE_16_G);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting WiFi");

  while (WiFi.status() != WL_CONNECTED)
  {
    Serial.print(".");
    delay(500);
  }

  Serial.println("\nWiFi Connected");

  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;
  config.token_status_callback = tokenStatusCallback;

  Firebase.signUp(&config, &auth, "", "");

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  Serial.println("Firebase Ready");
}

void loop()
{
  if (WiFi.status() != WL_CONNECTED)
  {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  }

  float temperature = readTemperature();
  float vibration = readVibration();
  int sound = readSound();
  float current = readCurrent();

  calculateRPM();

  Serial.println("------ CNC MACHINE DATA ------");

  Serial.print("Temperature : ");
  Serial.println(temperature);

  Serial.print("RPM : ");
  Serial.println(rpm);

  Serial.print("Vibration : ");
  Serial.println(vibration);

  Serial.print("Current : ");
  Serial.println(current);

  Serial.print("Sound : ");
  Serial.println(sound);

  sendToFirebase(temperature, vibration, sound, current, rpm);

  delay(3000);
}