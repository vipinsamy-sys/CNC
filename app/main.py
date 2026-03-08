import time
import sys
from datetime import datetime, timezone

from firebase import get_sensor_data, update_ai_predictions, test_connection
from analytics import compute_all_analytics, validate_numeric, reset_history
from ml import predict_from_sensors, load_model


# Configuration  
POLL_INTERVAL = 5
REQUIRED_FIELDS = ["rpm", "temperature", "vibration", "current"]


# Sensor Helpers  

def extract_sensor_value(sensors, field, default):
    value = sensors.get(field, default)
    return validate_numeric(value, field, default=default)


#Processing Pipeline  
def process_sensors(sensors):
    rpm = extract_sensor_value(sensors, "rpm", 1200)
    temperature = extract_sensor_value(sensors, "temperature", 70)
    vibration = extract_sensor_value(sensors, "vibration", 1.5)
    current = extract_sensor_value(sensors, "current", 8.0)

    failure_probability = predict_from_sensors(
        temperature_c=temperature,
        rpm=rpm,
        current=current,
        tool_wear_minutes=50
    )

    analytics_data = compute_all_analytics(
        rpm=rpm,
        temperature=temperature,
        vibration=vibration,
        current=current,
        failure_probability=failure_probability
    )

    analytics_data["last_updated"] = (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    return analytics_data


#Monitoring Loop  

def main():

    print("=" * 60)
    print("CNC Machine AI Monitoring Backend")
    print("=" * 60)

    print("[Startup] Checking Firebase connection...")
    if not test_connection():
        print("Firebase connection failed.")
        sys.exit(1)

    print("[Startup] Firebase connected")

    print("[Startup] Loading ML model...")
    load_model()
    print("[Startup] ML model loaded")

    reset_history()

    print(f"[Startup] Monitoring started (interval {POLL_INTERVAL}s)")
    print("-" * 60)

    while True:

        sensors = get_sensor_data()

        if sensors is None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No sensor data")
            time.sleep(POLL_INTERVAL)
            continue

        predictions = process_sensors(sensors)

        success = update_ai_predictions(predictions)

        if success:

            alerts = predictions["alerts"]

            alert_flag = ""
            if alerts["temperature_alert"]:
                alert_flag += "T!"
            if alerts["vibration_alert"]:
                alert_flag += "V!"
            if alerts["current_alert"]:
                alert_flag += "C!"

            if alert_flag == "":
                alert_flag = "OK"

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"H:{predictions['health_score']:5.1f}% | "
                f"F:{predictions['failure_probability']*100:5.1f}% | "
                f"TW:{predictions['tool_wear_percent']:5.1f}% | "
                f"A:{predictions['anomaly_score']:.2f} | "
                f"U:{predictions['machine_utilization']:5.1f}% | "
                f"[{alert_flag}]"
            )

        else:
            print("Failed to update Firebase")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()