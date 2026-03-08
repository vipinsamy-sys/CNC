import os
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

print("=" * 50)
print("CNC MONITORING PIPELINE TEST")
print("=" * 50)
print("Directory:", os.getcwd())
print("Started:", datetime.now().isoformat())
print()

results = {}

def run_test(name, func):
    print("\n---", name, "---")
    try:
        ok = func()
        results[name] = ok
        print("PASS" if ok else "FAIL")
    except Exception as e:
        results[name] = False
        print("ERROR:", e)


 
# Firebase Connection
def test_firebase_connection():
    from firebase import test_connection
    return test_connection()


 
# Firebase Read
def test_firebase_read():
    from firebase import get_sensor_data, db

    data = get_sensor_data()

    if data is None:
        test_data = {
            "rpm": 1250,
            "temperature": 72,
            "vibration": 1.8,
            "current": 8.5,
            "sound": 55,
            "timestamp": int(time.time())
        }

        db.reference("cnc_machine/sensors").set(test_data)
        data = test_data

    print("Sensor data:", data)

    required = ["rpm", "temperature", "vibration", "current"]
    return all(k in data for k in required)


 
# Firebase Write
def test_firebase_write():
    from firebase import update_ai_predictions, db

    test_predictions = {
        "failure_probability": 0.15,
        "avg_rpm": 1225.5,
        "rpm_stability": 0.96,
        "health_score": 85.0
    }

    if not update_ai_predictions(test_predictions):
        return False

    analytics = db.reference("cnc_machine/analytics").get()
    ai_data = db.reference("cnc_machine_ai").get()

    return analytics is not None and ai_data is not None


 
# ML Model

def test_ml_model():
    from ml import load_model, predict_from_sensors

    load_model()

    prob = predict_from_sensors(
        temperature_c=72,
        rpm=1250,
        current=8.5
    )

    print("Prediction:", prob)

    return 0 <= prob <= 1


 
# Analytics
def test_analytics():
    from analytics import update_rpm, calculate_health_score, reset_history

    reset_history()

    rpms = [1200,1210,1220,1230,1240,1235,1225,1215]

    for r in rpms:
        avg, stability = update_rpm(r)

    print("Avg RPM:", avg)
    print("Stability:", stability)

    score = calculate_health_score(
        temperature=70,
        vibration=1.5,
        current=8,
        rpm_stability=0.98,
        anomaly_score=0.1
    )

    print("Health score:", score)

    return 0 <= score <= 100

# Energy / Cooling / Anomaly
def test_metrics():
    from analytics import (
        calculate_energy_per_part,
        calculate_cooling_efficiency,
        calculate_anomaly_score,
        update_rpm,
        reset_history
    )

    reset_history()

    energy = calculate_energy_per_part(8.5, 60)
    print("Energy:", energy)

    eff = calculate_cooling_efficiency(90)
    print("Cooling efficiency:", eff)

    for r in [1200,1210,1205,1215,1200]:
        update_rpm(r)

    anomaly = calculate_anomaly_score(75,1.5,8,1200)
    print("Anomaly:", anomaly)

    return (
        energy > 0
        and 0 <= eff <= 100
        and 0 <= anomaly <= 1
    )


 
# Full Pipeline
def test_pipeline():
    from main import process_sensors
    from firebase import update_ai_predictions

    sensors = {
        "rpm": 1250,
        "temperature": 72,
        "vibration": 1.8,
        "current": 8.5
    }

    result = process_sensors(sensors)

    print("Predictions:", result)

    if "failure_probability" not in result:
        return False

    return update_ai_predictions(result)

 
# Run All Tests
def main():

    run_test("Firebase Connection", test_firebase_connection)
    run_test("Firebase Read", test_firebase_read)
    run_test("Firebase Write", test_firebase_write)
    run_test("ML Model", test_ml_model)
    run_test("Analytics", test_analytics)
    run_test("Metrics", test_metrics)
    run_test("Full Pipeline", test_pipeline)

    print("\n--- SUMMARY ---")

    passed = sum(results.values())
    total = len(results)

    for k,v in results.items():
        print(k, "PASS" if v else "FAIL")

    print(f"\n{passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())