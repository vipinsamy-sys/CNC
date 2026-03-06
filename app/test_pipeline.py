"""
Diagnostic Test Script for CNC Machine Monitoring Pipeline.
Tests Firebase connectivity, ML predictions, and end-to-end pipeline.

Usage:
    cd app
    python test_pipeline.py
"""

import os
import sys
import time
from datetime import datetime

# Ensure we're in the right directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

print("=" * 60)
print("CNC MACHINE MONITORING - DIAGNOSTIC TEST")
print("=" * 60)
print(f"Working directory: {os.getcwd()}")
print(f"Test started at: {datetime.now().isoformat()}")
print()

# Test results tracking
results = {
    "firebase_connection": False,
    "firebase_read": False,
    "firebase_write": False,
    "ml_model_load": False,
    "ml_prediction": False,
    "analytics": False,
    "full_pipeline": False
}


def test_step(name, func):
    """Run a test step and report results."""
    print(f"\n{'─' * 60}")
    print(f"TEST: {name}")
    print(f"{'─' * 60}")
    try:
        result = func()
        if result:
            print(f"✅ PASSED: {name}")
            return True
        else:
            print(f"❌ FAILED: {name}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {name}")
        print(f"   Exception: {type(e).__name__}: {e}")
        return False


# ============================================
# TEST 1: Firebase Connection
# ============================================
def test_firebase_connection():
    from firebase import test_connection
    
    print("Testing Firebase connection...")
    if test_connection():
        print("✓ Firebase connection successful")
        results["firebase_connection"] = True
        return True
    else:
        print("✗ Cannot connect to Firebase")
        print("  Check: firebase_key.json exists and is valid")
        return False


# ============================================
# TEST 2: Firebase Read
# ============================================
def test_firebase_read():
    from firebase import get_sensor_data
    
    print("Reading sensor data from Firebase...")
    data = get_sensor_data()
    
    if data is None:
        print("⚠ No sensor data found at 'cnc_machine' path")
        print("  This is normal if sensor_simulator.py hasn't run yet")
        print("  Writing test data instead...")
        
        # Write test data for subsequent tests
        from firebase import db
        test_data = {
            "rpm": 1250,
            "temperature": 72,
            "vibration": 1.8,
            "current": 8.5,
            "sound": 55,
            "timestamp": int(time.time())
        }
        ref = db.reference("cnc_machine")
        ref.set(test_data)
        print(f"  Test data written: {test_data}")
        data = test_data
    
    print(f"✓ Sensor data: {data}")
    
    # Validate required fields
    required = ["rpm", "temperature", "vibration", "current"]
    missing = [f for f in required if f not in data]
    if missing:
        print(f"⚠ Missing fields: {missing}")
    else:
        print("✓ All required fields present")
    
    results["firebase_read"] = True
    return True


# ============================================
# TEST 3: Firebase Write
# ============================================
def test_firebase_write():
    from firebase import update_ai_predictions, db
    
    print("Writing test AI predictions to Firebase...")
    
    test_predictions = {
        "failure_probability": 0.15,
        "avg_rpm": 1225.5,
        "rpm_stability": 0.96,
        "vibration_alert": False,
        "current_alert": False,
        "health_score": 85.0,
        "test_flag": "diagnostic_test"
    }
    
    success = update_ai_predictions(test_predictions)
    
    if success:
        print(f"✓ Predictions written successfully")
        
        # Verify write
        ref = db.reference("cnc_machine_ai")
        read_back = ref.get()
        print(f"✓ Read back: {read_back}")
        
        # Clean up test flag
        ref.child("test_flag").delete()
        
        results["firebase_write"] = True
        return True
    else:
        print("✗ Failed to write predictions")
        return False


# ============================================
# TEST 4: ML Model Loading
# ============================================
def test_ml_model_load():
    print("Loading ML model...")
    
    model_path = os.path.join(BASE_DIR, "models", "failure_model.pkl")
    
    if not os.path.exists(model_path):
        print(f"⚠ Model file not found: {model_path}")
        print("  Running model training first...")
        
        # Run training
        from training.train_failure_model import train_model
        train_model()
        print("✓ Model training complete")
    
    from ml import load_model, get_model
    
    model = load_model()
    print(f"✓ Model loaded: {type(model).__name__}")
    print(f"  Estimators: {getattr(model, 'n_estimators', 'N/A')}")
    
    results["ml_model_load"] = True
    return True


# ============================================
# TEST 5: ML Prediction
# ============================================
def test_ml_prediction():
    from ml import predict_failure, predict_from_sensors
    
    print("Testing ML predictions...")
    
    # Test with direct features (Kelvin units)
    print("\n1. Direct feature prediction:")
    prob1 = predict_failure(
        air_temp_k=308.0,      # ~35°C
        process_temp_k=318.0,  # ~45°C
        rpm=1400,
        torque=45.0,
        tool_wear=100
    )
    print(f"   Normal conditions → Failure probability: {prob1:.4f} ({prob1*100:.1f}%)")
    
    # Test with sensor values (Celsius)
    print("\n2. Sensor-based prediction:")
    prob2 = predict_from_sensors(
        temperature_c=72,
        rpm=1250,
        current=8.5,
        tool_wear_minutes=50
    )
    print(f"   Sensor data → Failure probability: {prob2:.4f} ({prob2*100:.1f}%)")
    
    # Test edge cases
    print("\n3. High-risk scenario:")
    prob3 = predict_from_sensors(
        temperature_c=95,  # Very high
        rpm=1800,         # High RPM
        current=12.0,     # High current
        tool_wear_minutes=200  # Worn tool
    )
    print(f"   High-risk conditions → Failure probability: {prob3:.4f} ({prob3*100:.1f}%)")
    
    # Validate probabilities are reasonable
    if 0 <= prob1 <= 1 and 0 <= prob2 <= 1 and 0 <= prob3 <= 1:
        print("\n✓ All predictions are valid probabilities (0-1)")
        results["ml_prediction"] = True
        return True
    else:
        print("\n✗ Invalid probability values returned")
        return False


# ============================================
# TEST 6: Analytics Module
# ============================================
def test_analytics():
    from analytics import (
        update_rpm, 
        detect_vibration_anomaly, 
        detect_current_overload,
        calculate_health_score,
        reset_history
    )
    
    print("Testing analytics module...")
    
    # Reset for clean test
    reset_history()
    
    # Test RPM tracking
    print("\n1. RPM tracking:")
    test_rpms = [1200, 1210, 1220, 1230, 1240, 1235, 1225, 1215]
    for rpm in test_rpms:
        avg_rpm, stability = update_rpm(rpm)
    print(f"   Average RPM: {avg_rpm:.1f}")
    print(f"   Stability: {stability:.4f} ({stability*100:.1f}%)")
    
    if not (1000 <= avg_rpm <= 1500):
        print("   ⚠ Average RPM out of expected range")
    if not (0 <= stability <= 1):
        print("   ⚠ Stability not in 0-1 range")
    
    # Test vibration anomaly
    print("\n2. Vibration anomaly detection:")
    reset_history()
    for _ in range(10):
        detect_vibration_anomaly(1.5)  # Normal baseline
    
    normal_alert = detect_vibration_anomaly(1.6)
    spike_alert = detect_vibration_anomaly(3.5)
    
    print(f"   Normal vibration (1.6): Alert = {normal_alert}")
    print(f"   Spike vibration (3.5): Alert = {spike_alert}")
    
    # Test health score
    print("\n3. Health score calculation:")
    score1 = calculate_health_score(0.1, False, False)
    score2 = calculate_health_score(0.1, True, False)
    score3 = calculate_health_score(0.5, True, True)
    
    print(f"   Low risk, no alerts: {score1:.1f}")
    print(f"   Low risk, vibration alert: {score2:.1f}")
    print(f"   High risk, both alerts: {score3:.1f}")
    
    results["analytics"] = True
    return True


# ============================================
# TEST 7: Full Pipeline
# ============================================
def test_full_pipeline():
    from main import process_sensors
    from firebase import get_sensor_data, update_ai_predictions
    from analytics import reset_history
    
    print("Testing full pipeline...")
    reset_history()
    
    # Get or create sensor data
    sensors = get_sensor_data()
    if sensors is None:
        sensors = {
            "rpm": 1250,
            "temperature": 72,
            "vibration": 1.8,
            "current": 8.5
        }
        print(f"Using test sensors: {sensors}")
    else:
        print(f"Using live sensors: {sensors}")
    
    # Process through pipeline
    predictions = process_sensors(sensors)
    print(f"\nGenerated predictions:")
    for key, value in predictions.items():
        print(f"   {key}: {value}")
    
    # Validate predictions
    required_keys = [
        "failure_probability", "avg_rpm", "rpm_stability",
        "vibration_alert", "current_alert", "health_score"
    ]
    missing = [k for k in required_keys if k not in predictions]
    if missing:
        print(f"\n✗ Missing prediction keys: {missing}")
        return False
    
    # Validate ranges
    valid = True
    if not (0 <= predictions["failure_probability"] <= 1):
        print("✗ failure_probability out of range")
        valid = False
    if not (0 <= predictions["rpm_stability"] <= 1):
        print("✗ rpm_stability out of range")
        valid = False
    if not (0 <= predictions["health_score"] <= 100):
        print("✗ health_score out of range")
        valid = False
    
    if valid:
        # Write to Firebase
        success = update_ai_predictions(predictions)
        if success:
            print("\n✓ Full pipeline completed successfully")
            print("✓ Predictions written to Firebase")
            results["full_pipeline"] = True
            return True
        else:
            print("\n✗ Failed to write predictions to Firebase")
            return False
    else:
        return False


# ============================================
# RUN ALL TESTS
# ============================================
def run_all_tests():
    test_step("Firebase Connection", test_firebase_connection)
    test_step("Firebase Read", test_firebase_read)
    test_step("Firebase Write", test_firebase_write)
    test_step("ML Model Loading", test_ml_model_load)
    test_step("ML Prediction", test_ml_prediction)
    test_step("Analytics Module", test_analytics)
    test_step("Full Pipeline", test_full_pipeline)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {status} : {test_name.replace('_', ' ').title()}")
    
    print(f"\n{'─' * 60}")
    print(f"RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready.")
        return 0
    else:
        print("⚠ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
