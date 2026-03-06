"""
Main loop for CNC Machine Monitoring backend.
Reads sensors, runs ML predictions, writes results to Firebase.
"""

import time
import sys
from datetime import datetime, timezone

# Import local modules
from firebase import get_sensor_data, update_ai_predictions, test_connection
from analytics import compute_all_analytics, validate_numeric, reset_history
from ml import predict_from_sensors, load_model

# Configuration
POLL_INTERVAL = 5  # seconds
REQUIRED_FIELDS = ["rpm", "temperature", "vibration", "current"]


def extract_sensor_value(sensors, field, default):
    """Safely extract and validate sensor value."""
    value = sensors.get(field, default)
    return validate_numeric(value, field, default=default)


def process_sensors(sensors):
    """
    Process sensor data and generate AI predictions.
    Returns dict of prediction results ready for Firebase.
    """
    # Extract values with defaults
    rpm = extract_sensor_value(sensors, "rpm", 1200)
    temperature = extract_sensor_value(sensors, "temperature", 70)
    vibration = extract_sensor_value(sensors, "vibration", 1.5)
    current = extract_sensor_value(sensors, "current", 8.0)
    
    # Run ML prediction (handles unit conversion internally)
    failure_probability = predict_from_sensors(
        temperature_c=temperature,
        rpm=rpm,
        current=current,
        tool_wear_minutes=50  # Will be tracked internally by analytics
    )
    
    # Compute all analytics (includes health score, alerts, trends, etc.)
    analytics_data = compute_all_analytics(
        rpm=rpm,
        temperature=temperature,
        vibration=vibration,
        current=current,
        failure_probability=failure_probability
    )
    
    # Add timestamp
    analytics_data["last_updated"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    return analytics_data


def main():
    """Main monitoring loop."""
    print("=" * 60)
    print("CNC Machine AI Monitoring Backend (Extended)")
    print("=" * 60)
    
    # Test Firebase connection
    print("[Startup] Testing Firebase connection...")
    if not test_connection():
        print("[ERROR] Cannot connect to Firebase. Check firebase_key.json and network.")
        sys.exit(1)
    print("[Startup] Firebase connected successfully")
    
    # Load ML model
    print("[Startup] Loading ML model...")
    try:
        load_model()
        print("[Startup] ML model loaded successfully")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[ERROR] Run: cd training && python train_failure_model.py")
        sys.exit(1)
    
    # Reset analytics history for fresh start
    reset_history()
    print("[Startup] Analytics history reset")
    
    print(f"[Startup] Monitoring started (polling every {POLL_INTERVAL}s)")
    print("-" * 60)
    print("Metrics: Health | Failure Risk | Tool Wear | Anomaly | Utilization")
    print("-" * 60)
    
    consecutive_errors = 0
    max_errors = 10
    
    while True:
        try:
            # Read sensor data
            sensors = get_sensor_data()
            
            if sensors is None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No sensor data available")
                time.sleep(POLL_INTERVAL)
                continue
            
            # Process and generate all predictions
            predictions = process_sensors(sensors)
            
            # Write to Firebase
            success = update_ai_predictions(predictions)
            
            if success:
                # Compact status line
                alerts_str = ""
                if predictions["alerts"]["temperature_alert"]:
                    alerts_str += "T!"
                if predictions["alerts"]["vibration_alert"]:
                    alerts_str += "V!"
                if predictions["alerts"]["current_alert"]:
                    alerts_str += "C!"
                alerts_str = alerts_str or "OK"
                
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"H:{predictions['health_score']:5.1f}% | "
                    f"F:{predictions['failure_probability']*100:5.1f}% | "
                    f"TW:{predictions['tool_wear_percent']:5.1f}% | "
                    f"A:{predictions['anomaly_score']:.2f} | "
                    f"U:{predictions['machine_utilization']:5.1f}% | "
                    f"[{alerts_str}]"
                )
                consecutive_errors = 0
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to update Firebase")
                consecutive_errors += 1
            
        except KeyboardInterrupt:
            print("\n[Shutdown] Monitoring stopped by user")
            break
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            consecutive_errors += 1
            
            if consecutive_errors >= max_errors:
                print(f"[FATAL] Too many consecutive errors ({max_errors}). Exiting.")
                sys.exit(1)
        
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()