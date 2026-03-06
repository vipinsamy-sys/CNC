"""
CNC Machine Sensor Simulator.
Generates realistic sensor data and sends to Firebase.
"""

import random
import time
import requests
from datetime import datetime

# Firebase configuration
FIREBASE_URL = "https://cnc-machine-8664d-default-rtdb.asia-southeast1.firebasedatabase.app/cnc_machine.json"

# Simulation parameters
POLL_INTERVAL = 5  # seconds

# Sensor ranges (realistic CNC machine values)
SENSOR_CONFIG = {
    "rpm": {"min": 1100, "max": 1400, "drift": 20},
    "temperature": {"min": 60, "max": 80, "drift": 3},
    "vibration": {"min": 1.0, "max": 3.0, "drift": 0.3},
    "current": {"min": 7.0, "max": 10.0, "drift": 0.5},
    "sound": {"min": 40, "max": 70, "drift": 5},
}

# State for smooth transitions
current_values = {
    "rpm": 1250,
    "temperature": 70,
    "vibration": 1.5,
    "current": 8.0,
    "sound": 55,
}


def smooth_transition(current, min_val, max_val, drift):
    """Generate smooth sensor value transitions."""
    change = random.uniform(-drift, drift)
    new_val = current + change
    return max(min_val, min(max_val, new_val))


def generate_sensor_data():
    """Generate realistic sensor readings with smooth transitions."""
    global current_values
    
    # Update each sensor with smooth transitions
    for sensor, config in SENSOR_CONFIG.items():
        current_values[sensor] = smooth_transition(
            current_values[sensor],
            config["min"],
            config["max"],
            config["drift"]
        )
    
    # Build data payload
    data = {
        "rpm": int(current_values["rpm"]),
        "temperature": round(current_values["temperature"], 1),
        "vibration": round(current_values["vibration"], 2),
        "current": round(current_values["current"], 2),
        "sound": int(current_values["sound"]),
        "timestamp": int(time.time())
    }
    
    return data


def send_to_firebase(data):
    """Send sensor data to Firebase Realtime Database."""
    try:
        response = requests.patch(FIREBASE_URL, json=data, timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Firebase update failed: {e}")
        return False


def main():
    """Main sensor simulation loop."""
    print("=" * 50)
    print("CNC Machine Sensor Simulator")
    print("=" * 50)
    print(f"Target: {FIREBASE_URL}")
    print(f"Interval: {POLL_INTERVAL} seconds")
    print("-" * 50)
    
    consecutive_failures = 0
    
    while True:
        try:
            # Generate data
            data = generate_sensor_data()
            
            # Send to Firebase
            success = send_to_firebase(data)
            
            if success:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{timestamp}] "
                    f"RPM: {data['rpm']} | "
                    f"Temp: {data['temperature']}°C | "
                    f"Vib: {data['vibration']}mm/s | "
                    f"Cur: {data['current']}A"
                )
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    print("[WARNING] Multiple consecutive failures")
            
        except KeyboardInterrupt:
            print("\n[Shutdown] Simulator stopped by user")
            break
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            consecutive_failures += 1
        
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()