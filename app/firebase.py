"""
Firebase connection module for CNC Machine Monitoring.
Handles all Firebase Realtime Database operations.
"""

import os
import firebase_admin
from firebase_admin import credentials, db

# Get absolute path to firebase_key.json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREBASE_KEY_PATH = os.path.join(BASE_DIR, "firebase_key.json")

DATABASE_URL = "https://cnc-machine-8664d-default-rtdb.asia-southeast1.firebasedatabase.app"

# Initialize Firebase only once
if not firebase_admin._apps:
    if not os.path.exists(FIREBASE_KEY_PATH):
        raise FileNotFoundError(f"Firebase key not found: {FIREBASE_KEY_PATH}")
    
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})


def get_sensor_data():
    """
    Fetch sensor data from Firebase.
    Returns dict with sensor values or None if unavailable.
    """
    try:
        ref = db.reference("cnc_machine")
        data = ref.get()
        return data if data else None
    except Exception as e:
        print(f"[Firebase] Error reading sensor data: {e}")
        return None


def update_ai_predictions(data):
    """
    Write AI predictions to Firebase.
    Handles nested structures (alerts, trends).
    Returns True on success, False on failure.
    """
    try:
        ref = db.reference("cnc_machine_ai")
        
        # Use set() for complete overwrite to ensure clean structure
        # This ensures nested objects like alerts and trends are properly updated
        ref.set(data)
        return True
    except Exception as e:
        print(f"[Firebase] Error updating predictions: {e}")
        return False


def update_trends_only(trends_data):
    """
    Update only the trends section (for efficiency).
    Use this if you want to update trends separately.
    """
    try:
        ref = db.reference("cnc_machine_ai/trends")
        ref.set(trends_data)
        return True
    except Exception as e:
        print(f"[Firebase] Error updating trends: {e}")
        return False


def update_alerts_only(alerts_data):
    """
    Update only the alerts section (for efficiency).
    """
    try:
        ref = db.reference("cnc_machine_ai/alerts")
        ref.set(alerts_data)
        return True
    except Exception as e:
        print(f"[Firebase] Error updating alerts: {e}")
        return False


def get_ai_predictions():
    """
    Fetch current AI predictions from Firebase.
    Useful for verification and testing.
    """
    try:
        ref = db.reference("cnc_machine_ai")
        data = ref.get()
        return data if data else None
    except Exception as e:
        print(f"[Firebase] Error reading AI predictions: {e}")
        return None


def test_connection():
    """
    Test Firebase connectivity.
    Returns True if connection is successful.
    """
    try:
        ref = db.reference("/")
        ref.get()
        return True
    except Exception as e:
        print(f"[Firebase] Connection test failed: {e}")
        return False


def get_database_structure():
    """
    Get the entire database structure (for debugging).
    """
    try:
        ref = db.reference("/")
        return ref.get()
    except Exception as e:
        print(f"[Firebase] Error getting structure: {e}")
        return None