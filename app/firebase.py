import os
import firebase_admin
from firebase_admin import credentials, db


#   Firebase Configuration  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREBASE_KEY_PATH = os.path.join(BASE_DIR, "firebase_key.json")
DATABASE_URL = "https://cnc-machine-8664d-default-rtdb.asia-southeast1.firebasedatabase.app"

ROOT_PATH = "cnc_machine"
SENSORS_PATH = f"{ROOT_PATH}/sensors"
ANALYTICS_PATH = f"{ROOT_PATH}/analytics"

AI_ROOT_PATH = "cnc_machine_ai"
AI_PREDICTIONS_PATH = AI_ROOT_PATH


#Firebase Initialization  

if not firebase_admin._apps:
    if not os.path.exists(FIREBASE_KEY_PATH):
        raise FileNotFoundError(f"Firebase key not found: {FIREBASE_KEY_PATH}")
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        "databaseURL": DATABASE_URL
    })


#Sensor Data  

def get_sensor_data():
    ref = db.reference(SENSORS_PATH)
    data = ref.get()
    if data:
        return data
    return None


#AI Predictions  

def update_ai_predictions(data):
    analytics_data = {
        "avg_rpm": data.get("avg_rpm"),
        "rpm_stability": data.get("rpm_stability"),
        "tool_wear_percent": data.get("tool_wear_percent"),
        "machine_utilization": data.get("machine_utilization"),
        "energy_per_part": data.get("energy_per_part"),
        "cooling_efficiency": data.get("cooling_efficiency"),
        "overheating_risk": data.get("overheating_risk"),
        "bearing_wear_probability": data.get("bearing_wear_probability"),
        "suggested_tool_replacement_hours": data.get("suggested_tool_replacement_hours"),
        "trends": data.get("trends", {}),
        "last_updated": data.get("last_updated")
    }

    ai_prediction_data = {
        "health_score": data.get("health_score"),
        "failure_probability": data.get("failure_probability"),
        "anomaly_score": data.get("anomaly_score"),
        "tool_wear_percent": data.get("tool_wear_percent"),
        "alerts": data.get("alerts", {}),
        "last_updated": data.get("last_updated")
    }
    db.reference(ANALYTICS_PATH).set(analytics_data)
    db.reference(AI_PREDICTIONS_PATH).set(ai_prediction_data)
    return True


# Update Trends  

def update_trends_only(trends_data):
    ref = db.reference(f"{ANALYTICS_PATH}/trends")
    ref.set(trends_data)
    return True


#Update Alerts  

def update_alerts_only(alerts_data):
    ref = db.reference(f"{AI_PREDICTIONS_PATH}/alerts")
    ref.set(alerts_data)
    return True


#Read AI Predictions  

def get_ai_predictions():
    analytics_data = db.reference(ANALYTICS_PATH).get() or {}
    ai_data = db.reference(AI_PREDICTIONS_PATH).get() or {}
    merged_data = {**analytics_data, **ai_data}
    if merged_data:
        return merged_data
    return None


#Connection Test  

def test_connection():
    ref = db.reference("/")
    ref.get()

    return True

#Debug Database Structure  

def get_database_structure():
    ref = db.reference("/")
    return ref.get()