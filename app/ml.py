import os
import joblib
import pandas as pd


# Paths  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "failure_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "feature_scaler.pkl")


#Feature Ranges AI4I Dataset

FEATURE_RANGES = {
    "air_temp_k": (295, 320),
    "process_temp_k": (300, 330),
    "rpm": (1000, 3000),
    "torque": (3, 80),
    "tool_wear": (0, 250)
}

FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

_model = None
_scaler = None


#   Model Loading  
def load_model():
    global _model, _scaler

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("ML model not found. Train the model first.")

    _model = joblib.load(MODEL_PATH)

    if os.path.exists(SCALER_PATH):
        _scaler = joblib.load(SCALER_PATH)
    else:
        _scaler = None

    return _model

def get_model():
    global _model

    if _model is None:
        load_model()

    return _model


#   Unit Conversion  
def celsius_to_kelvin(celsius):
    return celsius + 273.15


#   Torque Estimation  
def estimate_torque(current, rpm):

    if rpm <= 0:
        return 40

    torque = (current * 50) / 10

    if torque < 3:
        torque = 3
    elif torque > 80:
        torque = 80

    return torque


#   Feature Validation  
def clamp(value, min_v, max_v):
    return max(min_v, min(max_v, value))
def validate_features(air_temp_k, process_temp_k, rpm, torque, tool_wear):

    air_temp_k = clamp(air_temp_k, *FEATURE_RANGES["air_temp_k"])
    process_temp_k = clamp(process_temp_k, *FEATURE_RANGES["process_temp_k"])
    rpm = clamp(rpm, *FEATURE_RANGES["rpm"])
    torque = clamp(torque, *FEATURE_RANGES["torque"])
    tool_wear = clamp(tool_wear, *FEATURE_RANGES["tool_wear"])

    return air_temp_k, process_temp_k, rpm, torque, tool_wear


#   Failure Prediction  

def predict_failure(air_temp_k, process_temp_k, rpm, torque, tool_wear):

    model = get_model()

    features = validate_features(
        air_temp_k,
        process_temp_k,
        rpm,
        torque,
        tool_wear
    )

    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    if _scaler is not None:
        X = pd.DataFrame(_scaler.transform(X), columns=FEATURE_COLUMNS)

    if hasattr(model, "predict_proba"):
        prediction = model.predict_proba(X)
        probability = float(prediction[0][1])
    else:
        prediction = model.predict(X)
        probability = float(prediction[0])

    if probability < 0:
        probability = 0.0
    elif probability > 1:
        probability = 1.0

    return probability


#   Sensor Wrapper  

def predict_from_sensors(temperature_c, rpm, current, tool_wear_minutes=50):

    air_temp_k = celsius_to_kelvin(temperature_c)

    process_temp_k = air_temp_k + 10

    torque = estimate_torque(current, rpm)

    return predict_failure(
        air_temp_k=air_temp_k,
        process_temp_k=process_temp_k,
        rpm=rpm,
        torque=torque,
        tool_wear=tool_wear_minutes
    )