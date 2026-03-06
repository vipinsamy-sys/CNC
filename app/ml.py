"""
Machine Learning module for CNC failure prediction.
Uses AI4I 2020 Predictive Maintenance model.
"""

import os
import joblib
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "failure_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "feature_scaler.pkl")

# AI4I 2020 Dataset feature ranges (for validation)
FEATURE_RANGES = {
    "air_temp_k": (295, 320),       # Kelvin
    "process_temp_k": (300, 330),   # Kelvin
    "rpm": (1000, 3000),
    "torque": (3, 80),              # Nm
    "tool_wear": (0, 250)           # minutes
}

# Feature column names (must match training data)
FEATURE_COLUMNS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

# Global model and scaler
_model = None
_scaler = None


def load_model():
    """Load the trained model and scaler."""
    global _model, _scaler
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run training/train_failure_model.py first."
        )
    
    _model = joblib.load(MODEL_PATH)
    
    # Load scaler if exists (optional, for production)
    if os.path.exists(SCALER_PATH):
        _scaler = joblib.load(SCALER_PATH)
        print("[ML] Model and scaler loaded successfully")
    else:
        _scaler = None
        print("[ML] Model loaded (no scaler found)")
    
    return _model


def get_model():
    """Get model, loading if necessary."""
    global _model
    if _model is None:
        load_model()
    return _model


def celsius_to_kelvin(celsius):
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def estimate_torque(current, rpm):
    """
    Estimate torque from current and RPM.
    Torque ∝ Current / RPM (simplified motor model)
    Calibrated for typical CNC spindle motor.
    """
    # P = Torque * Angular velocity
    # Current is proportional to torque in DC motors
    # This is a rough approximation; adjust multiplier based on your motor
    if rpm <= 0:
        return 40  # Default torque
    
    # Typical CNC spindle: ~40-50 Nm at full load
    torque = (current * 50) / 10  # Scale factor based on motor specs
    return max(3, min(80, torque))  # Clamp to realistic range


def validate_features(air_temp_k, process_temp_k, rpm, torque, tool_wear):
    """Validate and clamp features to expected ranges."""
    def clamp(val, min_v, max_v, name):
        clamped = max(min_v, min(max_v, val))
        if clamped != val:
            print(f"[ML] Warning: {name}={val} clamped to {clamped}")
        return clamped
    
    air_temp_k = clamp(air_temp_k, *FEATURE_RANGES["air_temp_k"], "air_temp_k")
    process_temp_k = clamp(process_temp_k, *FEATURE_RANGES["process_temp_k"], "process_temp_k")
    rpm = clamp(rpm, *FEATURE_RANGES["rpm"], "rpm")
    torque = clamp(torque, *FEATURE_RANGES["torque"], "torque")
    tool_wear = clamp(tool_wear, *FEATURE_RANGES["tool_wear"], "tool_wear")
    
    return air_temp_k, process_temp_k, rpm, torque, tool_wear


def predict_failure(air_temp_k, process_temp_k, rpm, torque, tool_wear):
    """
    Predict machine failure probability.
    
    Args:
        air_temp_k: Air temperature in Kelvin
        process_temp_k: Process temperature in Kelvin
        rpm: Rotational speed in rev/min
        torque: Torque in Nm
        tool_wear: Tool wear in minutes
    
    Returns:
        failure_probability (float): 0.0 to 1.0
    """
    try:
        model = get_model()
        
        # Validate features
        features = validate_features(
            air_temp_k, process_temp_k, rpm, torque, tool_wear
        )
        
        # Create DataFrame with proper column names to avoid sklearn warning
        X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        
        # Apply scaling if available
        if _scaler is not None:
            X = pd.DataFrame(_scaler.transform(X), columns=FEATURE_COLUMNS)
        
        # Predict probability
        if hasattr(model, 'predict_proba'):
            prediction = model.predict_proba(X)
            failure_probability = float(prediction[0][1])
        else:
            # Binary prediction fallback
            prediction = model.predict(X)
            failure_probability = float(prediction[0])
        
        # Ensure valid probability
        failure_probability = max(0.0, min(1.0, failure_probability))
        
        return failure_probability
        
    except Exception as e:
        print(f"[ML] Prediction error: {e}")
        return 0.5  # Return moderate risk on error


def predict_from_sensors(temperature_c, rpm, current, tool_wear_minutes=50):
    """
    Convenience function to predict failure from raw sensor data.
    Handles unit conversions automatically.
    
    Args:
        temperature_c: Temperature in Celsius (from sensor)
        rpm: RPM from sensor
        current: Current in Amps (from sensor)
        tool_wear_minutes: Estimated tool wear (default 50 min)
    
    Returns:
        failure_probability (float): 0.0 to 1.0
    """
    # Convert Celsius to Kelvin
    air_temp_k = celsius_to_kelvin(temperature_c)
    
    # Process temp is typically ~10K higher than air temp
    process_temp_k = air_temp_k + 10
    
    # Estimate torque from current
    torque = estimate_torque(current, rpm)
    
    return predict_failure(
        air_temp_k=air_temp_k,
        process_temp_k=process_temp_k,
        rpm=rpm,
        torque=torque,
        tool_wear=tool_wear_minutes
    )