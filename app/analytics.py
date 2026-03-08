"""
Analytics module for CNC Machine Monitoring.
Provides real-time statistics, anomaly detection, and engineering metrics.
"""

import numpy as np
from collections import deque
import time

# HISTORY BUFFERS (fixed-size for efficiency)
HISTORY_SIZE = 100

rpm_history = deque(maxlen=HISTORY_SIZE)
vibration_history = deque(maxlen=HISTORY_SIZE)
current_history = deque(maxlen=HISTORY_SIZE)
temperature_history = deque(maxlen=HISTORY_SIZE)
energy_history = deque(maxlen=HISTORY_SIZE)

# CONFIGURABLE THRESHOLDS
# RPM
RPM_MIN = 500
RPM_MAX = 2000
RPM_NOMINAL = 1250
RPM_OPTIMAL_MIN = 1000   # Optimal operating window lower bound
RPM_OPTIMAL_MAX = 1300   # Optimal operating window upper bound

# Temperature (Celsius)
TEMP_AMBIENT = 25         # Ambient / workshop temperature
TEMP_IDEAL = 65           # Ideal spindle operating temperature
TEMP_OPTIMAL = 70         # Upper optimal temperature
TEMP_NORMAL_MAX = 70
TEMP_WARNING = 75
TEMP_CRITICAL = 95        # Realistic critical threshold for CNC spindle

# Vibration (mm/s)
VIBRATION_NORMAL_MAX = 2.0  # Safe baseline
VIBRATION_WARNING = 3.5
VIBRATION_ANOMALY_THRESHOLD = 1.6  # 60% above baseline

# Current (Amps)
CURRENT_NOMINAL = 8.5
CURRENT_OVERLOAD = 4.0    # Delta above nominal considered overload
CURRENT_OVERLOAD_THRESHOLD = 1.3  # 30% above average
CURRENT_MAX = 12.0

# Tool wear simulation
TOOL_MAX_HOURS = 50  # Tool replacement after ~50 hours of operation
tool_runtime_minutes = 0  # Accumulated runtime

# Energy calculation (DC / single-phase spindle motor)
VOLTAGE = 48              # DC bus voltage typical for CNC spindle
MAX_ENERGY_PER_PART = 0.5 # kWh clamp for sanity

# Cycle-time estimation from RPM transitions
_last_cycle_timestamp = None
_estimated_cycle_seconds = 60  # initial default, updated at runtime

# Timestamps for utilization
session_start_time = time.time()
active_cycles = 0
total_cycles = 0


# VALIDATION
def validate_numeric(value, name, min_val=None, max_val=None, default=0):
    """Validate and sanitize numeric input."""
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            print(f"[Analytics] Warning: {name} is NaN/Inf, using default {default}")
            return default
        if min_val is not None and val < min_val:
            return min_val
        if max_val is not None and val > max_val:
            return max_val
        return val
    except (TypeError, ValueError):
        print(f"[Analytics] Warning: Invalid {name}={value}, using default {default}")
        return default


 
# RPM ANALYTICS
 
def update_rpm(rpm):
    """
    Update RPM history and calculate statistics.
    Returns: (avg_rpm, stability_percentage)
    Stability is 0-1 where 1 = perfectly stable.
    """
    rpm = validate_numeric(rpm, "rpm", RPM_MIN, RPM_MAX, 1200)
    rpm_history.append(rpm)
    
    if len(rpm_history) < 2:
        return rpm, 1.0
    
    avg_rpm = np.mean(rpm_history)
    std_rpm = np.std(rpm_history)
    
    if avg_rpm > 0:
        cv = std_rpm / avg_rpm
        stability = max(0, min(1, 1 - cv))
    else:
        stability = 0
    
    return float(avg_rpm), float(stability)


def get_rpm_history():
    """Return RPM history as list (for Firebase/charts)."""
    return list(rpm_history)


 
# TEMPERATURE ANALYTICS
 
def update_temperature(temperature):
    """Update temperature history."""
    temperature = validate_numeric(temperature, "temperature", 0, 150, 70)
    temperature_history.append(temperature)
    return temperature


def get_temperature_history():
    """Return temperature history as list."""
    return list(temperature_history)


def detect_temperature_alert(temperature):
    """
    Detect temperature alerts.
    Returns: (alert_bool, risk_level)
    risk_level: "low", "medium", "high", "critical"
    """
    temperature = validate_numeric(temperature, "temperature", 0, 150, 70)
    
    if temperature >= TEMP_CRITICAL:
        return True, "critical"
    elif temperature >= TEMP_WARNING:
        return True, "high"
    elif temperature >= TEMP_NORMAL_MAX:
        return False, "medium"
    else:
        return False, "low"


def calculate_cooling_efficiency(temperature):
    """
    Calculate cooling system efficiency (0-100%).
    Uses thermal-delta formula:
      efficiency = 100 * (ideal_temp - ambient) / (current_temp - ambient)
    When current_temp equals ideal_temp the efficiency is 100%.
    Higher current_temp lowers the ratio (cooling can't keep up).
    """
    temperature = validate_numeric(temperature, "temperature", 0, 150, 70)

    delta_current = temperature - TEMP_AMBIENT
    delta_ideal = TEMP_IDEAL - TEMP_AMBIENT

    if delta_current <= 0:
        # At or below ambient – cooling has no work to do
        return 100.0

    efficiency = 100.0 * delta_ideal / delta_current
    return max(0.0, min(100.0, efficiency))


def calculate_overheating_risk(temperature):
    """
    Calculate overheating risk (0-100%).
    Normalized distance between optimal and critical temperature.
    risk = clamp((temp - optimal) / (critical - optimal), 0, 1) * 100
    """
    temperature = validate_numeric(temperature, "temperature", 0, 150, 70)

    span = TEMP_CRITICAL - TEMP_OPTIMAL
    if span <= 0:
        return 100.0 if temperature >= TEMP_CRITICAL else 0.0

    risk = (temperature - TEMP_OPTIMAL) / span
    return max(0.0, min(100.0, risk * 100))


 
# VIBRATION ANALYTICS
 
def update_vibration(vibration):
    """Update vibration history."""
    vibration = validate_numeric(vibration, "vibration", 0, 100, 1.5)
    vibration_history.append(vibration)
    return vibration


def get_vibration_history():
    """Return vibration history as list."""
    return list(vibration_history)


def detect_vibration_anomaly(vibration):
    """
    Detect abnormal vibration levels.
    Returns True if vibration exceeds threshold above baseline.
    """
    vibration = validate_numeric(vibration, "vibration", 0, 100, 1.5)
    
    if len(vibration_history) < 5:
        return vibration > VIBRATION_WARNING
    
    baseline = np.mean(vibration_history)
    
    if baseline > 0 and vibration > baseline * VIBRATION_ANOMALY_THRESHOLD:
        return True
    
    if vibration > VIBRATION_WARNING:
        return True
    
    return False


def detect_vibration_alert(vibration):
    """
    Detect vibration alert with severity.
    Returns: (alert_bool, severity)
    """
    vibration = validate_numeric(vibration, "vibration", 0, 100, 1.5)
    
    if vibration >= VIBRATION_WARNING:
        return True, "high"
    elif vibration >= VIBRATION_NORMAL_MAX:
        return True, "medium"
    else:
        return False, "low"


def calculate_bearing_wear_probability(vibration, rpm):
    """
    Estimate bearing wear probability based on vibration and RPM.
    Higher vibration at normal RPM suggests wear.
    Returns: 0-100%
    """
    vibration = validate_numeric(vibration, "vibration", 0, 100, 1.5)
    rpm = validate_numeric(rpm, "rpm", RPM_MIN, RPM_MAX, 1200)
    
    # Base probability from vibration level
    if vibration <= 1.5:
        base_prob = 5
    elif vibration <= 2.0:
        base_prob = 10
    elif vibration <= 2.5:
        base_prob = 20
    elif vibration <= 3.0:
        base_prob = 35
    elif vibration <= 3.5:
        base_prob = 50
    else:
        base_prob = 70
    
    # Adjust based on RPM (high vibration at low RPM = worse)
    rpm_factor = 1.0
    if rpm < 1000 and vibration > 2.0:
        rpm_factor = 1.3  # 30% more likely if low RPM but high vibration
    
    # Factor in history variance
    if len(vibration_history) >= 10:
        variance = np.var(vibration_history)
        if variance > 0.5:  # High variance suggests intermittent issues
            base_prob += 10
    
    return max(0, min(100, base_prob * rpm_factor))


 
# CURRENT ANALYTICS
 
def update_current(current):
    """Update current history."""
    current = validate_numeric(current, "current", 0, 100, 8.0)
    current_history.append(current)
    return current


def detect_current_overload(current):
    """
    Detect current overload conditions.
    Returns True if current exceeds threshold above average.
    """
    current = validate_numeric(current, "current", 0, 100, 8.0)
    
    if len(current_history) < 5:
        return current > CURRENT_MAX
    
    avg_current = np.mean(current_history)
    
    if avg_current > 0 and current > avg_current * CURRENT_OVERLOAD_THRESHOLD:
        return True
    
    if current > CURRENT_MAX:
        return True
    
    return False


def detect_current_alert(current):
    """
    Detect current alert with severity.
    Returns: (alert_bool, severity)
    """
    current = validate_numeric(current, "current", 0, 100, 8.0)
    
    if current >= CURRENT_MAX:
        return True, "critical"
    elif current >= CURRENT_NOMINAL * 1.2:
        return True, "high"
    elif current >= CURRENT_NOMINAL * 1.1:
        return False, "medium"
    else:
        return False, "low"


 
# ENERGY ANALYTICS
 
def estimate_cycle_time():
    """Return the current estimated cycle duration in seconds."""
    return _estimated_cycle_seconds


def update_cycle_time():
    """
    Estimate cycle duration from timestamps between RPM transitions.
    Called each polling iteration; updates the module-level estimate.
    """
    global _last_cycle_timestamp, _estimated_cycle_seconds

    now = time.time()
    if _last_cycle_timestamp is not None:
        elapsed = now - _last_cycle_timestamp
        # Only accept plausible cycle durations (2 s – 600 s)
        if 2 <= elapsed <= 600:
            _estimated_cycle_seconds = elapsed
    _last_cycle_timestamp = now


def calculate_energy_per_part(current, cycle_time_seconds=None):
    """
    Calculate energy consumption per part (kWh).
    Uses DC / single-phase formula: P = V * I
    Energy (kWh) = P * cycle_time / 3600 / 1000
    """
    current = validate_numeric(current, "current", 0, 100, 8.0)

    if cycle_time_seconds is None:
        cycle_time_seconds = _estimated_cycle_seconds

    # DC / single-phase power
    power_watts = VOLTAGE * current

    # Energy for one cycle (kWh)
    energy_kwh = (power_watts * cycle_time_seconds) / 3_600_000

    # Clamp unrealistic values
    energy_kwh = max(0, min(MAX_ENERGY_PER_PART, energy_kwh))

    energy_history.append(energy_kwh)
    return round(energy_kwh, 4)


def get_energy_history():
    """Return energy consumption history."""
    return list(energy_history)


 
# TOOL WEAR & UTILIZATION
 
def update_tool_wear(cycle_minutes=0.0833):
    """
    Update tool runtime and calculate wear percentage.
    Default: ~5 seconds per cycle = 0.0833 minutes
    Returns: (wear_percent, remaining_hours)
    """
    global tool_runtime_minutes
    
    tool_runtime_minutes += cycle_minutes
    
    # Calculate wear percentage
    total_tool_life_minutes = TOOL_MAX_HOURS * 60
    wear_percent = (tool_runtime_minutes / total_tool_life_minutes) * 100
    wear_percent = min(100, wear_percent)
    
    # Calculate remaining hours
    remaining_minutes = total_tool_life_minutes - tool_runtime_minutes
    remaining_hours = max(0, remaining_minutes / 60)
    
    # Reset if tool is "replaced" (over 100%)
    if wear_percent >= 100:
        tool_runtime_minutes = 0
        wear_percent = 0
        remaining_hours = TOOL_MAX_HOURS
    
    return round(wear_percent, 1), round(remaining_hours, 1)


def get_suggested_tool_replacement_hours():
    """Get hours until suggested tool replacement."""
    total_tool_life_minutes = TOOL_MAX_HOURS * 60
    remaining_minutes = total_tool_life_minutes - tool_runtime_minutes
    return max(0, round(remaining_minutes / 60, 1))


def calculate_machine_utilization():
    """
    Calculate machine utilization percentage.
    Based on active cycles vs total possible cycles.
    """
    global active_cycles, total_cycles
    
    total_cycles += 1
    
    # Assume machine is "active" if RPM history shows movement
    if len(rpm_history) > 0 and rpm_history[-1] > RPM_MIN:
        active_cycles += 1
    
    if total_cycles > 0:
        utilization = (active_cycles / total_cycles) * 100
    else:
        utilization = 0
    
    return round(utilization, 1)


 
# ANOMALY DETECTION
 
def calculate_anomaly_score(temperature, vibration, current, rpm):
    """
    Calculate overall anomaly score (0-1) using normalised deviations
    from safe operating ranges.

    Component scores (each 0-1):
      temp_score    = clamp((temp - 70) / 25)
      vib_score     = clamp((vibration - 2) / 6)
      current_score = clamp((current - CURRENT_NOMINAL) / CURRENT_OVERLOAD)
      rpm_score     = clamp(rpm_std_dev / 10)

    Weighted sum:
      0.35 * temp + 0.30 * vib + 0.20 * current + 0.15 * rpm
    """
    temperature = validate_numeric(temperature, "temperature", 0, 150, 70)
    vibration = validate_numeric(vibration, "vibration", 0, 100, 1.5)
    current = validate_numeric(current, "current", 0, 100, 8.0)

    def _clamp01(v):
        return max(0.0, min(1.0, v))

    temp_score = _clamp01((temperature - 70) / 25)
    vib_score = _clamp01((vibration - 2) / 6)
    current_score = _clamp01((current - CURRENT_NOMINAL) / CURRENT_OVERLOAD)

    # RPM instability from rolling std-dev
    if len(rpm_history) >= 5:
        rpm_score = _clamp01(float(np.std(rpm_history)) / 10)
    else:
        rpm_score = 0.0

    score = (0.35 * temp_score
             + 0.30 * vib_score
             + 0.20 * current_score
             + 0.15 * rpm_score)

    return round(_clamp01(score), 3)


 
# HEALTH SCORE
 
def calculate_health_score(temperature, vibration, current, rpm_stability, anomaly_score):
    """
    Calculate overall machine health score (0-100).
    Weighted risk model using normalised sensor deviations.

    health = 100 - temp_risk*30 - vib_risk*25 - curr_risk*20
                 - rpm_instability*15 - anomaly*10
    """
    temperature = validate_numeric(temperature, "temperature", 0, 150, 70)
    vibration = validate_numeric(vibration, "vibration", 0, 100, 1.5)
    current = validate_numeric(current, "current", 0, 100, 8.0)
    rpm_stability = validate_numeric(rpm_stability, "rpm_stability", 0, 1, 1.0)
    anomaly_score = validate_numeric(anomaly_score, "anomaly_score", 0, 1, 0.0)

    def _clamp01(v):
        return max(0.0, min(1.0, v))

    temp_risk = _clamp01((temperature - TEMP_OPTIMAL) / (TEMP_CRITICAL - TEMP_OPTIMAL))
    vib_risk = _clamp01((vibration - VIBRATION_NORMAL_MAX) / (VIBRATION_WARNING - VIBRATION_NORMAL_MAX))
    curr_risk = _clamp01((current - CURRENT_NOMINAL) / (CURRENT_MAX - CURRENT_NOMINAL))
    rpm_instability = _clamp01(1.0 - rpm_stability)  # stability 1=stable → instability 0

    health = (100
              - temp_risk * 30
              - vib_risk * 25
              - curr_risk * 20
              - rpm_instability * 15
              - anomaly_score * 10)

    return float(max(0.0, min(100.0, health)))


 
# COMPREHENSIVE ANALYTICS
 
def compute_all_analytics(rpm, temperature, vibration, current, failure_probability):
    """
    Compute all analytics in one call.
    Returns a dict with all computed values.
    """
    # Update histories
    update_temperature(temperature)
    update_vibration(vibration)
    update_current(current)
    avg_rpm, rpm_stability = update_rpm(rpm)
    
    # Alerts
    temp_alert, temp_severity = detect_temperature_alert(temperature)
    vib_alert, vib_severity = detect_vibration_alert(vibration)
    curr_alert, curr_severity = detect_current_alert(current)
    
    # Engineering metrics
    tool_wear, remaining_hours = update_tool_wear()
    utilization = calculate_machine_utilization()
    update_cycle_time()
    energy = calculate_energy_per_part(current)
    cooling_eff = calculate_cooling_efficiency(temperature)
    overheat_risk = calculate_overheating_risk(temperature)
    bearing_wear = calculate_bearing_wear_probability(vibration, rpm)
    anomaly = calculate_anomaly_score(temperature, vibration, current, rpm)

    # Health score (weighted risk model)
    health = calculate_health_score(
        temperature,
        vibration,
        current,
        rpm_stability,
        anomaly
    )
    
    return {
        # Core metrics
        "health_score": round(health, 1),
        "failure_probability": round(failure_probability, 4),
        "avg_rpm": round(avg_rpm, 1),
        "rpm_stability": round(rpm_stability, 4),
        
        # Engineering metrics
        "tool_wear_percent": tool_wear,
        "machine_utilization": utilization,
        "energy_per_part": energy,
        "cooling_efficiency": round(cooling_eff, 1),
        "overheating_risk": round(overheat_risk, 1),
        "bearing_wear_probability": round(bearing_wear, 1),
        "anomaly_score": anomaly,
        "suggested_tool_replacement_hours": remaining_hours,
        
        # Alerts
        "alerts": {
            "temperature_alert": temp_alert,
            "temperature_severity": temp_severity,
            "vibration_alert": vib_alert,
            "vibration_severity": vib_severity,
            "current_alert": curr_alert,
            "current_severity": curr_severity
        },
        
        # Trends (last N readings)
        "trends": {
            "rpm_history": get_rpm_history()[-20:],  # Last 20 for charts
            "vibration_history": get_vibration_history()[-20:],
            "temperature_history": get_temperature_history()[-20:],
            "energy_history": get_energy_history()[-20:]
        }
    }


 
# UTILITIES
 
def reset_history():
    """Clear all history buffers (useful for testing)."""
    global tool_runtime_minutes, active_cycles, total_cycles
    global _last_cycle_timestamp, _estimated_cycle_seconds
    
    rpm_history.clear()
    vibration_history.clear()
    current_history.clear()
    temperature_history.clear()
    energy_history.clear()
    
    tool_runtime_minutes = 0
    active_cycles = 0
    total_cycles = 0
    _last_cycle_timestamp = None
    _estimated_cycle_seconds = 60


def get_all_history():
    """Return all history data."""
    return {
        "rpm": list(rpm_history),
        "vibration": list(vibration_history),
        "temperature": list(temperature_history),
        "current": list(current_history),
        "energy": list(energy_history)
    }