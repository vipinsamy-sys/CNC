"""
Training script for CNC Machine Failure Prediction Model.
Uses AI4I 2020 Predictive Maintenance Dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Paths (relative to this script location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "datasets", "ai4i2020.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "failure_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")

# Feature columns from AI4I 2020 dataset
FEATURE_COLUMNS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

TARGET_COLUMN = 'Machine failure'


def load_dataset():
    """Load and validate the AI4I dataset."""
    print(f"Loading dataset from: {DATASET_PATH}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        sys.exit(1)
    
    data = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {len(data)} samples")
    
    # Validate columns
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in data.columns]
    if missing_cols:
        print(f"[ERROR] Missing columns: {missing_cols}")
        print(f"Available columns: {list(data.columns)}")
        sys.exit(1)
    
    return data


def train_model():
    """Train the failure prediction model."""
    print("=" * 50)
    print("CNC Machine Failure Model Training")
    print("=" * 50)
    
    # Load data
    data = load_dataset()
    
    # Extract features and target
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()
    
    # Print class distribution
    print(f"\nClass distribution:")
    print(f"  No failure (0): {(y == 0).sum()} samples")
    print(f"  Failure (1):    {(y == 1).sum()} samples")
    print(f"  Failure rate:   {(y == 1).mean() * 100:.2f}%")
    
    # Handle any missing values
    if X.isnull().any().any():
        print("\nWarning: Missing values found, filling with median")
        X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    
    # Feature scaling
    print("\nApplying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Cross-validation
    print("\nCross-validation (5-fold)...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
    print(f"  F1 scores: {cv_scores}")
    print(f"  Mean F1:   {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Test set evaluation
    print("\nTest set evaluation:")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    
    print("Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    print("\nFeature importance:")
    for name, importance in sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {name}: {importance:.4f}")
    
    # Save model and scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved: {SCALER_PATH}")
    
    # Save feature info for reference
    feature_info = {
        'feature_columns': FEATURE_COLUMNS,
        'feature_means': scaler.mean_.tolist(),
        'feature_stds': scaler.scale_.tolist()
    }
    joblib.dump(feature_info, os.path.join(MODELS_DIR, "feature_info.pkl"))
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    
    return model, scaler


if __name__ == "__main__":
    train_model()