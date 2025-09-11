import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# Load model and preprocessors (support new structured paths, fallback to legacy)
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]

model_path = _first_existing(["models/hybrid_model1.h5"]) 
scaler_path = _first_existing(["artifacts/scaler1.pkl"]) 
encoders_path = _first_existing(["artifacts/label_encoders1.pkl"]) 
target_encoder_path = _first_existing(["artifacts/target_encoder1.pkl"]) 

model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(encoders_path)
target_encoder = None
if os.path.exists(target_encoder_path):
    try:
        target_encoder = joblib.load(target_encoder_path)
    except Exception:
        target_encoder = None

# Define columns (same as training)
numeric_cols = ["Age","SleepHours","Weight","Height","BMI",
                "BloodSugar","HeartRate","ECG_RA","ECG_LA","ECG_RL","StressLevel"]

categorical_cols = ["Gender","Smoker","ActivityLevel","Diet","Alcohol",
                    "FamilyHistory","HighBP","Diabetes","HeartDisease"]

def preprocess_single(sample):
    df = pd.DataFrame([sample])

    # --- Numeric ---
    X_num = scaler.transform(df[numeric_cols].fillna(0))

    # --- Categorical ---
    X_cat = []
    for col in categorical_cols:
        le = label_encoders[col]
        val = df[col].fillna("NA").astype(str)

        # Handle unseen categories by mapping them to a known class
        known_classes = set(le.classes_)
        val = val.map(lambda x: x if x in known_classes else le.classes_[0])

        X_cat.append(le.transform(val))
    X_cat = np.stack(X_cat, axis=1).astype(np.int32)

    # Format for model
    inputs = [X_num] + [X_cat[:, i].reshape(-1, 1) for i in range(X_cat.shape[1])]
    return inputs

def predict(sample):
    inputs = preprocess_single(sample)
    y = model.predict(inputs, verbose=0)
    # Binary vs multiclass
    if y.ndim == 2 and y.shape[1] > 1:
        probs = y[0]
        pred_class = int(np.argmax(probs))
        pred_label = None
        if target_encoder is not None and hasattr(target_encoder, 'inverse_transform'):
            try:
                pred_label = target_encoder.inverse_transform([pred_class])[0]
            except Exception:
                pred_label = None
        return (pred_label if pred_label is not None else pred_class), float(np.max(probs))
    else:
        prob = float(y[0][0])
        pred = int(prob > 0.5)
        return pred, prob

if __name__ == "__main__":
    print("=== Enter Patient Details ===")

    # Numeric inputs
    sample = {}
    for col in numeric_cols:
        val = float(input(f"Enter {col}: "))
        sample[col] = val

    # Categorical inputs
    for col in categorical_cols:
        val = input(f"Enter {col}: ")
        sample[col] = val

    # Predict
    pred, prob = predict(sample)
    print("\n=== Prediction Result ===")

    # Helper to convert probability to risk tier
    def risk_tier(p):
        if p >= 0.70:
            return "High Risk"
        if p >= 0.40:
            return "Moderate Risk"
        return "Low Risk"

    # Detect multiclass vs binary from model output shape
    is_multiclass = hasattr(model.output_shape, "__iter__") and len(model.output_shape) > 1 and model.output_shape[1] > 1

    if is_multiclass:
        percent = prob * 100.0
        print(f"Predicted class: {pred}")
        print(f"Confidence: {percent:.1f}%")
        print("Note: Risk tiers (Low/Moderate/High) apply to binary models. This model is multiclass.")
    else:
        percent = prob * 100.0
        tier = risk_tier(prob)
        human = "Yes" if pred == 1 else "No"
        print(f"Heart attack risk: {tier}")
        print(f"Probability: {percent:.1f}%")
        print(f"High-risk prediction: {human}")
