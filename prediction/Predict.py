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

model_path = _first_existing(["models/hybrid_model.h5", "models/hybrid_model1.h5"]) 
scaler_path = _first_existing(["artifacts/scaler.pkl", "artifacts/scaler1.pkl"]) 
encoders_path = _first_existing(["artifacts/label_encoders.pkl", "artifacts/label_encoders1.pkl"]) 
target_encoder_path = _first_existing(["artifacts/target_encoder.pkl", "artifacts/target_encoder1.pkl"]) 

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

# Risk categorization function (matches training)
def categorize_risk(risk_value):
    if risk_value <= 50:
        return "Low"
    elif risk_value <= 100:
        return "Medium"
    else:
        return "High"

def compute_risk_score(sample):
    """Compute raw risk score from features (for reference)"""
    # This is a simplified risk calculation based on common factors
    risk_score = 0
    
    # Age factor
    age = sample.get('Age', 0)
    if age > 65:
        risk_score += 30
    elif age > 50:
        risk_score += 20
    elif age > 35:
        risk_score += 10
    
    # BMI factor
    bmi = sample.get('BMI', 0)
    if bmi > 30:
        risk_score += 25
    elif bmi > 25:
        risk_score += 15
    
    # Blood pressure
    if sample.get('HighBP', '').lower() in ['yes', 'true', '1']:
        risk_score += 20
    
    # Diabetes
    if sample.get('Diabetes', '').lower() in ['yes', 'true', '1']:
        risk_score += 25
    
    # Smoking
    if sample.get('Smoker', '').lower() in ['yes', 'true', '1']:
        risk_score += 30
    
    # Family history
    if sample.get('FamilyHistory', '').lower() in ['yes', 'true', '1']:
        risk_score += 15
    
    # Heart disease
    if sample.get('HeartDisease', '').lower() in ['yes', 'true', '1']:
        risk_score += 40
    
    # Blood sugar
    blood_sugar = sample.get('BloodSugar', 0)
    if blood_sugar > 140:
        risk_score += 20
    elif blood_sugar > 126:
        risk_score += 15
    
    # Heart rate
    heart_rate = sample.get('HeartRate', 0)
    if heart_rate > 100:
        risk_score += 10
    elif heart_rate < 60:
        risk_score += 5
    
    # Stress level
    stress = sample.get('StressLevel', 0)
    risk_score += stress * 2
    
    return max(0, min(100, risk_score))  # Clamp between 0-100

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
    
    # Handle 3-class risk categorization
    if y.ndim == 2 and y.shape[1] > 1:
        probs = y[0]
        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        
        # Map class indices to risk categories
        risk_categories = {0: "Low", 1: "Medium", 2: "High"}
        pred_label = risk_categories.get(pred_class, f"Class_{pred_class}")
        
        return pred_label, confidence, probs
    else:
        # Fallback for binary classification
        prob = float(y[0][0])
        pred = int(prob > 0.5)
        return ("High" if pred == 1 else "Low"), prob, [1-prob, prob]

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

    # Compute reference risk score
    risk_score = compute_risk_score(sample)
    risk_category_ref = categorize_risk(risk_score)
    
    # Predict using ML model
    pred_label, confidence, probs = predict(sample)
    print("\n=== Prediction Result ===")

    # Display ML model results
    print(f"ML Model Prediction: {pred_label}")
    print(f"Confidence: {confidence:.1%}")
    
    # Show reference calculation
    print(f"\nReference Calculation:")
    print(f"Computed Risk Score: {risk_score}/100")
    print(f"Risk Category: {risk_category_ref}")
    
    # Show probability distribution for all risk categories
    risk_categories = ["Low", "Medium", "High"]
    print("\nProbability Distribution:")
    for i, category in enumerate(risk_categories):
        if i < len(probs):
            print(f"  {category} Risk: {probs[i]:.1%}")
    
    # Check if ML prediction matches reference
    if pred_label == risk_category_ref:
        print(f"\n✅ ML model prediction matches reference calculation")
    else:
        print(f"\n⚠️  ML model prediction differs from reference calculation")
        print(f"   ML Model: {pred_label}, Reference: {risk_category_ref}")
    
    # Provide interpretation
    print(f"\nInterpretation:")
    if pred_label == "Low":
        print("  ✅ Low cardiovascular risk - Continue healthy lifestyle")
    elif pred_label == "Medium":
        print("  ⚠️  Medium cardiovascular risk - Consider lifestyle changes and regular monitoring")
    else:  # High
        print("  🚨 High cardiovascular risk - Consult healthcare provider immediately")
    
    # Additional recommendations based on risk level
    print(f"\nRecommendations:")
    if pred_label == "Low":
        print("  • Maintain current healthy habits")
        print("  • Regular exercise and balanced diet")
        print("  • Annual health checkups")
    elif pred_label == "Medium":
        print("  • Increase physical activity")
        print("  • Improve diet (reduce salt, saturated fats)")
        print("  • Monitor blood pressure and cholesterol")
        print("  • Consider stress management techniques")
    else:  # High
        print("  • Immediate consultation with cardiologist")
        print("  • Strict medication compliance if prescribed")
        print("  • Lifestyle modifications under medical supervision")
        print("  • Regular monitoring of vital signs")
