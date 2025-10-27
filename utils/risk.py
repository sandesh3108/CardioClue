from typing import Dict, Any


def compute_risk_level(responses: Dict[str, Any]) -> str:
    score = 0

    if responses.get("Smoker") in ("Yes", "Occasionally"):
        score += 2 if responses.get("Smoker") == "Yes" else 1
    if responses.get("ActivityLevel") in ("Sedentary", "Light"):
        score += 2 if responses.get("ActivityLevel") == "Sedentary" else 1
    if responses.get("Diet") in ("Unhealthy", "Average"):
        score += 2 if responses.get("Diet") == "Unhealthy" else 1
    if responses.get("Alcohol") in ("Regularly", "Daily"):
        score += 1
    if float(responses.get("SleepHours") or 0) < 6:
        score += 1
    if responses.get("FamilyHistory") == "Yes":
        score += 2
    if responses.get("HighBP") == "Yes":
        score += 2
    if responses.get("Diabetes") == "Yes":
        score += 2
    if responses.get("HeartDisease") == "Yes":
        score += 3
    if int(responses.get("StressLevel") or 0) >= 7:
        score += 1

    bmi = float(responses.get("BMI") or 0)
    if bmi >= 30:
        score += 2
    elif bmi >= 25:
        score += 1

    sugar = float(responses.get("BloodSugar") or 0)
    if sugar >= 180:
        score += 2
    elif sugar >= 140:
        score += 1

    hr = int(responses.get("HeartRate") or 0)
    if hr >= 100 or hr <= 50:
        score += 1

    # Rough thresholds
    if score >= 9:
        return "High"
    if score >= 5:
        return "Medium"
    return "Low"


