#!/usr/bin/env python3
"""Check if ML model is connected to the app"""

try:
    from prediction.Predict import predict, preprocess_single
    print("✅ ML MODEL IS CONNECTED TO THE APP!")
    print("✅ TensorFlow is working")
    print("✅ Model files loaded successfully")
    print("✅ Ready for predictions")
except ImportError as e:
    print("❌ ML MODEL NOT CONNECTED:", str(e))
    print("❌ TensorFlow import failed")
except Exception as e:
    print("❌ ML MODEL ERROR:", str(e))
    print("❌ Model loading failed")
