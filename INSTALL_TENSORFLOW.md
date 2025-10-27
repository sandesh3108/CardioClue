# How to Fix "ML Model not available" Error

## The Problem
TensorFlow cannot install because Windows has a 260-character limit on file paths. TensorFlow's files have paths longer than this limit.

## Solution Options

### Option 1: Enable Long Path Support (RECOMMENDED - One-time setup)

1. **Right-click on `enable_long_paths.bat` and select "Run as administrator"**
2. **Restart your computer**
3. After restart, open PowerShell in your project folder and run:
   ```powershell
   pip install -r requirements.txt
   ```
4. Then restart the app:
   ```powershell
   python app.py
   ```

### Option 2: Use Conda (Alternative - No restart needed)

1. Download and install Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Open Anaconda Prompt
3. Navigate to your project:
   ```bash
   cd "C:\Users\patil\OneDrive\Documents\sem 8\mega project\FRONTEND-copy"
   ```
4. Create environment and install:
   ```bash
   conda create -n cardioclue python=3.11 -y
   conda activate cardioclue
   conda install tensorflow -y
   pip install -r requirements.txt
   ```
5. Then run the app:
   ```bash
   python app.py
   ```

## How to Verify ML Model is Connected

After installing TensorFlow, when you start the app, you should see:
- ✅ `Model loaded successfully` or similar message
- ❌ Instead of `ML Model not available: No module named 'tensorflow.python'`

When you submit a health test, you'll see:
- ML Model Prediction with confidence scores
- Probability distributions (Low/Medium/High percentages)
- More accurate risk assessment

## Current Status
✅ Model files exist: `models/hybrid_model1.h5`
✅ Artifacts exist: `artifacts/scaler1.pkl`, `artifacts/label_encoders1.pkl`
✅ Integration code is complete
❌ TensorFlow not installed (due to Windows path limit)

## Troubleshooting
- If you still see the error after enabling long paths and restarting, try Option 2 (Conda)
- Make sure you're using the same Python environment where you installed TensorFlow

