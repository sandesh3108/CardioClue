# How to Fix "ML Model not available" Error

## The Problem
TensorFlow cannot install because Windows has a 260-character limit on file paths. TensorFlow's files have paths longer than this limit.

## Solution Options

### Option 1: Create Virtual Environment (RECOMMENDED)

#### Step 1: Create Virtual Environment
```bash
# Navigate to your project directory
cd "C:\Users\sande\OneDrive\Documents\Sem7\MEGA Project\git"

# Create virtual environment
python -m venv cardioclue_env

# Activate virtual environment
# On Windows Command Prompt:
cardioclue_env\Scripts\activate
# On Windows PowerShell:
cardioclue_env\Scripts\Activate.ps1
# On Windows Git Bash:
source cardioclue_env/Scripts/activate
```

#### Step 2: Install Dependencies
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

#### Step 3: Run the Application
```bash
# Navigate to FRONTEND directory
cd FRONTEND

# Run the Flask app
python app.py
```

### Option 2: Enable Long Path Support (One-time setup)

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

### Option 3: Use Conda (Alternative - No restart needed)

1. Download and install Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Open Anaconda Prompt
3. Navigate to your project:
   ```bash
   cd "C:\Users\sande\OneDrive\Documents\Sem7\MEGA Project\git"
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

## Virtual Environment Management

### Creating Virtual Environments

#### Method 1: Using `venv` (Built-in Python)
```bash
# Create virtual environment
python -m venv cardioclue_env

# Activate (Windows)
cardioclue_env\Scripts\activate

# Activate (macOS/Linux)
source cardioclue_env/bin/activate

# Deactivate
deactivate
```

#### Method 2: Using `virtualenv`
```bash
# Install virtualenv first
pip install virtualenv

# Create virtual environment
virtualenv cardioclue_env

# Activate
cardioclue_env\Scripts\activate  # Windows
source cardioclue_env/bin/activate  # macOS/Linux
```

#### Method 3: Using `conda`
```bash
# Create environment with specific Python version
conda create -n cardioclue python=3.11

# Activate
conda activate cardioclue

# Deactivate
conda deactivate
```

### Virtual Environment Best Practices

1. **Always use virtual environments** for Python projects
2. **Name environments descriptively** (e.g., `cardioclue_env`, `ml_project`)
3. **Keep requirements.txt updated** when adding new packages
4. **Don't commit virtual environment folders** to git (add to .gitignore)
5. **Use same Python version** across development and production

### Common Commands

```bash
# Check if virtual environment is active
echo $VIRTUAL_ENV  # macOS/Linux
echo %VIRTUAL_ENV%  # Windows

# List installed packages
pip list

# Save current packages to requirements.txt
pip freeze > requirements.txt

# Install from requirements.txt
pip install -r requirements.txt

# Remove virtual environment
rm -rf cardioclue_env  # macOS/Linux
rmdir /s cardioclue_env  # Windows
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

