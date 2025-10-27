# CardioClue - Cardiovascular Risk Assessment System

## 🫀 Overview

CardioClue is a comprehensive web application for cardiovascular risk assessment that combines machine learning predictions with user-friendly health questionnaires. The system uses a hybrid deep learning model to predict cardiovascular risk levels (Low, Medium, High) based on user health data.

## ✨ Features

### 🔐 User Authentication & Management
- Phone number-based login with OTP verification
- User profile management (name, age, gender, phone)
- Admin dashboard for viewing all users
- Session-based authentication

### 🧠 Machine Learning Integration
- **TensorFlow-based deep learning model** (.h5 format)
- **Hybrid architecture** combining numeric and categorical features
- **Risk level prediction**: Low, Medium, High
- **Confidence scores** and probability distributions
- Pre-trained model using health dataset with 10K+ samples

### 📊 Health Assessment
- Comprehensive health questionnaire with 16 questions
- Captures vital signs:
  - Weight, Height, BMI calculation
  - Blood pressure, Blood sugar levels
  - Heart rate and ECG data
  - Stress levels, Sleep hours
- Lifestyle factors:
  - Smoking status, Activity level
  - Diet quality, Alcohol consumption
  - Family history, Existing conditions

### 📈 Report Generation
- Interactive risk visualization with gauge charts
- Probability breakdown (Low/Medium/High percentages)
- Detailed health parameter summary
- PDF report download
- Share via SMS/WhatsApp simulation

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or 3.12
- MongoDB (local or cloud instance)
- Windows with long path support enabled (for TensorFlow)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd FRONTEND-copy
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   SECRET_KEY=your-secret-key-here
   MONGODB_URI=mongodb://localhost:27017/
   MONGODB_DB=cardioclue
   PORT=5000
   ```

4. **Enable Windows Long Path Support** (Required for TensorFlow)
   
   **Option 1: Using the batch file** (Run as Administrator)
   ```bash
   # Right-click on enable_long_paths.bat and select "Run as administrator"
   # Then restart your computer
   ```

   **Option 2: Manual registry edit**
   - Press `Win + X` → Select "Windows PowerShell (Admin)"
   - Run:
     ```powershell
     reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d 1 /f
     ```
   - Restart your computer

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to: `http://localhost:5000`

## 📁 Project Structure

```
FRONTEND-copy/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── models/
│   └── hybrid_model1.h5       # Pre-trained TensorFlow model
├── artifacts/
│   ├── scaler1.pkl            # Feature scaler
│   └── label_encoders1.pkl    # Categorical encoders
├── prediction/
│   └── Predict.py             # ML prediction module
├── utils/
│   ├── otp.py                 # OTP generation
│   ├── pdf.py                 # PDF report generation
│   └── risk.py                # Basic risk calculation
├── templates/
│   ├── base.html              # Base template
│   ├── login.html             # Login page
│   ├── verify.html            # OTP verification
│   ├── onboarding.html        # User onboarding
│   ├── user_dashboard.html    # User dashboard
│   ├── test.html              # Health questionnaire
│   └── report.html            # Risk report display
├── static/
│   └── css/
│       └── style.css          # Custom styles
├── exports/                    # Generated PDF reports
└── dataset/                    # Training dataset
```

## 🔧 Technology Stack

### Backend
- **Flask** 3.0.3 - Web framework
- **MongoDB** (PyMongo 4.8.0) - Database
- **TensorFlow** 2.16.1 - Deep learning
- **scikit-learn** 1.5.0 - Data preprocessing
- **pandas** 2.2.1 - Data handling
- **numpy** 1.26.4 - Numerical operations

### Frontend
- **Bootstrap** 5.3.3 - UI framework
- **Chart.js** - Data visualization
- **Canvas Confetti** - Visual feedback
- Custom CSS with gradient effects

### ML Model
- **Architecture**: Hybrid Neural Network
- **Input Features**: 11 numeric + 9 categorical
- **Output**: 3-class risk classification
- **Preprocessing**: StandardScaler + LabelEncoder

## 📱 Application Workflow

1. **Login** → User enters name and phone number
2. **Verification** → OTP sent to phone
3. **Onboarding** → User provides age and gender
4. **Dashboard** → View previous reports or start new test
5. **Questionnaire** → 16 health-related questions
6. **Prediction** → ML model processes input
7. **Report** → Displays risk level with confidence scores
8. **Download** → Generate PDF report

## 🎯 Key Features

### Machine Learning Model
- **Input Processing**: Converts form data to model-compatible format
- **Feature Engineering**: Handles numeric scaling and categorical encoding
- **ECG Parsing**: Extracts numeric values from ECG text input
- **BMI Calculation**: Automatic BMI calculation from weight/height
- **Fallback Mechanism**: Uses basic risk calculator if ML fails

### Risk Categories
- **Low Risk** (≤50 points): Normal cardiovascular health
- **Medium Risk** (51-100 points): Lifestyle modifications recommended
- **High Risk** (>100 points): Immediate medical consultation advised

### Admin Features
- View all registered users
- Access to user health reports
- Latest risk level display
- User management interface

## 🔒 Security & Privacy

- Session-based authentication
- OTP verification for user registration
- User data isolation (users can only view their own reports)
- Admin access control
- Environment variable configuration

## 📊 Model Details

### Model Architecture
```
Input Layer (Numeric) → Standard Scaler → Dense(128) → Dropout(0.3)
                                            ↓
Input Layer (Categorical) → Embedding → Flatten → Dense(64) → Dropout(0.2)
                                            ↓
                                             Concat
                                            ↓
                            Dense(32) → Dense(1, sigmoid) / Dense(3, softmax)
```

### Training Data
- **Dataset**: health_dataset_10k_with_risk_sample_with_ECG.xlsx
- **Sample Size**: 10,000+ records
- **Train/Val/Test Split**: 70%/15%/15%
- **Cross-validation**: Stratified splitting

### Model Performance
- Binary classification for high-risk detection
- Multi-class classification for granular risk levels
- Probability distributions for each risk category

## 🛠️ Troubleshooting

### TensorFlow Installation Issues

**Error**: `ML Model not available: No module named 'tensorflow.python'`

**Solution**: 
1. Ensure Windows long path support is enabled
2. Run `check_model.py` to verify installation:
   ```bash
   python check_model.py
   ```
3. Expected output:
   ```
   ✅ ML MODEL IS CONNECTED TO THE APP!
   ✅ TensorFlow is working
   ✅ Model files loaded successfully
   ```

### MongoDB Connection Issues

**Error**: `ServerSelectionTimeoutError`

**Solution**:
1. Ensure MongoDB is running locally or update `MONGODB_URI` in `.env`
2. For cloud MongoDB, use connection string format:
   ```env
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
   ```

### Version Warnings

**Warning**: `Trying to unpickle estimator from version 1.5.2 when using version 1.5.0`

**Solution**: This is a compatibility warning but doesn't affect functionality. To fix:
```bash
pip install scikit-learn==1.5.2
```

## 📈 Usage Examples

### Starting the Application
```bash
# Development mode (auto-reload on file changes)
python app.py

# Production mode
gunicorn -w 4 -b 0.0.0.0:5000 app:create_app()
```

### Testing ML Model
```bash
# Check if model is connected
python check_model.py

# Run standalone prediction test
python prediction/Predict.py
```

### Database Setup
```python
# MongoDB collections used:
- users: User profiles and authentication
- tests: Health test results and predictions
```

## 📝 Configuration

### Environment Variables
```env
SECRET_KEY=your-secret-key-change-this
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=cardioclue
PORT=5000
```

### Model Configuration
- Model file: `models/hybrid_model1.h5`
- Scaler: `artifacts/scaler1.pkl`
- Encoders: `artifacts/label_encoders1.pkl`

## 🎨 UI/UX Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Gradient Backgrounds**: Modern pink and indigo themes
- **Interactive Gauges**: Visual risk representation
- **Progress Tracking**: Questionnaire progress indicator
- **Toast Notifications**: User feedback for actions
- **Confetti Animation**: Celebration for low-risk results

## 🔄 Update Log

### Latest Version (Current)
- ✅ ML Model Integration
- ✅ TensorFlow 2.16.1 Support
- ✅ Long Path Support Configuration
- ✅ PDF Report Generation
- ✅ Admin Dashboard
- ✅ User Authentication System

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure Windows long path support is enabled
4. Run `python check_model.py` to diagnose ML model issues

## 📄 License

This project is part of a semester academic project. All rights reserved.

## 🙏 Acknowledgments

- TensorFlow team for deep learning capabilities
- Flask team for the web framework
- Bootstrap for UI components
- MongoDB for database support

---

## ✅ System Status

**Current Status**: ✅ **FULLY OPERATIONAL**

- ✅ ML Model: Connected and Working
- ✅ TensorFlow: Version 2.16.1
- ✅ Frontend: Fully Integrated
- ✅ Backend: Flask Running
- ✅ Database: MongoDB Connected
- ✅ Predictions: With Confidence Scores
- ✅ Reports: PDF Generation Active

**Last Updated**: October 2025

