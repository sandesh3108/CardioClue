# CardioClue - Technical Documentation

## 🏗️ System Architecture

### Component Diagram
```
┌─────────────────┐
│   User Browser  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Flask App     │◄──┐
│   (app.py)      │   │
└────────┬────────┘   │
         │           │
         ├───────────┼───────────┐
         │           │           │
         ▼           ▼           ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ MongoDB  │  │TensorFlow│  │ Utils    │
│ Database │  │   Model  │  │  (PDF)   │
└──────────┘  └──────────┘  └──────────┘
```

## 🔄 Data Flow

### Prediction Pipeline
```
User Input → Form Submission
             ↓
    app.py (test route)
             ↓
convert_to_model_input()
             ↓
predict() → preprocess_single()
             ↓
model.predict() → Risk Level + Confidence
             ↓
Store in MongoDB → Redirect to Report
```

### Input Mapping
```python
Form Fields → Model Input:
- Age, Gender → from user profile
- Weight, Height, BMI → from form
- BloodSugar, HeartRate → from form
- Smoker, ActivityLevel → from form
- ECG (text) → parsed to ECG_RA, ECG_LA, ECG_RL
```

## 🧩 File Dependencies

### app.py Imports
```python
from utils.otp import generate_otp           # OTP generation
from utils.pdf import generate_report_pdf   # PDF creation
from utils.risk import compute_risk_level   # Fallback calculator
from prediction.Predict import predict      # ML predictions
```

### Critical Functions

#### 1. `convert_to_model_input(responses, user)`
- Maps form data to model format
- Handles ECG text parsing
- Calculates BMI if missing
- Returns dict compatible with model

#### 2. `predict(model_input)`
- Loads pre-trained model
- Applies preprocessing (scaler + encoders)
- Returns: (risk_level, confidence, probabilities)

#### 3. `compute_risk_level(responses)`
- Fallback risk calculator
- Rule-based scoring system
- No ML required

## 🗄️ Database Schema

### Users Collection
```python
{
  "userId": str,
  "name": str,
  "phone": str,
  "age": int,
  "gender": str,
  "role": "user" | "admin"
}
```

### Tests Collection
```python
{
  "userId": str,
  "testId": str,
  "responses": dict,              # Original form data
  "riskResult": str,              # ML prediction
  "simpleRiskResult": str,         # Fallback result
  "mlConfidence": float,          # 0.0-1.0
  "mlProbabilities": {             # Risk distribution
    "Low": float,
    "Medium": float,
    "High": float
  },
  "timestamp": datetime,
  "report": {
    "userName": str,
    "userId": str,
    "generatedAt": str
  }
}
```

## 🔐 Security Implementation

### Session Management
```python
session["userId"] = user_id
session["name"] = user_name
session["role"] = "user" | "admin"
```

### Access Control
- Route protection via `require_login()`
- Admin-only routes via `is_admin()`
- User isolation in dashboards

### OTP System
- 6-digit numeric code
- Session-based storage
- Console logging for development

## 📊 ML Model Details

### Model Load Process
```python
# In prediction/Predict.py
model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(encoders_path)
```

### Preprocessing Steps
1. **Numeric Features**: StandardScaler transformation
2. **Categorical Features**: LabelEncoder transformation
3. **Handling Unseen Values**: Map to first known class

### Output Format
```python
# 3-class classification
prediction = ("Low" | "Medium" | "High", confidence, [low_prob, medium_prob, high_prob])

# Binary classification fallback
prediction = ("Low" | "High", probability, [prob_not_risk, prob_risk])
```

## 🎨 Frontend Components

### Bootstrap Integration
- Navbar: User navigation
- Cards: Content containers
- Forms: Input validation
- Badges: Risk level display
- Progress: Questionnaire flow

### JavaScript Features
- Question navigation (Next/Previous)
- Form validation
- Progress tracking
- Chart.js gauge visualization
- Confetti animation

### Styling
- CSS gradients (pink/indigo)
- Hover effects
- Responsive layouts
- Blob decorations

## 🔧 Configuration Files

### requirements.txt
```
Flask==3.0.3
pymongo==4.8.0
python-dotenv==1.0.1
reportlab==4.2.5
numpy==1.26.4
pandas==2.2.1
tensorflow==2.16.1
scikit-learn==1.5.0
joblib==1.4.0
openpyxl==3.1.5
```

### .env (Create this file)
```env
SECRET_KEY=dev_secret_key_change_me
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=cardioclue
PORT=5000
```

## 🚨 Error Handling

### ML Model Fallback
```python
try:
    ml_prediction = predict(model_input)
except Exception as e:
    app.logger.error(f"ML prediction failed: {e}")
    # Falls back to simple_risk_calculation
```

### Import Handling
```python
try:
    from prediction.Predict import predict
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"ML Model not available: {e}")
```

## 📝 API Endpoints

### GET `/`
- Login page
- Admins can login here

### POST `/` → `/verify`
- Initiates OTP verification
- Creates session variables

### GET/POST `/verify`
- OTP verification page
- Creates or updates user

### GET `/onboarding`
- New user information collection
- Age and gender setup

### GET `/user/<userId>`
- User dashboard
- Lists all user reports
- Profile management

### GET/POST `/test/<userId>`
- Health questionnaire
- **ML prediction happens here**
- Stores test in database

### GET `/report/<testId>`
- Risk report display
- Shows ML predictions
- Download/share options

## 🧪 Testing

### Check Model Connection
```bash
python check_model.py
```

Expected output:
```
✅ ML MODEL IS CONNECTED TO THE APP!
✅ TensorFlow is working
✅ Model files loaded successfully
```

### Verify TensorFlow
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Expected: `2.16.1` or similar

### Test Prediction
```bash
python prediction/Predict.py
```

Follow prompts to test with sample data.

## 🔍 Debugging

### Enable Debug Mode
Already enabled in `app.py`:
```python
app.run(debug=True)
```

### View Logs
- Flask logs appear in terminal
- TensorFlow warnings visible
- ML errors logged with traceback

### Common Issues

#### 1. "ML Model not available"
- Check: `python check_model.py`
- Solution: Install TensorFlow correctly
- Verify: Long path support enabled

#### 2. "MongoDB connection failed"
- Check: MongoDB service running
- Solution: Start MongoDB or update URI

#### 3. "Version warnings"
- Meaning: Compatible but different versions
- Solution: Update scikit-learn or ignore
- Impact: None on functionality

## 📚 Code Organization

### Backend Structure
```
app.py
├── create_app() - Factory function
│   ├── Helper functions (require_login, is_admin, etc.)
│   ├── convert_to_model_input() - Data transformation
│   └── Routes (/, /verify, /test, /report, etc.)
```

### ML Module Structure
```
prediction/Predict.py
├── Global model loading
├── preprocess_single() - Data preprocessing
├── predict() - Model prediction
└── LabelEncoder mapping
```

### Utils Structure
```
utils/
├── otp.py - generate_otp() function
├── pdf.py - PDF report generation
└── risk.py - compute_risk_level() function
```

## 🌐 Deployment Considerations

### Production Checklist
- [ ] Set `SECRET_KEY` in environment
- [ ] Use production WSGI server (gunicorn)
- [ ] Enable HTTPS
- [ ] Configure MongoDB properly
- [ ] Set up monitoring/logging
- [ ] Disable debug mode
- [ ] Optimize model loading (cache globally)

### Performance Notes
- Model loads on app startup
- Takes ~5-10 seconds first time
- Subsequent predictions: <1 second
- Use production-grade server for traffic

## 📞 Maintenance

### Updating Model
1. Train new model in `model_building/Model.py`
2. Save to `models/hybrid_model2.h5`
3. Update path in `prediction/Predict.py`
4. Retrain preprocessors if schema changes

### Database Backups
```bash
mongodump --out=/backup/cardioclue
mongorestore /backup/cardioclue
```

## 🔗 External Dependencies

### Python Packages
- Flask: Web server
- pymongo: MongoDB driver
- tensorflow: Machine learning
- scikit-learn: Preprocessing
- pandas, numpy: Data handling
- reportlab: PDF generation

### External Services
- MongoDB: Database
- (Optional) SMS/WhatsApp APIs for production

---

**Last Updated**: October 2025
**Version**: 1.0
**Status**: Production Ready

