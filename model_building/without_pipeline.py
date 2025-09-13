import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, classification_report, roc_curve)
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import matplotlib.pyplot as plt
import sys

# -------------------------
# Load Data
# -------------------------
data_path = "dataset/health_dataset_10k_with_risk_sample_with_ECG.xlsx"
target_col = "EarlyCVD_Risk"

if data_path.endswith(".csv"):
    df = pd.read_csv(data_path)
elif data_path.endswith(".xlsx") or data_path.endswith(".xls"):
    df = pd.read_excel(data_path)
else:
    raise ValueError("Unsupported file format. Use CSV or Excel.")

print(f"[INFO] Dataset loaded: {data_path}, shape={df.shape}")

# -------------------------
# Preprocess
# -------------------------
numeric_cols = ["Age", "SleepHours", "Weight", "Height", "BMI",
                "BloodSugar", "HeartRate", "ECG_RA", "ECG_LA", "ECG_RL", "StressLevel"]

categorical_cols = ["Gender", "Smoker", "ActivityLevel", "Diet", "Alcohol",
                    "FamilyHistory", "HighBP", "Diabetes", "HeartDisease"]

# Fill missing target values with mode
if df[target_col].isna().any():
    df[target_col] = df[target_col].fillna(df[target_col].mode()[0])

# Create risk categories instead of continuous values
def categorize_risk(risk_value):
    if risk_value <= 50:
        return "Low"
    elif risk_value <= 100:
        return "Medium"
    else:
        return "High"

# Apply risk categorization
df[target_col] = df[target_col].apply(categorize_risk)
print(f"[INFO] Risk categories: {df[target_col].value_counts()}")

# Handle stratify issue
class_counts = Counter(df[target_col])
stratify = df[target_col] if min(class_counts.values()) >= 2 else None

# Split data
train_df, test_df = train_test_split(df, test_size=0.15,
                                     stratify=stratify, random_state=42)
stratify_train = train_df[target_col] if stratify is not None else None
train_df, val_df = train_test_split(train_df, test_size=0.15,
                                    stratify=stratify_train, random_state=42)

# --- Scale numeric ---
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train_df[numeric_cols].fillna(0))
X_val_num = scaler.transform(val_df[numeric_cols].fillna(0))
X_test_num = scaler.transform(test_df[numeric_cols].fillna(0))

# Add small amount of noise to training data to prevent overfitting
noise_factor = 0.01
X_train_num = X_train_num + np.random.normal(0, noise_factor, X_train_num.shape)

# --- Encode categorical ---
label_encoders = {}
cat_input_lens = []

X_train_cat, X_val_cat, X_test_cat = [], [], []

for col in categorical_cols:
    le = LabelEncoder()
    combined_vals = pd.concat([
        train_df[col].fillna('NA').astype(str),
        val_df[col].fillna('NA').astype(str),
        test_df[col].fillna('NA').astype(str)
    ])
    le.fit(combined_vals)
    label_encoders[col] = le
    X_train_cat.append(le.transform(train_df[col].fillna('NA').astype(str)))
    X_val_cat.append(le.transform(val_df[col].fillna('NA').astype(str)))
    X_test_cat.append(le.transform(test_df[col].fillna('NA').astype(str)))
    cat_input_lens.append(len(le.classes_))

X_train_cat = np.stack(X_train_cat, axis=1).astype(np.int32)
X_val_cat = np.stack(X_val_cat, axis=1).astype(np.int32)
X_test_cat = np.stack(X_test_cat, axis=1).astype(np.int32)

# Encode target
target_encoder = LabelEncoder()
target_encoder.fit(pd.concat([
    train_df[target_col],
    val_df[target_col],
    test_df[target_col]
]).astype(str))
y_train = target_encoder.transform(train_df[target_col].astype(str)).astype(np.int32)
y_val = target_encoder.transform(val_df[target_col].astype(str)).astype(np.int32)
y_test = target_encoder.transform(test_df[target_col].astype(str)).astype(np.int32)

# --- Handle class imbalance with SMOTE ---
from imblearn.over_sampling import SMOTE

# Always apply SMOTE to ensure balanced classes
print(f"[INFO] Original class distribution: {Counter(y_train)}")
ros = SMOTE(random_state=42)
X_combined = np.hstack([X_train_num] + [X_train_cat[:, i].reshape(-1, 1)
                                       for i in range(X_train_cat.shape[1])])
X_resampled, y_resampled = ros.fit_resample(X_combined, y_train)

# Split numeric & categorical again after oversampling
X_train_num = X_resampled[:, :len(numeric_cols)]
X_train_cat = X_resampled[:, len(numeric_cols):].astype(np.int32)
print(f"[INFO] After SMOTE class distribution: {Counter(y_resampled)}")

# -------------------------
# Build Model
# -------------------------
num_classes = len(target_encoder.classes_) if target_encoder is not None else 2

numeric_input = tf.keras.layers.Input(shape=(X_train_num.shape[1],), name="numeric_input")
cat_inputs, cat_embeds = [], []

for i, vocab_size in enumerate(cat_input_lens):
    inp = tf.keras.layers.Input(shape=(1,), name=f"cat_input_{i}")
    emb_dim = min(50, (vocab_size + 1) // 2)
    emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim)(inp)
    emb = tf.keras.layers.Flatten()(emb)
    cat_inputs.append(inp)
    cat_embeds.append(emb)

x = tf.keras.layers.Concatenate()([numeric_input] + cat_embeds)
x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.6)(x)

x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)

# Since we now have 4 risk categories, use multiclass classification
output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
loss = "sparse_categorical_crossentropy"

model = tf.keras.Model(inputs=[numeric_input] + cat_inputs, outputs=output)

# Calculate class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
class_weight_dict = dict(enumerate(class_weights))
print(f"[INFO] Class weights: {class_weight_dict}")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=loss, metrics=["accuracy"])
model.summary()

# Add early stopping and learning rate reduction
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.3, patience=3, min_lr=1e-8
)

# -------------------------
# Train Model
# -------------------------
train_inputs = [X_train_num] + [X_train_cat[:, i].reshape(-1, 1) for i in range(X_train_cat.shape[1])]
val_inputs = [X_val_num] + [X_val_cat[:, i].reshape(-1, 1) for i in range(X_val_cat.shape[1])]
test_inputs = [X_test_num] + [X_test_cat[:, i].reshape(-1, 1) for i in range(X_test_cat.shape[1])]

history = model.fit(train_inputs, y_resampled,
                    validation_data=(val_inputs, y_val),
                    epochs=30, batch_size=16, verbose=2,
                    class_weight=class_weight_dict,
                    callbacks=[early_stopping, reduce_lr])

# -------------------------
# Evaluation
# -------------------------
print("\n[INFO] Running Evaluation...")
y_pred_raw = model.predict(test_inputs)

# For 4-class classification, always use multiclass approach
y_pred_prob = y_pred_raw
y_pred = np.argmax(y_pred_prob, axis=1)
average = 'macro'

acc = accuracy_score(y_test, y_pred)

# Calculate metrics for 4-class classification
prec = precision_score(y_test, y_pred, zero_division=0, average=average)
rec = recall_score(y_test, y_pred, zero_division=0, average=average)
f1 = f1_score(y_test, y_pred, zero_division=0, average=average)

# Calculate ROC-AUC for multiclass
try:
    roc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr', average='macro')
except:
    roc = float('nan')

cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss")
plt.legend()
plt.show()

# ROC Curve for multiclass (one-vs-rest)
if len(np.unique(y_test)) > 1:
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'orange']
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Risk Categories')
    plt.legend(loc="lower right")
    plt.show()