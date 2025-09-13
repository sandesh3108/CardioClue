
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
import joblib
import os
import sys


class HybridPipeline:
    def __init__(self, data_path, target_col="EarlyCVD_Risk", test_size=0.15, val_size=0.15):
        self.data_path = data_path
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size

        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.cat_input_lens = []
        self.target_encoder = None

        self.df = self._load_data()
        print(f"[INFO] Dataset loaded: {self.data_path}, shape={self.df.shape}")

    # -------------------------
    # Load Data
    # -------------------------
    def _load_data(self):
        if self.data_path.endswith(".csv"):
            return pd.read_csv(self.data_path)
        elif self.data_path.endswith(".xlsx") or self.data_path.endswith(".xls"):
            return pd.read_excel(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")

    # -------------------------
    # Preprocess
    # -------------------------
    def preprocess(self):
        df = self.df.copy()

        numeric_cols = ["Age", "SleepHours", "Weight", "Height", "BMI",
                        "BloodSugar", "HeartRate", "ECG_RA", "ECG_LA", "ECG_RL", "StressLevel"]
        categorical_cols = ["Gender", "Smoker", "ActivityLevel", "Diet", "Alcohol",
                            "FamilyHistory", "HighBP", "Diabetes", "HeartDisease"]
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        # Fill missing target values with mode
        if df[self.target_col].isna().any():
            df[self.target_col] = df[self.target_col].fillna(df[self.target_col].mode()[0])

        # Create risk categories instead of continuous values
        def categorize_risk(risk_value):
            if risk_value <= 50:
                return "Low"
            elif risk_value <= 100:
                return "Medium"
            else:
                return "High"

        # Apply risk categorization
        df[self.target_col] = df[self.target_col].apply(categorize_risk)
        print(f"[INFO] Risk categories: {df[self.target_col].value_counts()}")

        # Handle stratify issue
        class_counts = Counter(df[self.target_col])
        stratify = df[self.target_col] if min(class_counts.values()) >= 2 else None

        # Split data
        train_df, test_df = train_test_split(df, test_size=self.test_size,
                                             stratify=stratify, random_state=42)
        stratify_train = train_df[self.target_col] if stratify is not None else None
        train_df, val_df = train_test_split(train_df, test_size=self.val_size,
                                            stratify=stratify_train, random_state=42)

        # --- Scale numeric ---
        X_train_num = self.scaler.fit_transform(train_df[numeric_cols].fillna(0))
        X_val_num = self.scaler.transform(val_df[numeric_cols].fillna(0))
        X_test_num = self.scaler.transform(test_df[numeric_cols].fillna(0))

        # Add small amount of noise to training data to prevent overfitting
        noise_factor = 0.01
        X_train_num = X_train_num + np.random.normal(0, noise_factor, X_train_num.shape)

        # --- Encode categorical ---
        X_train_cat, X_val_cat, X_test_cat = [], [], []
        self.cat_input_lens = []

        for col in categorical_cols:
            le = LabelEncoder()
            combined_vals = pd.concat([
                train_df[col].fillna('NA').astype(str),
                val_df[col].fillna('NA').astype(str),
                test_df[col].fillna('NA').astype(str)
            ])
            le.fit(combined_vals)
            self.label_encoders[col] = le
            X_train_cat.append(le.transform(train_df[col].fillna('NA').astype(str)))
            X_val_cat.append(le.transform(val_df[col].fillna('NA').astype(str)))
            X_test_cat.append(le.transform(test_df[col].fillna('NA').astype(str)))
            self.cat_input_lens.append(len(le.classes_))

        X_train_cat = np.stack(X_train_cat, axis=1).astype(np.int32)
        X_val_cat = np.stack(X_val_cat, axis=1).astype(np.int32)
        X_test_cat = np.stack(X_test_cat, axis=1).astype(np.int32)

        # Encode target
        self.target_encoder = LabelEncoder()
        self.target_encoder.fit(pd.concat([
            train_df[self.target_col],
            val_df[self.target_col],
            test_df[self.target_col]
        ]).astype(str))
        y_train = self.target_encoder.transform(train_df[self.target_col].astype(str)).astype(np.int32)
        y_val = self.target_encoder.transform(val_df[self.target_col].astype(str)).astype(np.int32)
        y_test = self.target_encoder.transform(test_df[self.target_col].astype(str)).astype(np.int32)

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

        return X_train_num, X_val_num, X_test_num, X_train_cat, X_val_cat, X_test_cat, y_resampled, y_val, y_test

    # -------------------------
    # Build Model
    # -------------------------
    def build_model(self, num_numeric, cat_input_lens, num_classes=2):
        numeric_input = tf.keras.layers.Input(shape=(num_numeric,), name="numeric_input")
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

        # Since we now have 3 risk categories, use multiclass classification
        output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"

        model = tf.keras.Model(inputs=[numeric_input] + cat_inputs, outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                      loss=loss, metrics=["accuracy"])
        model.summary()
        return model

    # -------------------------
    # Evaluate
    # -------------------------
    def evaluate(self, model, history, test_inputs, y_test, num_classes=3):
        print("\n[INFO] Running Evaluation...")
        try:
            y_pred_raw = model.predict(test_inputs)

            # For 3-class classification, always use multiclass approach
            y_pred_prob = y_pred_raw
            y_pred = np.argmax(y_pred_prob, axis=1)
            average = 'macro'

            acc = accuracy_score(y_test, y_pred)

            # Calculate metrics for 3-class classification
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
            if history:
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
                colors = ['blue', 'red', 'green']
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

        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            print("Accuracy: N/A\nPrecision: N/A\nRecall: N/A\nF1-Score: N/A\nROC-AUC: N/A\nConfusion Matrix:\n N/A")

    # -------------------------
    # Run Pipeline
    # -------------------------
    def run(self):
        X_train_num, X_val_num, X_test_num, X_train_cat, X_val_cat, X_test_cat, y_train, y_val, y_test = self.preprocess()
        num_classes = len(self.target_encoder.classes_) if self.target_encoder is not None else 2
        model = self.build_model(num_numeric=X_train_num.shape[1], cat_input_lens=self.cat_input_lens, num_classes=num_classes)

        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))
        print(f"[INFO] Class weights: {class_weight_dict}")

        train_inputs = [X_train_num] + [X_train_cat[:, i].reshape(-1, 1) for i in range(X_train_cat.shape[1])]
        val_inputs = [X_val_num] + [X_val_cat[:, i].reshape(-1, 1) for i in range(X_val_cat.shape[1])]
        test_inputs = [X_test_num] + [X_test_cat[:, i].reshape(-1, 1) for i in range(X_test_cat.shape[1])]

        # Add early stopping and learning rate reduction
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.3, patience=3, min_lr=1e-8
        )

        history = model.fit(train_inputs, y_train,
                            validation_data=(val_inputs, y_val),
                            epochs=30, batch_size=16, verbose=2,
                            class_weight=class_weight_dict,
                            callbacks=[early_stopping, reduce_lr])

        self.evaluate(model, history, test_inputs, y_test, num_classes)

        os.makedirs("models", exist_ok=True)
        os.makedirs("artifacts", exist_ok=True)
        model.save("models/hybrid_model.h5")
        joblib.dump(self.scaler, "artifacts/scaler.pkl")
        joblib.dump(self.label_encoders, "artifacts/label_encoders.pkl")
        joblib.dump(self.target_encoder, "artifacts/target_encoder.pkl")
        print("[INFO] Model and preprocessors saved successfully.")


# ------------------------------
# Run pipeline
# ------------------------------
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

    pipeline = HybridPipeline(
        data_path="dataset/health_dataset_10k_with_risk_sample_with_ECG.xlsx",
        target_col="EarlyCVD_Risk"
    )
    pipeline.run()
