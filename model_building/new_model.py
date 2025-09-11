
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

        # --- Handle class imbalance ---
        ros = RandomOverSampler(random_state=42)
        X_combined = np.hstack([X_train_num] + [X_train_cat[:, i].reshape(-1, 1)
                                               for i in range(X_train_cat.shape[1])])
        X_resampled, y_resampled = ros.fit_resample(X_combined, y_train)

        # Split numeric & categorical again after oversampling
        X_train_num = X_resampled[:, :len(numeric_cols)]
        X_train_cat = X_resampled[:, len(numeric_cols):].astype(np.int32)

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
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        if num_classes == 2:
            output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            loss = "binary_crossentropy"
        else:
            output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
            loss = "sparse_categorical_crossentropy"

        model = tf.keras.Model(inputs=[numeric_input] + cat_inputs, outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=loss, metrics=["accuracy"])
        model.summary()
        return model

    # -------------------------
    # Evaluate
    # -------------------------
    def evaluate(self, model, history, test_inputs, y_test):
        print("\n[INFO] Running Evaluation...")
        try:
            y_pred_raw = model.predict(test_inputs)

            if y_pred_raw.ndim == 2 and y_pred_raw.shape[1] > 1:
                y_pred_prob = y_pred_raw
                y_pred = np.argmax(y_pred_prob, axis=1)
                average = 'macro'
            else:
                y_pred_prob = y_pred_raw.ravel()
                y_pred = (y_pred_prob > 0.5).astype(int)
                average = 'binary'

            acc = accuracy_score(y_test, y_pred)

            if average == 'binary':
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                # Handle single-class edge case safely for ROC-AUC
                unique_classes = np.unique(y_test)
                if unique_classes.size < 2:
                    roc = float('nan')
                else:
                    roc = roc_auc_score(y_test, y_pred_prob)
            else:
                prec = precision_score(y_test, y_pred, zero_division=0, average=average)
                rec = recall_score(y_test, y_pred, zero_division=0, average=average)
                f1 = f1_score(y_test, y_pred, zero_division=0, average=average)
                # Align y_score columns with classes present in y_test to avoid mismatch errors
                classes_in_test = np.unique(y_test)
                score_matrix = y_pred_prob.astype(np.float64)
                # If model outputs more classes than present in y_test, subset and remap
                if score_matrix.shape[1] != classes_in_test.size:
                    # Subset score columns to only classes present in y_test
                    score_matrix = score_matrix[:, classes_in_test]
                    # Renormalize per row so probabilities sum to 1
                    row_sums = score_matrix.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0.0] = 1.0
                    score_matrix = score_matrix / row_sums
                    # Remap y_test to 0..k-1 based on present classes
                    label_to_index = {label: idx for idx, label in enumerate(classes_in_test)}
                    y_test_mapped = np.vectorize(label_to_index.get)(y_test)
                    roc = roc_auc_score(y_test_mapped, score_matrix, multi_class='ovr', average='macro')
                else:
                    # Ensure rows sum to 1.0 (numerical safety)
                    row_sums = score_matrix.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0.0] = 1.0
                    score_matrix = score_matrix / row_sums
                    roc = roc_auc_score(y_test, score_matrix, multi_class='ovr', average='macro')

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

            if average == 'binary':
                unique_classes = np.unique(y_test)
                if unique_classes.size < 2:
                    return
                fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                plt.figure(figsize=(6, 6))
                plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc:.2f})")
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.legend()
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

        train_inputs = [X_train_num] + [X_train_cat[:, i].reshape(-1, 1) for i in range(X_train_cat.shape[1])]
        val_inputs = [X_val_num] + [X_val_cat[:, i].reshape(-1, 1) for i in range(X_val_cat.shape[1])]
        test_inputs = [X_test_num] + [X_test_cat[:, i].reshape(-1, 1) for i in range(X_test_cat.shape[1])]

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(train_inputs, y_train,
                            validation_data=(val_inputs, y_val),
                            epochs=100, batch_size=32, verbose=2,
                            callbacks=[])

        self.evaluate(model, history, test_inputs, y_test)

        # os.makedirs("models", exist_ok=True)
        # os.makedirs("artifacts", exist_ok=True)
        # model.save("models/hybrid_model.h5")
        # joblib.dump(self.scaler, "artifacts/scaler.pkl")
        # joblib.dump(self.label_encoders, "artifacts/label_encoders.pkl")
        # joblib.dump(self.target_encoder, "artifacts/target_encoder.pkl")
        # print("[INFO] Model and preprocessors saved successfully.")


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
