import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_auc_score, classification_report, r2_score, roc_curve)
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

        # Save preprocessors
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.cat_input_lens = []
        self.target_encoder = None

        # Load dataset
        self.df = self._load_data()
        print(f"[INFO] Dataset loaded: {self.data_path}, shape={self.df.shape}")

    def _load_data(self):
        """Auto-detect CSV or Excel"""
        if self.data_path.endswith(".csv"):
            df = pd.read_csv(self.data_path)
        elif self.data_path.endswith(".xlsx") or self.data_path.endswith(".xls"):
            df = pd.read_excel(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        return df

    def preprocess(self):
        df = self.df.copy()

        # Define columns
        numeric_cols = ["Age","SleepHours","Weight","Height","BMI",
                        "BloodSugar","HeartRate","ECG_RA","ECG_LA","ECG_RL","StressLevel"]
        categorical_cols = ["Gender","Smoker","ActivityLevel","Diet","Alcohol",
                            "FamilyHistory","HighBP","Diabetes","HeartDisease"]

        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        # Clean/prepare target column: fill missing with mode
        if df[self.target_col].isna().any():
            mode_value = df[self.target_col].mode().iloc[0]
            df[self.target_col] = df[self.target_col].fillna(mode_value)

        # --- Check class distribution ---
        class_counts = Counter(df[self.target_col])
        min_class_size = min(class_counts.values())

        if min_class_size < 2:
            print("[WARNING] Some classes have <2 samples. Using random split (no stratify).")
            stratify = None
        else:
            stratify = df[self.target_col]

        # Train/test split
        train_df, test_df = train_test_split(df, test_size=self.test_size,
                                             stratify=stratify, random_state=42)

        if stratify is not None:
            stratify_train = train_df[self.target_col]
        else:
            stratify_train = None

        # Train/val split
        train_df, val_df = train_test_split(train_df, test_size=self.val_size,
                                            stratify=stratify_train, random_state=42)

        self.train_df, self.val_df, self.test_df = train_df, val_df, test_df

        # --- Scale numeric ---
        X_train_num = self.scaler.fit_transform(train_df[numeric_cols].fillna(0))
        X_val_num = self.scaler.transform(val_df[numeric_cols].fillna(0))
        X_test_num = self.scaler.transform(test_df[numeric_cols].fillna(0))

        # --- Encode categorical ---
        X_train_cat, X_val_cat, X_test_cat = [], [], []
        self.cat_input_lens = []
        for col in categorical_cols:
            le = LabelEncoder()
            train_vals = train_df[col].fillna('NA').astype(str)

            le.fit(pd.concat([train_vals,
                              val_df[col].fillna('NA').astype(str),
                              test_df[col].fillna('NA').astype(str)]))

            self.label_encoders[col] = le
            X_train_cat.append(le.transform(train_df[col].fillna('NA').astype(str)))
            X_val_cat.append(le.transform(val_df[col].fillna('NA').astype(str)))
            X_test_cat.append(le.transform(test_df[col].fillna('NA').astype(str)))
            self.cat_input_lens.append(len(le.classes_))

        X_train_cat = np.stack(X_train_cat, axis=1)
        X_val_cat = np.stack(X_val_cat, axis=1)
        X_test_cat = np.stack(X_test_cat, axis=1)

        # Ensure integer dtype for embedding inputs
        X_train_cat = X_train_cat.astype(np.int32)
        X_val_cat = X_val_cat.astype(np.int32)
        X_test_cat = X_test_cat.astype(np.int32)

        # --- Encode target as 0/1 integers consistently ---
        self.target_encoder = LabelEncoder()
        self.target_encoder.fit(pd.concat([
            train_df[self.target_col].astype(str),
            val_df[self.target_col].astype(str),
            test_df[self.target_col].astype(str)
        ], axis=0))

        y_train = self.target_encoder.transform(train_df[self.target_col].astype(str)).astype(np.int32)
        y_val = self.target_encoder.transform(val_df[self.target_col].astype(str)).astype(np.int32)
        y_test = self.target_encoder.transform(test_df[self.target_col].astype(str)).astype(np.int32)

        return (X_train_num, X_val_num, X_test_num,
                X_train_cat, X_val_cat, X_test_cat,
                y_train, y_val, y_test)

    def build_model(self, num_numeric, cat_input_lens, num_classes=2, embedding_dim=4):
        """Build hybrid model with numeric + categorical inputs. Supports binary and multiclass."""

        # Numeric input
        numeric_input = tf.keras.layers.Input(shape=(num_numeric,), name="numeric_input")

        # Categorical inputs
        cat_inputs = []
        cat_embeds = []
        for i, vocab_size in enumerate(cat_input_lens):
            inp = tf.keras.layers.Input(shape=(1,), name=f"cat_input_{i}")
            emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inp)
            emb = tf.keras.layers.Flatten()(emb)
            cat_inputs.append(inp)
            cat_embeds.append(emb)

        # Combine
        x = tf.keras.layers.Concatenate()([numeric_input] + cat_embeds)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        if num_classes == 2:
            output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            loss = "binary_crossentropy"
        else:
            output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
            loss = "sparse_categorical_crossentropy"

        model = tf.keras.Model(inputs=[numeric_input] + cat_inputs, outputs=output)
        model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        model.summary()
        return model

    def evaluate(self, model, history, test_inputs, y_test):
        """Evaluation Section with Metrics and Graphs"""
        print("\n[INFO] Running Evaluation...")

        # Predictions
        y_pred_raw = model.predict(test_inputs)

        # Determine binary vs multiclass from prediction shape
        if y_pred_raw.ndim == 2 and y_pred_raw.shape[1] > 1:
            # multiclass
            y_pred_prob = y_pred_raw
            y_pred = np.argmax(y_pred_prob, axis=1)
            average = 'macro'
        else:
            # binary
            y_pred_prob = y_pred_raw.ravel()
            y_pred = (y_pred_prob > 0.5).astype(int)
            average = 'binary'

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0, average=(None if average=='binary' else average))
        rec = recall_score(y_test, y_pred, zero_division=0, average=(None if average=='binary' else average))
        f1 = f1_score(y_test, y_pred, zero_division=0, average=(None if average=='binary' else average))
        if average == 'binary':
            roc = roc_auc_score(y_test, y_pred_prob)
        else:
            # Align y_score columns to labels present in y_test to avoid mismatch
            present_labels = np.unique(y_test)
            label_to_index = {label: idx for idx, label in enumerate(present_labels)}
            y_test_mapped = np.vectorize(label_to_index.get)(y_test)
            # Select only columns for present labels and renormalize rows to sum to 1
            y_score_present = y_pred_prob[:, present_labels]
            row_sums = y_score_present.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            y_score_present = y_score_present / row_sums
            roc = roc_auc_score(y_test_mapped, y_score_present, multi_class='ovr', average='macro')
        cm = confusion_matrix(y_test, y_pred)

        # Print results
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc:.4f}")
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Optional R2 Score (not standard for classification)
        # R2 on probabilities is not meaningful; keep for reference
        try:
            r2 = r2_score(y_test, y_pred_prob if average=='binary' else np.max(y_pred_prob, axis=1))
        except Exception:
            r2 = float('nan')
        print(f"R² Score (for reference only): {r2:.4f}")

        # ----------- Plot Training Curves -----------
        plt.figure(figsize=(12, 5))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label="Train Accuracy")
        plt.plot(history.history['val_accuracy'], label="Val Accuracy")
        plt.title("Training & Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label="Train Loss")
        plt.plot(history.history['val_loss'], label="Val Loss")
        plt.title("Training & Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.show()

        # ----------- ROC Curve -----------
        if average == 'binary':
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc:.2f})")
            plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.show()

    def run(self):
        # Preprocess
        (X_train_num, X_val_num, X_test_num,
         X_train_cat, X_val_cat, X_test_cat,
         y_train, y_val, y_test) = self.preprocess()

        # Build model
        num_classes = len(self.target_encoder.classes_) if self.target_encoder is not None else 2
        model = self.build_model(num_numeric=X_train_num.shape[1],
                                 cat_input_lens=self.cat_input_lens,
                                 num_classes=num_classes)

        # Prepare inputs
        # Reshape categorical features to (batch, 1) for Keras Inputs
        train_inputs = [X_train_num] + [X_train_cat[:, i].reshape(-1, 1) for i in range(X_train_cat.shape[1])]
        val_inputs = [X_val_num] + [X_val_cat[:, i].reshape(-1, 1) for i in range(X_val_cat.shape[1])]
        test_inputs = [X_test_num] + [X_test_cat[:, i].reshape(-1, 1) for i in range(X_test_cat.shape[1])]

        # Train
        history = model.fit(train_inputs, y_train,
                            validation_data=(val_inputs, y_val),
                            epochs=10, batch_size=32,
                            verbose=2)

        # Evaluate
        self.evaluate(model, history, test_inputs, y_test)

        # Save model & preprocessors into structured directories
        models_dir = os.path.join("models")
        artifacts_dir = os.path.join("artifacts")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)

        model_path = os.path.join(models_dir, "hybrid_model1.h5")
        scaler_path = os.path.join(artifacts_dir, "scaler1.pkl")
        encoders_path = os.path.join(artifacts_dir, "label_encoders1.pkl")
        target_encoder_path = os.path.join(artifacts_dir, "target_encoder1.pkl")

        model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoders, encoders_path)
        joblib.dump(self.target_encoder, target_encoder_path)
        print(f"[INFO] Model saved to: {model_path}")
        print(f"[INFO] Scaler saved to: {scaler_path}")
        print(f"[INFO] Label encoders saved to: {encoders_path}")
        print(f"[INFO] Target encoder saved to: {target_encoder_path}")


# -----------------------------
# Run Pipeline
# -----------------------------
if __name__ == "__main__":
    # Ensure UTF-8 console to avoid UnicodeEncodeError on Windows terminals
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
    pipeline = HybridPipeline(
        data_path="datsset\health_dataset_10k_with_risk_sample_with_ECG.xlsx",  # <-- change path here
        target_col="EarlyCVD_Risk"
    )
    pipeline.run()
