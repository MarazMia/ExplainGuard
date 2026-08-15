import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)

# ============================================================
# CONFIGURATION & ADAPTIVE PATHS
# ============================================================
DATASET_NAME = "compas"

# SCRIPT_DIR is ~/Desktop/ExplainGuard/model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# PROJECT_ROOT goes 1 level up to ~/Desktop/ExplainGuard
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DEV_DIR = os.path.join(PROJECT_ROOT, "Data", "Dev_End", DATASET_NAME)
MODEL_DIR = SCRIPT_DIR  # Model directory is where this script lives

print(f"Project Root Resolved : {PROJECT_ROOT}")
print(f"Loading Dev Data From : {DATA_DEV_DIR}")

# 1. Load Preprocessed Data from Dev_End
X_train = pd.read_csv(os.path.join(DATA_DEV_DIR, f"{DATASET_NAME}_X_train_full.csv"))
y_train = pd.read_csv(os.path.join(DATA_DEV_DIR, f"{DATASET_NAME}_y_train_full.csv")).squeeze()

X_test = pd.read_csv(os.path.join(DATA_DEV_DIR, f"{DATASET_NAME}_X_test_full.csv"))
y_test = pd.read_csv(os.path.join(DATA_DEV_DIR, f"{DATASET_NAME}_y_test_full.csv")).squeeze()

# ============================================================
# 2. TRAIN XGBOOST MODEL
# ============================================================
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

print("\nTraining XGBoost Classifier...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ============================================================
# 3. GLOBAL MODEL EVALUATION
# ============================================================
y_preds_prob = model.predict_proba(X_test)[:, 1]
y_preds = (y_preds_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_preds_prob)
acc = accuracy_score(y_test, y_preds)
prec = precision_score(y_test, y_preds)
rec = recall_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds)

print("\n" + "="*50)
print(f" MODEL PERFORMANCE METRICS [{DATASET_NAME.upper()}]")
print("="*50)
print(f" ROC-AUC   : {auc:.4f}")
print(f" Accuracy  : {acc:.4f}")
print(f" Precision : {prec:.4f}")
print(f" Recall    : {rec:.4f}")
print(f" F1-Score  : {f1:.4f}")
print("-" * 50)
print("\nClassification Report:\n", classification_report(y_test, y_preds))

# ============================================================
# 4. SAVE MODEL ARTIFACT
# ============================================================
model_save_path = os.path.join(MODEL_DIR, f"original_{DATASET_NAME}_xgboost_model.json")
model.save_model(model_save_path)

print(f"\nModel Artifact Successfully Saved: {model_save_path}")