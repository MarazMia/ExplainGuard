import os
import sys
from typing import Any, Callable, Dict, List, Tuple
import dill as pickle
import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from collections import defaultdict

# ============================================================
# 1. DYNAMIC PROJECT ROOT & PATH RESOLUTION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Model.blackbox_endpoint import BlackBoxModelEndpoint

# ============================================================
# 2. PATH CONFIGURATION & ENVIRONMENT SETUP
# ============================================================
DATASET_NAME = "compas"

MODEL_DIR = os.path.join(PROJECT_ROOT, "Model")
AUDITOR_DATA_DIR = os.path.join(
    PROJECT_ROOT, "Data", "Auditor_End", DATASET_NAME
)
SUBMISSION_DIR = os.path.join(
    PROJECT_ROOT, "Data", "Auditor_End", "Submissions"
)

os.makedirs(SUBMISSION_DIR, exist_ok=True)


# ============================================================
# 3. SERIALIZABLE MODEL FUNCTION
# ============================================================
class SerializableModelFunction:
    """
    Serializable representation of the auditor's model function (f').

    Instead of serializing heavy XGBoost memory pointers or closures,
    this class stores the model file path and lazily reconstructs the model
    upon unpickling/calling.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_path)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        p1 = self.model.predict_proba(x)[:, 1]
        p0 = 1.0 - p1

        return np.column_stack((p0, p1))


# ============================================================
# 4. SERIALIZABLE SHAP EXPLAINER WRAPPER
# ============================================================
class SerializableKernelExplainer:
    """
    Pickle-safe representation of a SHAP KernelExplainer.

    Bypasses SHAP's unpicklable Cython MemoryViews while preserving:
        1. Model prediction function f'
        2. Background dataset B

    Enables full native reconstruction on the Auditor/Pillar I side.
    """

    def __init__(
        self, model_function: Callable[[np.ndarray], np.ndarray], background_data: np.ndarray
    ) -> None:
        self.model_function = model_function
        self.background_data = np.ascontiguousarray(
            background_data, dtype=np.float64
        )
        self._native_explainer = None

    def reconstruct(self) -> Any:
        """Reconstructs and caches the native SHAP KernelExplainer on demand."""
        if self._native_explainer is None:
            import shap  # Explicit import ensures 'shap' is available when called post-unpickling
            self._native_explainer = shap.KernelExplainer(
                self.model_function, self.background_data
            )
        return self._native_explainer

    @property
    def model(self) -> Any:
        return self.reconstruct().model

    @property
    def data(self) -> Any:
        return self.reconstruct().data


# ============================================================
# 5. MAIN AUDIT PIPELINE
# ============================================================
def run_audit_pipeline() -> None:
    print("====================================================")
    print("        EXPLAINGUARD - AUDIT SUBMISSION PIPELINE    ")
    print("====================================================\n")

    # ----------------------------------------------------
    # Load Original Black-Box Model
    # ----------------------------------------------------
    model_path = os.path.join(
        MODEL_DIR, f"original_{DATASET_NAME}_xgboost_model.json"
    )
    print(f"[+] Loading original XGBoost model from:\n    {model_path}")

    clean_model = xgb.XGBClassifier()
    clean_model.load_model(model_path)
    blackbox_endpoint = BlackBoxModelEndpoint(clean_model)
    print("    --> Original model loaded successfully.\n")

    # ----------------------------------------------------
    # Load Auditor Test Dataset Pool
    # ----------------------------------------------------
    audit_path = os.path.join(
        AUDITOR_DATA_DIR, f"{DATASET_NAME}_X_audit_test.csv"
    )
    print(f"[+] Loading auditor evaluation pool from:\n    {audit_path}")

    X_audit_test = pd.read_csv(audit_path)
    feature_names = X_audit_test.columns.tolist()

    # Data Splits
    N_BG = 50
    # bg_data = X_audit_test.iloc[:N_BG].to_numpy()
    # explain_data = X_audit_test.iloc[N_BG:].to_numpy()
    
    # random sampling
    bg_df = X_audit_test.sample(n=N_BG, random_state=42)
    bg_data = bg_df.to_numpy()

    explain_data = X_audit_test.loc[X_audit_test.index.difference(bg_df.index)].to_numpy()

    print("\n--- DATASET SPLIT SUMMARY ---")
    print(f"  • Total Pool Samples    : {len(X_audit_test)}")
    print(f"  • Background Dataset (B): {len(bg_data)} instances")
    print(f"  • Explanation Targets   : {len(explain_data)} instances")
    print(f"  • Feature Count (d)     : {len(feature_names)}")

    # Initialize serializable prediction endpoint
    model_function = SerializableModelFunction(model_path=model_path)

    def predict_wrapper(x: np.ndarray) -> np.ndarray:
        return model_function(x)

    # ----------------------------------------------------
    # KernelSHAP Audit Execution
    # ----------------------------------------------------
    print("\n----------------------------------------------------")
    print(" [1/2] RUNNING KERNEL SHAP AUDIT")
    print("----------------------------------------------------")

    shap_explainer = shap.KernelExplainer(predict_wrapper, bg_data)
    shap_values = shap_explainer.shap_values(explain_data)

    # Format SHAP values matrix
    if isinstance(shap_values, list):
        shap_matrix = np.asarray(
            shap_values[1] if len(shap_values) > 1 else shap_values[0]
        )
    else:
        shap_matrix = np.asarray(shap_values)

    if shap_matrix.ndim == 3:
        shap_matrix = (
            shap_matrix[:, :, 1]
            if shap_matrix.shape[2] == 2
            else shap_matrix[1, :, :]
        )

    global_shap_importance = np.abs(shap_matrix).mean(axis=0).ravel()

    shap_summary_df = pd.DataFrame(
        {"Feature": feature_names, "Mean |SHAP|": global_shap_importance}
    ).sort_values(by="Mean |SHAP|", ascending=False)

    print("\n--- GLOBAL SHAP FEATURE IMPORTANCE ---")
    print(shap_summary_df.to_string(index=False))

    # Serialize KernelSHAP artifact
    serializable_shap_explainer = SerializableKernelExplainer(
        model_function=model_function, background_data=bg_data
    )

    shap_submission = {
        "explainer_type": "KernelSHAP",
        "explainer_object": serializable_shap_explainer,
        "explained_instances": explain_data,
        "attributions": shap_values,
        "feature_names": feature_names,
    }

    shap_path = os.path.join(
        SUBMISSION_DIR, f"clean_{DATASET_NAME}_shap_submission.pkl"
    )
    print("\n[+] Saving KernelSHAP submission artifact...")
    with open(shap_path, "wb") as f:
        pickle.dump(shap_submission, f)
    print(f"    --> Saved to: {shap_path}")

    # ----------------------------------------------------
    # TabularLIME Audit Execution
    # ----------------------------------------------------
    print("\n----------------------------------------------------")
    print(" [2/2] RUNNING TABULAR LIME AUDIT")
    print("----------------------------------------------------")

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=bg_data,
        feature_names=feature_names,
        class_names=["Class 0", "Class 1"],
        mode="classification",
        discretize_continuous=False,
        random_state=42,
    )

    lime_explainer.predict_fn = predict_wrapper
    lime_explainer.background_data = np.ascontiguousarray(
        bg_data, dtype=np.float64
    )

    lime_attributions_list = []
    lime_intercepts = []  # NEW

    for i in range(len(explain_data)):
        exp = lime_explainer.explain_instance(
            explain_data[i],
            predict_fn=predict_wrapper,
            num_features=len(feature_names),
            num_samples=500,
        )
        lime_attributions_list.append(dict(exp.as_list(label=1)))
        lime_intercepts.append(float(exp.intercept[1]))  # NEW: per-instance intercept


    # ----------------------------------------------------
    # Global LIME Feature Importance (mean |attribution|)
    # ----------------------------------------------------

    lime_abs_sums = defaultdict(float)
    lime_counts = defaultdict(int)

    for attr_dict in lime_attributions_list:
        for feat, val in attr_dict.items():
            lime_abs_sums[feat] += abs(val)
            lime_counts[feat] += 1

    global_lime_importance = {
        feat: lime_abs_sums[feat] / lime_counts[feat]
        for feat in lime_abs_sums
    }

    lime_summary_df = pd.DataFrame(
        {
            "Feature": list(global_lime_importance.keys()),
            "Mean |LIME|": list(global_lime_importance.values()),
        }
    ).sort_values(by="Mean |LIME|", ascending=False)

    print("\n--- GLOBAL LIME FEATURE IMPORTANCE ---")
    print(lime_summary_df.to_string(index=False))


    lime_submission = {
    "explainer_type": "TabularLIME",
    "explainer_object": lime_explainer,
    "explained_instances": explain_data,
    "attributions": lime_attributions_list,
    "intercepts": lime_intercepts,     # NEW
    "feature_names": feature_names,
    }

    lime_path = os.path.join(
        SUBMISSION_DIR, f"clean_{DATASET_NAME}_lime_submission.pkl"
    )
    print("\n[+] Saving TabularLIME submission artifact...")
    with open(lime_path, "wb") as f:
        pickle.dump(lime_submission, f)
    print(f"    --> Saved to: {lime_path}")

    # ----------------------------------------------------
    # Verification & Reconstruction Checks
    # ----------------------------------------------------
    print("\n====================================================")
    print("      VERIFYING SUBMISSION ARTIFACT INTEGRITY      ")
    print("====================================================")

    # Verify SHAP
    with open(shap_path, "rb") as f:
        loaded_shap = pickle.load(f)

    shap_obj: SerializableKernelExplainer = loaded_shap["explainer_object"]
    native_shap = shap_obj.reconstruct()
    extracted_shap_f = native_shap.model.f
    extracted_shap_bg = native_shap.data.data

    print("\n[✓] SHAP Artifact Integrity:")
    print(f"  • Explainer Class  : {type(native_shap).__name__}")
    print(f"  • Extracted f'     : {extracted_shap_f}")
    print(f"  • Extracted B Shape: {extracted_shap_bg.shape}")

    # Test reconstructed f'(x) execution
    test_input = explain_data[:2]
    reconstructed_preds = extracted_shap_f(test_input)
    print(f"  • Test f'(x) Output:\n{reconstructed_preds}")

    # Verify LIME
    with open(lime_path, "rb") as f:
        loaded_lime = pickle.load(f)

    lime_obj = loaded_lime["explainer_object"]
    extracted_lime_f = lime_obj.predict_fn
    extracted_lime_bg = lime_obj.background_data

    print("\n[✓] LIME Artifact Integrity:")
    print(f"  • Explainer Class  : {type(lime_obj).__name__}")
    print(f"  • Extracted f'     : {extracted_lime_f}")
    print(f"  • Extracted B Shape: {extracted_lime_bg.shape}")

    print("\n====================================================")
    print(" AUDIT HARNESS EXECUTION & PICKLING COMPLETED! ")
    print("====================================================\n")


if __name__ == "__main__":
    run_audit_pipeline()