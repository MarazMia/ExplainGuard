import os
import sys
from typing import Any, Callable, Dict, List

import dill as pickle
import lime.lime_tabular
import numpy as np
import pandas as pd
import xgboost as xgb

# ============================================================
# 1. DYNAMIC PROJECT ROOT & PATH RESOLUTION
#    (identical to the clean audit pipeline)
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
ATTACK_NAME = "RandomRank"

MODEL_DIR = os.path.join(PROJECT_ROOT, "Model")
AUDITOR_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Auditor_End", DATASET_NAME)
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "Data", "Auditor_End", "Submissions")

os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ------------------------------------------------------------
# Reproducibility & attack strength
# ------------------------------------------------------------
SEED = 1337
RNG = np.random.default_rng(seed=SEED)

# Scale of the random importance values. Set close to the typical
# |attribution| magnitude of your real explainers so the attack is a
# meaningful test rather than a trivially-obvious outlier.
IMPORTANCE_SCALE = 0.15


# ============================================================
# 3. SERIALIZABLE MODEL FUNCTION
#    (identical to the clean audit pipeline)
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
#    (identical to the clean audit pipeline)
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
# 5. DIVERSE RANDOM ATTRIBUTION GENERATION
# ============================================================
def generate_diverse_attributions(n_instances: int, n_features: int) -> np.ndarray:
    """
    Produces a (N, d) attribution matrix with diverse feature-wise scales 
    so that global feature importances vary widely (avoiding flat uniform means).
    """
    # Generate feature-specific scale multipliers (e.g. exponential or uniform spread)
    feature_scales = RNG.uniform(low=0.01, high=0.30, size=(1, n_features))
    # Base noise matrix
    base_noise = RNG.uniform(low=-1.0, high=1.0, size=(n_instances, n_features))
    # Scale each feature column independently
    attributions = base_noise * feature_scales
    return attributions


# ============================================================
# 6. MAIN ATTACK GENERATION PIPELINE
# ============================================================
def run_random_attack_pipeline() -> None:
    print("====================================================")
    print("    EXPLAINGUARD - RANDOM-RANK ATTACK GENERATION     ")
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
    #    (identical split logic to the clean pipeline, so N/background
    #    line up exactly with the clean submissions for fair comparison)
    # ----------------------------------------------------
    audit_path = os.path.join(
        AUDITOR_DATA_DIR, f"{DATASET_NAME}_X_audit_test.csv"
    )
    print(f"[+] Loading auditor evaluation pool from:\n    {audit_path}")

    X_audit_test = pd.read_csv(audit_path)
    feature_names = X_audit_test.columns.tolist()

    N_BG = 50
    bg_df = X_audit_test.sample(n=N_BG, random_state=1334)
    bg_data = bg_df.to_numpy()
    explain_data = X_audit_test.loc[X_audit_test.index.difference(bg_df.index)].to_numpy()

    N, d = explain_data.shape

    print("\n--- DATASET SPLIT SUMMARY ---")
    print(f"  • Total Pool Samples     : {len(X_audit_test)}")
    print(f"  • Background Dataset (B) : {len(bg_data)} instances")
    print(f"  • Explanation Targets N  : {N} instances")
    print(f"  • Feature Count (d)      : {d}")

    # Initialize serializable prediction endpoint (real model, unchanged)
    model_function = SerializableModelFunction(model_path=model_path)

    # ----------------------------------------------------
    # [1/2] Random SHAP-style attack submission
    # ----------------------------------------------------
    print("\n----------------------------------------------------")
    print(" [1/2] GENERATING RANDOM KERNEL SHAP-STYLE SUBMISSION")
    print("----------------------------------------------------")

    random_shap_matrix = generate_diverse_attributions(N, d)

    print(f"  • Random attribution matrix shape: {random_shap_matrix.shape}")
    print(f"  • Mean |phi| : {np.mean(np.abs(random_shap_matrix)):.4f}")
    print(f"  • Std  |phi| : {np.std(np.abs(random_shap_matrix)):.4f}")

    # Print Global SHAP Feature Importance Summary
    mean_shap_importances = np.abs(random_shap_matrix).mean(axis=0)
    print("\n--- GLOBAL SHAP FEATURE IMPORTANCE ---")
    shap_summary_df = pd.DataFrame(
        {"Feature": feature_names, "Mean |SHAP|": mean_shap_importances}
    ).sort_values(by="Mean |SHAP|", ascending=False)
    print(shap_summary_df.to_string(index=False))

    serializable_shap_explainer = SerializableKernelExplainer(
        model_function=model_function, background_data=bg_data
    )

    # Compute feature rank matrix for SHAP (descending order of absolute attribution values)
    shap_feature_rank = np.argsort(-np.abs(random_shap_matrix), axis=1)

    shap_submission = {
        "explainer_type": "KernelSHAP",
        "explainer_object": serializable_shap_explainer,
        "explained_instances": explain_data,
        "attributions": random_shap_matrix,
        "feature_rank": shap_feature_rank,
        "feature_names": feature_names,
    }

    shap_path = os.path.join(
        SUBMISSION_DIR, f"attack_{ATTACK_NAME}_{DATASET_NAME}_shap_submission.pkl"
    )
    print("\n[+] Saving random-rank SHAP-style submission artifact...")
    with open(shap_path, "wb") as f:
        pickle.dump(shap_submission, f)
    print(f"    --> Saved to: {shap_path}")

    # ----------------------------------------------------
    # [2/2] Random LIME-style attack submission
    # ----------------------------------------------------
    print("\n----------------------------------------------------")
    print(" [2/2] GENERATING RANDOM TABULAR LIME-STYLE SUBMISSION")
    print("----------------------------------------------------")

    random_lime_matrix = generate_diverse_attributions(N, d)

    def predict_wrapper(x: np.ndarray) -> np.ndarray:
        return model_function(x)

    # Build a genuine LimeTabularExplainer object so explainer_object has
    # the exact same type/interface as the clean submission. This step
    # only fits summary statistics on the background data -- it does NOT
    # call explain_instance, so no real LIME sampling/computation happens.
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=bg_data,
        feature_names=feature_names,
        class_names=["Class 0", "Class 1"],
        mode="classification",
        discretize_continuous=False,
        random_state=42,
    )
    lime_explainer.predict_fn = predict_wrapper
    lime_explainer.background_data = np.ascontiguousarray(bg_data, dtype=np.float64)

    # LIME's native attribution format: list of {feature_name: value} dicts,
    # one dict per explained instance (matches exp.as_list(label=1) output
    # shape from the clean pipeline, since discretize_continuous=False keeps
    # keys as plain feature names rather than binned conditions).
    random_lime_attributions: List[Dict[str, float]] = [
        dict(zip(feature_names, random_lime_matrix[i])) for i in range(N)
    ]

    lime_intercepts = []  # NEW
    
    for i in range(len(explain_data)):
            exp = lime_explainer.explain_instance(
                explain_data[i],
                predict_fn=predict_wrapper,
                num_features=len(feature_names),
                num_samples=500,
            )
            lime_intercepts.append(float(exp.intercept[1]))  # NEW: per-instance intercept

    # Compute feature rank matrix for LIME (descending order of absolute attribution values per instance)
    lime_feature_rank = np.argsort(-np.abs(random_lime_matrix), axis=1)

    print(f"  • Random attribution list length : {len(random_lime_attributions)}")
    print(f"  • Mean |phi| : {np.mean(np.abs(random_lime_matrix)):.4f}")
    print(f"  • Std  |phi| : {np.std(np.abs(random_lime_matrix)):.4f}")

    # Print Global LIME Feature Importance Summary
    mean_lime_importances = np.abs(random_lime_matrix).mean(axis=0)
    print("\n--- GLOBAL LIME FEATURE IMPORTANCE ---")
    lime_summary_df = pd.DataFrame(
        {"Feature": feature_names, "Mean |LIME|": mean_lime_importances}
    ).sort_values(by="Mean |LIME|", ascending=False)
    print(lime_summary_df.to_string(index=False))

    lime_submission = {
        "explainer_type": "TabularLIME",
        "explainer_object": lime_explainer,
        "explained_instances": explain_data,
        "attributions": random_lime_attributions,
        "intercepts": lime_intercepts,     # NEW
        "feature_names": feature_names,
        }

    lime_path = os.path.join(
        SUBMISSION_DIR, f"attack_{ATTACK_NAME}_{DATASET_NAME}_lime_submission.pkl"
    )
    print("\n[+] Saving random-rank LIME-style submission artifact...")
    with open(lime_path, "wb") as f:
        pickle.dump(lime_submission, f)
    print(f"    --> Saved to: {lime_path}")

    # ----------------------------------------------------
    # Verification & Reconstruction Checks
    #    (mirrors the integrity checks in the clean pipeline, so you can
    #    confirm the attack bundles load and reconstruct identically)
    # ----------------------------------------------------
    print("\n====================================================")
    print("       VERIFYING ATTACK SUBMISSION ARTIFACT INTEGRITY")
    print("====================================================")

    with open(shap_path, "rb") as f:
        loaded_shap = pickle.load(f)

    shap_obj: SerializableKernelExplainer = loaded_shap["explainer_object"]
    native_shap = shap_obj.reconstruct()
    extracted_shap_f = native_shap.model.f
    extracted_shap_bg = native_shap.data.data

    print("\n[✓] Random SHAP-style Artifact Integrity:")
    print(f"  • Explainer Class  : {type(native_shap).__name__}")
    print(f"  • Extracted f'     : {extracted_shap_f}")
    print(f"  • Extracted B Shape: {extracted_shap_bg.shape}")
    print(f"  • Attributions Shape: {loaded_shap['attributions'].shape}")
    print(f"  • Feature Rank Shape: {loaded_shap['feature_rank'].shape}")

    test_input = explain_data[:2]
    reconstructed_preds = extracted_shap_f(test_input)
    print(f"  • Test f'(x) Output:\n{reconstructed_preds}")

    with open(lime_path, "rb") as f:
        loaded_lime = pickle.load(f)

    lime_obj = loaded_lime["explainer_object"]
    extracted_lime_f = lime_obj.predict_fn
    extracted_lime_bg = lime_obj.background_data

    print("\n[✓] Random LIME-style Artifact Integrity:")
    print(f"  • Explainer Class  : {type(lime_obj).__name__}")
    print(f"  • Extracted f'     : {extracted_lime_f}")
    print(f"  • Extracted B Shape: {extracted_lime_bg.shape}")
    print(f"  • Attributions Length: {len(loaded_lime['attributions'])}")
    print(f"  • Feature Rank Shape: {loaded_lime['feature_rank'].shape}")

    print("\n====================================================")
    print(f" RANDOM-RANK ATTACK GENERATION COMPLETE (seed={SEED})")
    print("====================================================\n")


if __name__ == "__main__":
    run_random_attack_pipeline()