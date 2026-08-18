
import os
import sys
from typing import Any, Callable, Dict, List, Tuple
import dill as pickle
import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

# ============================================================
# 1. DYNAMIC PROJECT ROOT & PATH RESOLUTION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Model.blackbox_endpoint import BlackBoxModelEndpoint

# Setup relative paths
DATASET_NAME = "compas"
ATTACK_PREFIX = "attack_Shuffling_"

MODEL_DIR = os.path.join(PROJECT_ROOT, "Model")
AUDITOR_DATA_DIR = os.path.join(
    PROJECT_ROOT, "Data", "Auditor_End", DATASET_NAME
)
SUBMISSION_DIR = os.path.join(
    PROJECT_ROOT, "Data", "Auditor_End", "Submissions"
)

os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Load Auditor Evaluation Dataset
X_audit_test = pd.read_csv(
    os.path.join(AUDITOR_DATA_DIR, f"{DATASET_NAME}_X_audit_test.csv")
)
feature_names = X_audit_test.columns.tolist()

# Load Original Clean Model
model_path = os.path.join(
    MODEL_DIR, f"original_{DATASET_NAME}_xgboost_model.json"
)
clean_model = xgb.XGBClassifier()
clean_model.load_model(model_path)
clean_endpoint = BlackBoxModelEndpoint(clean_model)


# ============================================================
# 2. ADVERSARIAL SHUFFLED MODEL (Algorithms 1 & 4 Implementation)
# ============================================================
class AdversarialShuffledModel:
    """
    Wrapper model executing Adversarial Mixing Output Shuffling.
    Supports multi-feature group partitioning via positional integer encoding.
    """

    def __init__(
        self,
        clean_endpoint: BlackBoxModelEndpoint,
        protected_indices: List[int],
        higher_is_superior: bool = True,
        mixing_p: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.clean_endpoint = clean_endpoint
        self.protected_indices = protected_indices
        self.higher_is_superior = higher_is_superior
        self.mixing_p = mixing_p
        self.random_state = random_state

    def _apply_mixing_attack(
        self,
        X: np.ndarray,
        y_clean: np.ndarray,
        p: float = 0.5,
    ) -> np.ndarray:
        N = len(y_clean)
        if N <= 1:
            return y_clean.copy()

        y_clean = np.asarray(y_clean, dtype=float).ravel()
        y_final = y_clean.copy()

        # --------------------------------------------------------
        # 1. Multi-Feature Signature Encoding (Positional Integer Base-2)
        # --------------------------------------------------------
        protected_group = np.zeros(N, dtype=int)
        for bit_pos, p_idx in enumerate(self.protected_indices):
            p_vals = X[:, p_idx]
            p_median = np.median(p_vals)
            p_bin = (p_vals > p_median).astype(int)
            protected_group += p_bin * (2**bit_pos)  # Unique encoding: 0, 1, 2, 3...

        # Exit if all instances in batch belong to the exact same group
        unique_groups = np.unique(protected_group)
        if len(unique_groups) <= 1:
            return y_final

        rng = np.random.default_rng(self.random_state)

        # --------------------------------------------------------
        # 2. Sort scores according to superiority
        # --------------------------------------------------------
        if self.higher_is_superior:
            score_order = np.argsort(-y_clean)
        else:
            score_order = np.argsort(y_clean)

        sorted_scores = y_clean[score_order]

        # --------------------------------------------------------
        # 3. Probabilistic Multi-Group Re-assignment
        # --------------------------------------------------------
        # Bucket indices by their positional group signature
        group_buckets = {g: [i for i in score_order if protected_group[i] == g] for g in unique_groups}

        for rank in range(N):
            # Identify groups that still have unassigned instances
            active_groups = [g for g in unique_groups if len(group_buckets[g]) > 0]

            if len(active_groups) == 1:
                selected_group = active_groups[0]
            else:
                # Uniform random selection across active protected groups
                selected_group = rng.choice(active_groups)

            selected_idx = group_buckets[selected_group].pop(0)
            y_final[selected_idx] = sorted_scores[rank]

        return y_final

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        y_clean = np.asarray(self.clean_endpoint.predict_proba(X))
        y_score = y_clean[:, 1] if y_clean.ndim == 2 else y_clean

        return self._apply_mixing_attack(X, y_score, p=self.mixing_p)

    


# ============================================================
# 3. SERIALIZABLE WRAPPERS FOR SHUFFLED MODEL & SHAP
# ============================================================
class SerializableShuffledModelFunction:
    """
    Serializable wrapper for the shuffled model function.
    Preserves batch prediction behavior for Pillar I inspection.
    """

    def __init__(self, shuffled_model: AdversarialShuffledModel) -> None:
        self.shuffled_model = shuffled_model

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        p1 = self.shuffled_model.predict_proba(x)
        p0 = 1.0 - p1

        return np.column_stack((p0, p1))


class SerializableKernelExplainer:
    """
    Pickle-safe representation of KernelSHAP for shuffling audit.
    """

    def __init__(
        self,
        model_function: Callable[[np.ndarray], np.ndarray],
        background_data: np.ndarray,
    ) -> None:
        self.model_function = model_function
        self.background_data = np.ascontiguousarray(
            background_data, dtype=np.float64
        )
        self._native_explainer = None

    def reconstruct(self) -> Any:
        if self._native_explainer is None:
            import shap

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
# 4. MAIN AUDIT EXECUTION PIPELINE
# ============================================================
def run_shuffling_audit() -> None:
    print("====================================================")
    print("  EXPLAINGUARD - ADVERSARIAL OUTPUT SHUFFLING ATTACK")
    print("====================================================\n")

    # Data Splits
    N_BG = 50
    # bg_data = X_audit_test.iloc[:N_BG].to_numpy()
    # explain_data = X_audit_test.iloc[N_BG:].to_numpy()

    # random sampling
    bg_df = X_audit_test.sample(n=N_BG, random_state=100)
    bg_data = bg_df.to_numpy()

    explain_data = X_audit_test.loc[X_audit_test.index.difference(bg_df.index)].to_numpy()

    # Protected Feature Target Identification
    PROTECTED_FEATURES = ["sex", "race_African-American"]
    protected_indices = []
    for fname in PROTECTED_FEATURES:
        if fname in feature_names:
            protected_indices.append(feature_names.index(fname))
            print(
                f"[+] Targeting protected feature '{fname}' at index {feature_names.index(fname)}"
            )

    print("\n[+] Constructing Adversarial Output Shuffled Model...")
    shuffled_model = AdversarialShuffledModel(
        clean_endpoint=clean_endpoint,
        protected_indices=protected_indices,
        higher_is_superior=True,
    )

    # # ----------------------------------------------------
    # # Verification of Single vs. Batch Shuffling Discrepancy
    # # ----------------------------------------------------
    # sample_test = explain_data[:20]
    # batch_preds = shuffled_model.predict_proba(sample_test)
    # indiv_preds = np.array(
    #     [shuffled_model.predict_proba(sample_test[i : i + 1])[0] for i in range(len(sample_test))]
    # )
    # max_batch_diff = float(np.max(np.abs(batch_preds - indiv_preds)))

    # print("\n--- BATCH VS. SINGLE INSTANCE DISCREPANCY CHECK ---")
    # print(f"  • Max Batch Shuffling Discrepancy (V_os metric): {max_batch_diff:.8f}")
    # print(
    #     f"  • Attack Behavior Verification                   : "
    #     f"{'SUCCESS - Batch Discrepancy Induced' if max_batch_diff > 1e-4 else 'INACTIVE'}"
    # )

    # Wrap Model Function
    shuffled_model_function = SerializableShuffledModelFunction(shuffled_model)

    def predict_wrapper(x: np.ndarray) -> np.ndarray:
        return shuffled_model_function(x)

    # ----------------------------------------------------
    # KernelSHAP Execution under Output Shuffling
    # ----------------------------------------------------
    print("\n----------------------------------------------------")
    print(" [1/2] RUNNING KERNEL SHAP UNDER OUTPUT SHUFFLING")
    print("----------------------------------------------------")

    shap_explainer = shap.KernelExplainer(predict_wrapper, bg_data)
    shap_values = shap_explainer.shap_values(explain_data)

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

    print("\n--- SHAP FEATURE IMPORTANCE UNDER SHUFFLING ---")
    shap_summary_df = pd.DataFrame(
        {"Feature": feature_names, "Mean |SHAP|": global_shap_importance}
    ).sort_values(by="Mean |SHAP|", ascending=False)
    print(shap_summary_df.to_string(index=False))

    print("\n--- PROTECTED FEATURE ATTRIBUTION STATUS (SHAP) ---")
    for fname, idx in zip(PROTECTED_FEATURES, protected_indices):
        val = global_shap_importance[idx]
        print(f"  • {fname:<22} (idx {idx}): {val:.6f}")

    # Serialize SHAP Submission
    serializable_shap_explainer = SerializableKernelExplainer(
        model_function=shuffled_model_function, background_data=bg_data
    )

    shap_submission = {
        "explainer_type": "KernelSHAP",
        "explainer_object": serializable_shap_explainer,
        "explained_instances": explain_data,
        "attributions": shap_values,
        "feature_names": feature_names,
    }

    shap_path = os.path.join(
        SUBMISSION_DIR, f"{ATTACK_PREFIX}{DATASET_NAME}_shap_submission.pkl"
    )
    print("\n[+] Saving Output Shuffling SHAP submission artifact...")
    with open(shap_path, "wb") as f:
        pickle.dump(shap_submission, f)
    print(f"    --> Saved to: {shap_path}")

    # ----------------------------------------------------
    # TabularLIME Execution under Output Shuffling
    # ----------------------------------------------------
    print("\n----------------------------------------------------")
    print(" [2/2] RUNNING TABULAR LIME UNDER OUTPUT SHUFFLING")
    print("----------------------------------------------------")

    binary_indices = [
        feature_names.index("sex"),
        feature_names.index("race_African-American"),
        feature_names.index("c_charge_degree_F"),
    ]

    categorical_names = {
        feature_names.index("sex"): ["0", "1"],
        feature_names.index("race_African-American"): ["0", "1"],
        feature_names.index("c_charge_degree_F"): ["0", "1"],
    }

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=bg_data,
        feature_names=feature_names,
        class_names=["Class 0", "Class 1"],
        mode="classification",
        categorical_features=binary_indices,
        categorical_names=categorical_names,
        discretize_continuous=False,   # keep this so continuous features stay continuous
        random_state=42,
    )

    lime_explainer.predict_fn = predict_wrapper
    lime_explainer.background_data = np.ascontiguousarray(
        bg_data, dtype=np.float64
    )

    # ----------------------------------------------------
    # Collect raw LIME attributions (still with "feat=val" keys)
    # ----------------------------------------------------
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
        lime_intercepts.append(float(exp.intercept[1]))

    # ----------------------------------------------------
    # Collapse categorical conditions back to original feature names
    # ----------------------------------------------------
    from collections import defaultdict

    def collapse_lime_dict(raw_dict: dict) -> dict:
        """
        Convert {'sex=0': 0.12, 'sex=1': -0.08, 'age': 0.05, ...}
        into     {'sex': 0.20, 'age': 0.05, ...}   (sum of absolute weights)
        """
        collapsed = defaultdict(float)
        for key, weight in raw_dict.items():
            # If the key looks like "feature=value", take only the feature part
            base_feat = key.split("=")[0] if "=" in key else key
            collapsed[base_feat] += abs(weight)          # use abs for importance
        return dict(collapsed)

    # Apply collapse to every instance
    lime_attributions_collapsed = [
        collapse_lime_dict(d) for d in lime_attributions_list
    ]

    # ----------------------------------------------------
    # Global importance on the original 8 features
    # ----------------------------------------------------
    lime_df = pd.DataFrame(lime_attributions_collapsed).fillna(0.0)
    # Ensure all original features are present (even if they never appeared)
    for f in feature_names:
        if f not in lime_df.columns:
            lime_df[f] = 0.0
    lime_df = lime_df[feature_names]          # force original order

    global_lime_importance = lime_df.mean(axis=0)

    print("\n--- LIME FEATURE IMPORTANCE UNDER SHUFFLING ---")
    lime_summary_df = pd.DataFrame(
        {
            "Feature": global_lime_importance.index,
            "Mean |LIME Weight|": global_lime_importance.values,
        }
    ).sort_values(by="Mean |LIME Weight|", ascending=False)
    print(lime_summary_df.to_string(index=False))

    print("\n--- PROTECTED FEATURE ATTRIBUTION STATUS (LIME) ---")
    for fname in PROTECTED_FEATURES:
        val = global_lime_importance.get(fname, 0.0)
        print(f"  • {fname:<22}: {val:.6f}")

    # ----------------------------------------------------
    # Submission (use the collapsed attributions so everything is consistent)
    # ----------------------------------------------------
    lime_submission = {
        "explainer_type": "TabularLIME",
        "explainer_object": lime_explainer,
        "explained_instances": explain_data,
        "attributions": lime_attributions_list,
        "intercepts": lime_intercepts,     # NEW
        "feature_names": feature_names,
        }

    lime_path = os.path.join(
        SUBMISSION_DIR, f"{ATTACK_PREFIX}{DATASET_NAME}_lime_submission.pkl"
    )
    print("\n[+] Saving Output Shuffling LIME submission artifact...")
    with open(lime_path, "wb") as f:
        pickle.dump(lime_submission, f)
    print(f"    --> Saved to: {lime_path}")

    print("\n====================================================")
    print(" OUTPUT SHUFFLING ATTACK PIPELINE COMPLETED SUCCESSFULLY! ")
    print("====================================================\n")


if __name__ == "__main__":
    run_shuffling_audit()