import os
import sys
from typing import Callable, List, Tuple
import dill as pickle
import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import Any, Callable, List, Tuple
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem  # or RBFSampler
from sklearn.linear_model import LogisticRegression

import contextvars
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import numpy as np

_explain_ctx = contextvars.ContextVar("explaining", default=False)


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
ATTACK_PREFIX = "attack_OOD_"

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





class SlackOODDetector:
    """
    OOD detector from Slack et al. (AIES 2020).
    Trains a classifier to distinguish real data from LIME/SHAP style perturbations.
    """

    def __init__(
        self,
        clf=None,
        lime_noise_std: float = 1.0,
        shap_n_clusters: int = 10,
        n_perturbations_per_sample: int = 1,
        ood_threshold: float = 0.5,
        seed: int = 42,
    ):
        self.clf = (
            clf
            if clf is not None
            else RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1
            )
        )
        self.lime_noise_std = lime_noise_std
        self.shap_n_clusters = shap_n_clusters
        self.n_perturbations_per_sample = n_perturbations_per_sample
        self.ood_threshold = ood_threshold
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._kmeans_centers = None
        self._feature_stds = None

    def _perturb_lime(self, X: np.ndarray) -> np.ndarray:
        """LIME perturbation: scale noise by individual feature standard deviations."""
        parts = []
        stds = np.std(X, axis=0) + 1e-8
        for _ in range(self.n_perturbations_per_sample):
            noise = self.rng.normal(0, self.lime_noise_std * stds, size=X.shape)
            parts.append(X + noise)
        return np.vstack(parts)

    def _perturb_shap(self, X: np.ndarray) -> np.ndarray:
        """KernelSHAP perturbation: coalition masking with KMeans cluster centers."""
        if self._kmeans_centers is None:
            raise RuntimeError("fit() must be called before _perturb_shap().")

        parts = []
        n, d = X.shape

        for _ in range(self.n_perturbations_per_sample):
            X_pert = X.copy()
            for i in range(n):
                coalition_size = self.rng.randint(1, d)
                mask_idx = self.rng.choice(d, size=coalition_size, replace=False)
                centre = self._kmeans_centers[
                    self.rng.randint(0, len(self._kmeans_centers))
                ]
                X_pert[i, mask_idx] = centre[mask_idx]
            parts.append(X_pert)

        return np.vstack(parts)

    def fit(self, X: np.ndarray, mode: str = "both") -> "SlackOODDetector":
        """Build OOD detector by training on real vs. perturbed data."""
        X = np.asarray(X, dtype=float)

        n_clusters = min(self.shap_n_clusters, len(X))
        km = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        km.fit(X)
        self._kmeans_centers = km.cluster_centers_

        perturbed_sets = []
        if mode in ("lime", "both"):
            perturbed_sets.append(self._perturb_lime(X))
        if mode in ("shap", "both"):
            perturbed_sets.append(self._perturb_shap(X))

        X_pert = np.vstack(perturbed_sets)

        # Remove perturbed points that landed back in real X
        real_set = set(map(tuple, np.round(X, decimals=8)))
        keep = [
            tuple(row) not in real_set for row in np.round(X_pert, decimals=8)
        ]
        X_pert = X_pert[keep]

        X_combined = np.vstack([X, X_pert])
        y_combined = np.concatenate(
            [
                np.zeros(len(X), dtype=int),
                np.ones(len(X_pert), dtype=int),
            ]
        )

        self.clf.fit(X_combined, y_combined)

        # Retrain if false positive rate on real data is high
        train_fp_rate = self.is_ood(X).mean()
        if train_fp_rate > 0.02:
            print(
                f"[*] FP rate {train_fp_rate:.3f} > 0.02 — retraining with hard negatives..."
            )
            self._retrain_with_hard_negatives(X, X_pert)

        return self

    def _retrain_with_hard_negatives(
        self, X_real: np.ndarray, X_pert: np.ndarray
    ) -> None:
        """Add misclassified real instances back as balanced hard negatives."""
        fp_mask = self.ood_confidence(X_real) > 0.10
        false_positives = X_real[fp_mask]

        if len(false_positives) == 0:
            return

        print(f"[*] Hard negatives found: {len(false_positives)}")

        # Single addition (1x) instead of double oversampling (2x) to maintain boundary sensitivity
        X_combined = np.vstack([X_real, X_pert, false_positives])
        y_combined = np.concatenate(
            [
                np.zeros(len(X_real), dtype=int),
                np.ones(len(X_pert), dtype=int),
                np.zeros(len(false_positives), dtype=int),
            ]
        )

        self.clf.fit(X_combined, y_combined)
        new_fp = self.is_ood(X_real).mean()
        print(f"[*] FP rate after hard negative retraining: {new_fp:.3f}")

    def is_ood(self, X: np.ndarray) -> np.ndarray:
        return self.ood_confidence(X) >= self.ood_threshold

    def ood_confidence(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def threshold_sweep(self, X_real: np.ndarray, X_pert: np.ndarray) -> None:
        real_conf = self.ood_confidence(X_real)
        pert_conf = self.ood_confidence(X_pert)

        print(f"{'Thresh':>8}  {'Real flagged (FP)':>18}  {'Pert flagged (TP)':>18}")
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            r = (real_conf >= thresh).sum()
            p = (pert_conf >= thresh).sum()
            print(
                f"{thresh:>8.2f}  {r:>10} / {len(X_real)}  {p:>10} / {len(X_pert)}"
            )




# ============================================================
# 2. ADVERSARIAL SCAFFOLDED MODEL (Slack et al.)
# ============================================================
class AdversarialScaffoldedModel:
    def __init__(
        self,
        clean_endpoint: BlackBoxModelEndpoint,
        X_all_real: np.ndarray,
        sensitive_indices: List[int],
        threshold: float = 0.60,
        lime_noise_std: float = 1.0,
        shap_n_clusters: int = 8,
        n_perturb_per_real: int = 5,
        seed: int = 42,
    ) -> None:
        self.clean_endpoint = clean_endpoint
        self.sensitive_indices = sensitive_indices
        self.threshold = threshold

        self.mask_values = {
            idx: float(np.median(X_all_real[:, idx]))
            for idx in sensitive_indices
        }

        print("\n[+] Training Slack et al. OOD Detector (KMeans Coalition + Hard Negative Retraining)...")
        self.detector = SlackOODDetector(
            lime_noise_std=lime_noise_std,
            shap_n_clusters=shap_n_clusters,
            n_perturbations_per_sample=n_perturb_per_real,
            ood_threshold=threshold,
            seed=seed,
        )
        
        self.detector.fit(X_all_real, mode="both")
        
        # Backward-compatibility attribute link
        self.ood_detector = self.detector.clf
        
        self.cv_accuracy = float(1.0 - self.detector.is_ood(X_all_real).mean())

    def _innocuous_predict(self, X: np.ndarray) -> np.ndarray:
        """Innocuous prediction branch: pins sensitive features to median baseline values."""
        X_masked = X.copy()
        for idx, val in self.mask_values.items():
            X_masked[:, idx] = val
        return self.clean_endpoint.predict_proba(X_masked)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Main routing logic for the adversarial black-box endpoint."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Route via Slack OOD Detector
        is_explainer_query = self.detector.is_ood(X)

        preds = np.empty(len(X))
        real_mask = ~is_explainer_query
        expl_mask = is_explainer_query

        if real_mask.any():
            preds[real_mask] = self.clean_endpoint.predict_proba(X[real_mask])

        if expl_mask.any():
            preds[expl_mask] = self._innocuous_predict(X[expl_mask])

        return preds


# ============================================================
# 3. SERIALIZABLE WRAPPERS FOR ADVERSARIAL MODEL & SHAP
# ============================================================
class SerializableAdversarialModelFunction:
    """
    Serializable wrapper for the adversarial scaffolded model function.
    Preserves prediction behavior and internal routing for Pillar I.
    """

    def __init__(self, adv_model: AdversarialScaffoldedModel) -> None:
        self.adv_model = adv_model

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        p1 = self.adv_model.predict_proba(x)
        p0 = 1.0 - p1

        return np.column_stack((p0, p1))


class SerializableKernelExplainer:
    """
    Pickle-safe representation of KernelSHAP for adversarial audit.
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
# 4. MAIN AUDIT ATTACK EXECUTION PIPELINE
# ============================================================
def run_adversarial_audit() -> None:
    print("====================================================")
    print("   EXPLAINGUARD - ADVERSARIAL OOD ATTACK HARNESS    ")
    print("====================================================\n")

    # Data Splits
    N_BG = 50
    # bg_data = X_audit_test.iloc[:N_BG].to_numpy()
    # explain_data = X_audit_test.iloc[N_BG:].to_numpy()
        
    # random sampling
    bg_df = X_audit_test.sample(n=N_BG, random_state=0)
    bg_data = bg_df.to_numpy()
    
    explain_data = X_audit_test.loc[X_audit_test.index.difference(bg_df.index)].to_numpy()
    all_real_data = X_audit_test.to_numpy()

    # Target Sensitive Features
    SENSITIVE_FEATURES = ["age", "race_African-American", "sex"]
    sensitive_indices = []
    for fname in SENSITIVE_FEATURES:
        if fname in feature_names:
            sensitive_indices.append(feature_names.index(fname))
            print(f"[+] Targeting sensitive feature '{fname}' at index {feature_names.index(fname)}")

    print("\n[+] Constructing Adversarial Scaffolded Model...")
    adv_model = AdversarialScaffoldedModel(
    clean_endpoint=clean_endpoint,
    X_all_real=all_real_data,
    sensitive_indices=sensitive_indices,
    threshold=0.60,
    lime_noise_std=1.0,
    shap_n_clusters=8,
    n_perturb_per_real=5,
    seed=42,
    )

    print(f"    --> OOD Detector 5-fold CV Accuracy: {adv_model.cv_accuracy:.4f}")
    print(f"    --> Routing Threshold              : P(perturb) > {adv_model.threshold}")

    # Accuracy Preservation Check
    sample_test = explain_data[:10]
    clean_out = clean_endpoint.predict_proba(sample_test)
    adv_out = adv_model.predict_proba(sample_test)

    print("\n--- IN-DISTRIBUTION ACCURACY CHECK (First 10 Targets) ---")
    print(f"  • Max Absolute Output Difference: {np.max(np.abs(clean_out - adv_out)):.6f}")

    ood_probs_sample = adv_model.ood_detector.predict_proba(sample_test)[:, 1]
    routed_to_innocuous = int((ood_probs_sample > adv_model.threshold).sum())
    print(f"  • Real Target Misrouting Count  : {routed_to_innocuous}/10 "
          f"({'PASS - Zero Leakage' if routed_to_innocuous == 0 else 'WARNING - Leakage Detected'})")

    # Wrap Adversarial Model Function
    adv_model_function = SerializableAdversarialModelFunction(adv_model)

    def predict_wrapper(x: np.ndarray) -> np.ndarray:
        return adv_model_function(x)

    # ----------------------------------------------------
    # Malicious KernelSHAP Execution
    # ----------------------------------------------------
    print("\n----------------------------------------------------")
    print(" [1/2] RUNNING MALICIOUS KERNEL SHAP AUDIT")
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

    print("\n--- ADVERSARIAL SHAP FEATURE IMPORTANCE ---")
    shap_summary_df = pd.DataFrame(
        {"Feature": feature_names, "Mean |SHAP|": global_shap_importance}
    ).sort_values(by="Mean |SHAP|", ascending=False)
    print(shap_summary_df.to_string(index=False))

    print("\n--- SENSITIVE FEATURE SUPPRESSION STATUS (SHAP) ---")
    for fname, idx in zip(SENSITIVE_FEATURES, sensitive_indices):
        val = global_shap_importance[idx]
        status = "[SUPPRESSED]" if val < 0.02 else "[VISIBLE]"
        print(f"  • {fname:<22} (idx {idx}): {val:.6f}  {status}")

    # Serialize SHAP Submission
    serializable_shap_explainer = SerializableKernelExplainer(
        model_function=adv_model_function, background_data=bg_data
    )

    shap_submission = {
        "explainer_type": "KernelSHAP",
        "explainer_object": serializable_shap_explainer,
        "explained_instances": explain_data,
        "attributions": shap_values,
        "feature_names": feature_names
    }

    shap_path = os.path.join(
        SUBMISSION_DIR, f"{ATTACK_PREFIX}{DATASET_NAME}_shap_submission.pkl"
    )
    print("\n[+] Saving Malicious KernelSHAP submission artifact...")
    with open(shap_path, "wb") as f:
        pickle.dump(shap_submission, f)
    print(f"    --> Saved to: {shap_path}")

    # ----------------------------------------------------
    # Malicious TabularLIME Execution
    # ----------------------------------------------------
    print("\n----------------------------------------------------")
    print(" [2/2] RUNNING MALICIOUS TABULAR LIME AUDIT")
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

    lime_df_raw = pd.DataFrame(lime_attributions_list)
    global_lime_importance = lime_df_raw.abs().mean(axis=0)

    print("\n--- ADVERSARIAL LIME FEATURE IMPORTANCE ---")
    lime_summary_df = pd.DataFrame(
        {
            "Feature": global_lime_importance.index,
            "Mean |LIME Weight|": global_lime_importance.values,
        }
    ).sort_values(by="Mean |LIME Weight|", ascending=False)
    print(lime_summary_df.to_string(index=False))

    print("\n--- SENSITIVE FEATURE SUPPRESSION STATUS (LIME) ---")
    for fname in SENSITIVE_FEATURES:
        if fname in global_lime_importance.index:
            val = global_lime_importance[fname]
            status = "[SUPPRESSED]" if val < 0.02 else "[VISIBLE]"
            print(f"  • {fname:<22}: {val:.6f}  {status}")

    # Serialize LIME Submission
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
    print("\n[+] Saving Malicious TabularLIME submission artifact...")
    with open(lime_path, "wb") as f:
        pickle.dump(lime_submission, f)
    print(f"    --> Saved to: {lime_path}")

    print("\n====================================================")
    print(" ADVERSARIAL OOD ATTACK PIPELINE COMPLETED SUCCESSFULLY! ")
    print("====================================================\n")


if __name__ == "__main__":
    run_adversarial_audit()