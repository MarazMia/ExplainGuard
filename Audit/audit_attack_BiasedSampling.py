import os
import random
import sys
from typing import Dict, List, Tuple
import cvxpy as cp
import dill as pickle
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind
import shap
import xgboost as xgb

# Set global random seed for 100% reproducibility across runs
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 1. Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Model.blackbox_endpoint import BlackBoxModelEndpoint

DATASET_NAME = "compas"
ATTACK_PREFIX = "attack_BiasedSampling_"
MODEL_DIR = os.path.join(PROJECT_ROOT, "Model")
AUDITOR_DATA_DIR = os.path.join(
    PROJECT_ROOT, "Data", "Auditor_End", DATASET_NAME
)
SUBMISSION_DIR = os.path.join(
    PROJECT_ROOT, "Data", "Auditor_End", "Submissions"
)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Load Auditor Data & Untouched Clean Model
X_audit_test = pd.read_csv(
    os.path.join(AUDITOR_DATA_DIR, f"{DATASET_NAME}_X_audit_test.csv")
)
feature_names = X_audit_test.columns.tolist()

model_path = os.path.join(
    MODEL_DIR, f"original_{DATASET_NAME}_xgboost_model.json"
)
clean_model = xgb.XGBClassifier()
clean_model.load_model(model_path)
clean_endpoint = BlackBoxModelEndpoint(clean_model)


# ============================================================
# ALGORITHM 2: DETECT_FRAUD (Laberge et al., 2023)
# ============================================================
def detect_fraud(
    f_D0: np.ndarray,
    f_D1: np.ndarray,
    f_S0: np.ndarray,
    f_S1: np.ndarray,
    alpha: float = 0.1,
) -> int:
    """
    Algorithm 2: Fraud Detection using two-sample tests.
    Checks if f(S_0) matches f(D_0) and f(S_1) matches f(D_1).
    """
    threshold = alpha / 4.0

    # KS and Wald (t-test) for group 0
    ks_p0 = ks_2samp(f_S0, f_D0).pvalue
    wald_p0 = ttest_ind(f_S0, f_D0, equal_var=False).pvalue

    # KS and Wald (t-test) for group 1
    ks_p1 = ks_2samp(f_S1, f_D1).pvalue
    wald_p1 = ttest_ind(f_S1, f_D1, equal_var=False).pvalue

    # Reject null if any p-value falls below alpha / 4
    if (
        ks_p0 < threshold
        or wald_p0 < threshold
        or ks_p1 < threshold
        or wald_p1 < threshold
    ):
        return 1  # Fraud Detected
    return 0  # Passed


# ============================================================
# ALGORITHM 1: COMPUTE_WEIGHTS (Laberge et al., 2023)
# ============================================================
def compute_weights(
    D1: np.ndarray, phi_s_individual: np.ndarray, lam: float
) -> np.ndarray:
    """
    Algorithm 1: Solves min_w beta * sum(w_j * phi_j) + lam * ||w - 1/N1||^2
    s.t. sum(w) = 1, w >= 0
    """
    N1 = len(D1)
    beta = np.sign(np.sum(phi_s_individual))
    if beta == 0:
        beta = 1.0

    w = cp.Variable(N1)
    w0 = np.ones(N1) / N1

    # Convex Objective: Target attribution minimization + L2 regularization penalty
    objective = cp.Minimize(
        beta * (phi_s_individual @ w) + lam * cp.sum_squares(w - w0)
    )
    constraints = [w >= 0, cp.sum(w) == 1]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except Exception:
        prob.solve(solver=cp.ECOS, verbose=False)

    if w.value is None:
        return w0

    weights = np.array(w.value).ravel()
    weights = np.maximum(weights, 0)
    return weights / np.sum(weights)


# ============================================================
# ALGORITHM 3: FOOL_SHAP (Laberge et al., 2023)
# ============================================================
def fool_shap(
    f_predict_fn,
    D0: np.ndarray,
    D1: np.ndarray,
    M: int = 50,
    lam_min: float = 1e-4,
    lam_max: float = 1.0,
    n_lam_steps: int = 10,
    tau: float = 0.05,
    alpha: float = 0.1,
    sensitive_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Algorithm 3: Whole procedure for Fooling SHAP with Stealthily Biased Sampling.
    """
    rng = np.random.default_rng(SEED)
    N0, N1 = len(D0), len(D1)

    # Line 2: S^1_0 sampled without cheating
    idx_S0 = rng.choice(N0, size=M, replace=True)
    S0_prime = D0[idx_S0]

    # Line 3: Compute individual SHAP contributions for each point in D1 against S0_prime
    explainer_init = shap.KernelExplainer(f_predict_fn, S0_prime)

    # Compute local SHAP values for D1
    shap_vals_D1 = explainer_init.shap_values(D1, nsamples=100)
    if isinstance(shap_vals_D1, list):
        shap_vals_D1 = (
            shap_vals_D1[1] if len(shap_vals_D1) > 1 else shap_vals_D1[0]
        )

    phi_s_individual = shap_vals_D1[:, sensitive_idx]

    # Line 4-5: Initialize solutions
    phi_star_s = np.abs(np.mean(phi_s_individual))
    B_star_weights = np.ones(N1) / N1

    # Pre-evaluate ground truth outputs for fast fraud detection
    f_D0 = f_predict_fn(D0)
    f_D1 = f_predict_fn(D1)
    f_S0 = f_predict_fn(S0_prime)

    # Line 6: Grid search over lambda in log space
    lambda_grid = np.logspace(
        np.log10(lam_max), np.log10(lam_min), num=n_lam_steps
    )

    print("[+] Running Algorithm 3 (FOOL_SHAP) search over lambda...")
    for lam in lambda_grid:
        omega = compute_weights(D1, phi_s_individual, lam)

        detection_count = 0
        for _ in range(100):
            idx_S1 = rng.choice(N1, size=M, replace=True, p=omega)
            S1_prime = D1[idx_S1]
            f_S1 = f_predict_fn(S1_prime)

            detection_count += detect_fraud(f_D0, f_D1, f_S0, f_S1, alpha)

        current_phi_s = np.abs(np.sum(omega * phi_s_individual))

        if current_phi_s < phi_star_s and detection_count < (100 * tau):
            phi_star_s = current_phi_s
            B_star_weights = omega.copy()

    # Line 15: Sample final S1_prime from B_star
    idx_S1_final = rng.choice(N1, size=M, replace=True, p=B_star_weights)
    S1_prime_final = D1[idx_S1_final]

    return S0_prime, S1_prime_final, phi_star_s


# ============================================================
# SERIALIZABLE EXPLAINER FOR AUDIT SUBMISSION
# ============================================================
class SerializableKernelExplainer:
    """Pickle-safe wrapper for KernelSHAP submission artifact."""

    def __init__(self, model_function, background_data):
        self.model_function = model_function
        self.background_data = np.ascontiguousarray(
            background_data, dtype=np.float64
        )
        self._native_explainer = None

    def reconstruct(self):
        if self._native_explainer is None:
            import shap  # Explicit local import ensures availability upon unpickling

            self._native_explainer = shap.KernelExplainer(
                self.model_function, self.background_data
            )
        return self._native_explainer

    @property
    def model(self):
        return self.reconstruct().model

    @property
    def data(self):
        return self.reconstruct().data


class ModelPredictionWrapper:
    """Pickle-safe model prediction function wrapper for SHAP."""

    def __init__(self, endpoint) -> None:
        self.endpoint = endpoint

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        preds = np.asarray(self.endpoint.predict_proba(x))
        return preds[:, 1] if preds.ndim == 2 else preds


# ============================================================
# AUDIT PIPELINE EXECUTION
# ============================================================
def run_biased_sampling_audit():
    print("====================================================")
    print(" EXPLAINGUARD - LABERGE ET AL. (ICLR '23) ALGORITHM 3")
    print("====================================================\n")

    X_pool = X_audit_test.to_numpy()

    # Identify protected feature index (e.g., 'sex')
    PROTECTED_FEATURE = "sex"
    sensitive_idx = feature_names.index(PROTECTED_FEATURE)
    print(
        f"[+] Target Sensitive Feature: '{PROTECTED_FEATURE}' (Index {sensitive_idx})"
    )

    # Split dataset into D0 (sex = 0) and D1 (sex = 1) for group-wise evaluation
    p_vals = X_pool[:, sensitive_idx]
    D0 = X_pool[p_vals == 0]
    D1 = X_pool[p_vals == 1]

    # Instantiate the self-contained wrapper
    predict_wrapper = ModelPredictionWrapper(clean_endpoint)

    # Run Algorithm 3 using predict_wrapper
    S0_prime, S1_prime_final, final_phi = fool_shap(
        f_predict_fn=predict_wrapper,
        D0=D0,
        D1=D1,
        M=50,
        lam_min=1e-4,
        lam_max=1.0,
        n_lam_steps=8,
        tau=0.05,
        alpha=0.1,
        sensitive_idx=sensitive_idx,
    )

    # Run Verification KernelSHAP on S1_prime_final background
    explainer_attacked = shap.KernelExplainer(predict_wrapper, S1_prime_final)
    shap_attacked = explainer_attacked.shap_values(
        X_pool[:100], nsamples=100
    )
    if isinstance(shap_attacked, list):
        shap_attacked = (
            shap_attacked[1] if len(shap_attacked) > 1 else shap_attacked[0]
        )

    mean_shaps = np.abs(shap_attacked).mean(axis=0)

    print("\n--- SHAP FEATURE IMPORTANCE AFTER ALGORITHM 3 ATTACK ---")
    summary_df = pd.DataFrame(
        {"Feature": feature_names, "Mean |SHAP|": mean_shaps}
    ).sort_values(by="Mean |SHAP|", ascending=False)
    print(summary_df.to_string(index=False))

    # Save Submission Object for Pillar I Audit Inspection
    serializable_explainer = SerializableKernelExplainer(
        model_function=predict_wrapper, background_data=S1_prime_final
    )

    submission_pkg = {
        "explainer_type": "KernelSHAP",
        "explainer_object": serializable_explainer,
        "explained_instances": X_pool[:100],
        "background_data": S1_prime_final,
        "feature_names": feature_names,
    }

    output_path = os.path.join(
        SUBMISSION_DIR, f"{ATTACK_PREFIX}{DATASET_NAME}_shap_submission.pkl"
    )
    with open(output_path, "wb") as f:
        pickle.dump(submission_pkg, f)

    print(f"\n[+] Saved Algorithm 3 submission artifact to: {output_path}")


if __name__ == "__main__":
    run_biased_sampling_audit()