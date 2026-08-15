import os
import sys
from typing import Any, Dict, List, Tuple
import dill as pickle
import numpy as np
import pandas as pd
from scipy import stats
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
AUDIT_TYPE = "clean_"  # Set to "clean_" or "attack_OOD_" or "attack_Shuffling_" or "attack_BiasedSampling_"
DATASET_NAME = "compas"
MODEL_DIR = os.path.join(PROJECT_ROOT, "Model")
AUDITOR_DATA_DIR = os.path.join(
    PROJECT_ROOT, "Data", "Auditor_End", DATASET_NAME
)
SUBMISSION_DIR = os.path.join(
    PROJECT_ROOT, "Data", "Auditor_End", "Submissions"
)


# ============================================================
# 2. PILLAR I VERIFIER ENGINE
# ============================================================
class Pillar1Verifier:
    """
    Policy Decision Point (PDP) - Pillar I Verifier.

    Extracts f' and background dataset B directly from the submitted explainer
    objects, standardizes predictions to 1D probability arrays, and executes
    all four verification checks.
    """

    def __init__(
        self,
        ground_truth_endpoint: BlackBoxModelEndpoint,
        explainer_submission: Dict[str, Any],
        secret_fingerprints: np.ndarray,
    ) -> None:
        """
        Parameters:
        -----------
        ground_truth_endpoint : BlackBoxModelEndpoint
            The ground-truth black-box model endpoint f.
        explainer_submission : dict
            Unpickled submission package dictionary.
        secret_fingerprints : np.ndarray
            Secret evaluation instances I_c held by the auditor.
        """
        # Ground Truth Endpoint: f(x) -> 1D positive class probability P(Y=1)
        self.f_raw = ground_truth_endpoint.predict_proba
        self.f = lambda x: self._standardize_output(self.f_raw(x))

        # Ground Truth Values on Secret Fingerprints: Y_GT = f(I_c)
        self.Ic = secret_fingerprints
        self.Y_GT = self.f(self.Ic)

        # Unpack Explainer Object
        self.explainer_obj = explainer_submission.get(
            "explainer_object", explainer_submission
        )

        # 1. Extract f' directly from the explainer object
        f_prime_raw = self._extract_f_prime(self.explainer_obj)
        self.f_prime = lambda x: self._standardize_output(f_prime_raw(x))

        # 2. Extract background dataset B directly from the explainer object
        self.background_B = self._extract_background_dataset(self.explainer_obj)

    @staticmethod
    def _standardize_output(preds: np.ndarray) -> np.ndarray:
        """
        Ensures predictions are consistently formatted as a 1D numpy array
        representing P(Y=1).
        """
        preds = np.asarray(preds)
        if preds.ndim == 2:
            if preds.shape[1] == 2:
                return preds[:, 1]  # Extract positive class probability
            elif preds.shape[1] == 1:
                return preds.ravel()
        return preds.ravel()

    def _extract_f_prime(self, explainer: Any) -> Any:
        """
        Extracts f' directly from internal properties of KernelSHAP
        (including SerializableKernelExplainer) or TabularLIME.
        """
        # 1. SerializableKernelExplainer or native KernelExplainer
        if hasattr(explainer, "reconstruct"):
            native_explainer = explainer.reconstruct()
            if hasattr(native_explainer, "model") and hasattr(
                native_explainer.model, "f"
            ):
                return native_explainer.model.f

        if hasattr(explainer, "model") and hasattr(explainer.model, "f"):
            if callable(explainer.model.f):
                return explainer.model.f

        # 2. TabularLIME Explainer via custom `.predict_fn` property
        if hasattr(explainer, "predict_fn") and callable(explainer.predict_fn):
            return explainer.predict_fn

        raise AttributeError(
            f"Failed Security Inspection: Could not extract embedded model function f' "
            f"from explainer object of type '{type(explainer).__name__}'."
        )

    def _extract_background_dataset(self, explainer: Any) -> np.ndarray:
        """
        Extracts background dataset B directly from the explainer object.
        """
        # 1. SerializableKernelExplainer / SHAP KernelExplainer
        if hasattr(explainer, "background_data"):
            return np.asarray(explainer.background_data)
        if hasattr(explainer, "data") and hasattr(explainer.data, "data"):
            return np.asarray(explainer.data.data)

        # 2. LIME Explainer
        if hasattr(explainer, "training_data"):
            return np.asarray(explainer.training_data)

        raise AttributeError(
            f"Failed Security Inspection: Could not extract embedded background dataset B "
            f"from explainer object of type '{type(explainer).__name__}'."
        )

    # ------------------------------------------------------------
    # CHECK 1: Behavioral Fingerprint Check (V_id)
    # ------------------------------------------------------------
    def verify_behavioral_fingerprint(
        self, delta_fin: float = 1e-3
    ) -> Tuple[bool, float]:
        """
        Eq (1): V_id = I(|| f(I_c) - f'(I_c) ||_max < delta_fin)
        """
        f_prime_ic = self.f_prime(self.Ic)
        max_diff = float(np.max(np.abs(self.Y_GT - f_prime_ic)))
        v_id = max_diff < delta_fin
        return v_id, max_diff

    # ------------------------------------------------------------
    # CHECK 2: Batch-Invariance Check / Output Shuffling (V_os)
    # ------------------------------------------------------------
    def verify_batch_invariance(
        self, delta_os: float = 1e-5
    ) -> Tuple[bool, float]:
        """
        Eq (2): V_os = I( max_i | f'(I_c)_i - f'(I_c[i]) | < delta_os )
        """
        batch_preds = self.f_prime(self.Ic)
        indiv_preds = np.array(
            [self.f_prime(self.Ic[i : i + 1])[0] for i in range(len(self.Ic))]
        )

        max_batch_diff = float(np.max(np.abs(batch_preds - indiv_preds)))
        v_os = max_batch_diff < delta_os
        return v_os, max_batch_diff

    # ------------------------------------------------------------
    # CHECK 3: Scaffold Detection / OOD Audit (V_sc)
    # ------------------------------------------------------------
    def generate_fingerprint_distributions(
        self, X_seed: np.ndarray, sparsity_k: int = 3, samples_M: int = 20
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Algorithm 1: Black-Box Model Fingerprinting Data Generation via
                    Deterministic-Stochastic OOD Probing.
        
        - S_expl: Explainer coalition perturbations (KMeans / LIME jitter).
        - S_ood : In-distribution baseline reference points (X_seed raw rows).
        """
        n, d = X_seed.shape
        if n < 2:
            raise ValueError("X_seed must contain at least 2 rows for reference sampling.")

        from sklearn.cluster import KMeans
        n_clusters = min(5, n)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_seed)
        kmeans_centers = kmeans.cluster_centers_

        feature_vars = np.var(X_seed, axis=0)
        feature_vars_safe = np.where(feature_vars > 0, feature_vars, 1e-8)

        S_expl_list, S_ood_list = [], []

        for seed_idx, x in enumerate(X_seed):
            S_expl_x = []

            # 1. Explainer-Style Perturbations (S_expl)
            for _ in range(samples_M):
                z = x.copy()
                s = np.random.choice([1, 2])

                if s == 1:
                    # KernelSHAP Style: Replace random coalition with KMeans center
                    coalition_size = np.random.randint(1, d) if d > 1 else 1
                    mask_idx = np.random.choice(d, size=coalition_size, replace=False)
                    centre = kmeans_centers[np.random.randint(0, len(kmeans_centers))]
                    z[mask_idx] = centre[mask_idx]
                else:
                    # LIME Style: Add local Gaussian perturbation
                    noise = np.random.normal(0, 0.1 * np.sqrt(feature_vars_safe), size=d)
                    z = z + noise

                S_expl_x.append(z)

            # 2. In-Distribution Baseline (S_ood / Reference Raw Data)
            # Unperturbed x repeated to compare against explainer perturbation behavior
            S_ood_x = np.tile(x, (samples_M, 1))

            S_expl_list.append(np.array(S_expl_x))
            S_ood_list.append(S_ood_x)

        return S_expl_list, S_ood_list
        

    def verify_scaffold_detection(
        self,
        X_seed: np.ndarray,
        delta_ood: float = 0.08,
        sparsity_k: int = 3,
        samples_M: int = 20,
    ) -> Tuple[bool, float]:
        """
        Eq (3): Explainer-Conditioned OOD Audit using Algorithm 1
        """
        S_expl_list, S_ood_list = self.generate_fingerprint_distributions(
            X_seed, sparsity_k=sparsity_k, samples_M=samples_M
        )

        discrepancies = []
        for idx in range(len(X_seed)):
            S_expl = S_expl_list[idx]
            S_ood = S_ood_list[idx]

            E_expl = float(np.mean(self.f_prime(S_expl)))
            E_ood = float(np.mean(self.f_prime(S_ood)))

            delta_x = abs(E_expl - E_ood)
            discrepancies.append(delta_x)

        mean_delta = float(np.mean(discrepancies))
        v_sc = mean_delta < delta_ood
        return v_sc, mean_delta

    # ------------------------------------------------------------
    # CHECK 4: Biased Sampling Detection (V_sampling)
    # ------------------------------------------------------------
    def verify_biased_sampling(
        self, X_seed: np.ndarray, delta_sample: float = 0.6
    ) -> Tuple[bool, float, float]:
        """
        Eq (4): Biased Sampling Detection via KS-Test & Wald Test
        V_sampling = I( p_ks >= delta_sample/2  AND  p_wald >= delta_sample/2 )
        """
        Y_seed_prime = self.f_prime(X_seed)
        Y_B_prime = self.f_prime(self.background_B)

        # 1. Kolmogorov-Smirnov Test
        _, p_ks = stats.ks_2samp(Y_seed_prime, Y_B_prime)

        # 2. Wald Test / Welch's Two-Sample T-Test
        _, p_wald = stats.ttest_ind(Y_seed_prime, Y_B_prime, equal_var=False)

        p_ks, p_wald = float(p_ks), float(p_wald)
        threshold = delta_sample / 2.0
        v_sampling = (p_ks >= threshold) and (p_wald >= threshold)

        return v_sampling, p_ks, p_wald


# ============================================================
# 3. VERIFICATION EXECUTION PIPELINE
# ============================================================
def run_pillar1_audit(submission_file: str) -> None:
    print("====================================================")
    print("      EXPLAINGUARD - PILLAR I VERIFIER HARNESS      ")
    print("====================================================\n")

    # Load Ground Truth Model Endpoint f
    model_path = os.path.join(
        MODEL_DIR, f"original_{DATASET_NAME}_xgboost_model.json"
    )
    clean_model = xgb.XGBClassifier()
    clean_model.load_model(model_path)
    ground_truth_endpoint = BlackBoxModelEndpoint(clean_model)

    # Load Auditor Evaluation Dataset
    audit_path = os.path.join(
        AUDITOR_DATA_DIR, f"{DATASET_NAME}_X_audit_test.csv"
    )
    X_audit_test = pd.read_csv(audit_path)

    # Secret Fingerprints (I_c)
    np.random.seed(42)
    secret_indices = np.random.choice(
        len(X_audit_test), size=20, replace=False
    )
    I_c = X_audit_test.iloc[secret_indices].to_numpy()

    # Load Submission Package
    print(f"[+] Loading submission package:\n    {submission_file}")
    with open(submission_file, "rb") as f_pkl:
        submission_pkg = pickle.load(f_pkl)

    explainer_type = submission_pkg.get("explainer_type", "Unknown")
    print(f"    --> Explainer Type Detected: {explainer_type}\n")

    # Instantiate PDP Verifier
    verifier = Pillar1Verifier(
        ground_truth_endpoint=ground_truth_endpoint,
        explainer_submission=submission_pkg,
        secret_fingerprints=I_c,
    )

    X_seed = X_audit_test.iloc[:50].to_numpy()

    # ----------------------------------------------------
    # Execute Pillar I Verification Checks
    # ----------------------------------------------------
    print("----------------------------------------------------")
    print(" PILLAR I SECURITY CHECKS EXECUTION ")
    print("----------------------------------------------------")

    # Check 1: Behavioral Fingerprint
    v_id, max_diff_id = verifier.verify_behavioral_fingerprint()
    print(
        f" [1/4] Behavioral Fingerprint Check (V_id)    : {'PASS' if v_id else 'FAIL'}"
    )
    print(f"       └── Max Output Difference: {max_diff_id:.8f}")

    # Check 2: Batch Invariance
    v_os, max_diff_os = verifier.verify_batch_invariance()
    print(
        f" [2/4] Batch-Invariance Check (V_os)         : {'PASS' if v_os else 'FAIL'}"
    )
    print(f"       └── Max Shuffling Discrepancy: {max_diff_os:.8f}")

    # Check 3: Scaffold Detection
    v_sc, mean_delta_sc = verifier.verify_scaffold_detection(
        X_seed, sparsity_k=3, samples_M=20
    )
    print(
        f" [3/4] Scaffold Detection Check (V_sc)        : {'PASS' if v_sc else 'FAIL'}"
    )
    print(f"       └── Mean OOD Discrepancy: {mean_delta_sc:.8f}")

    # Check 4: Biased Sampling
    v_sampling, p_ks, p_wald = verifier.verify_biased_sampling(X_seed)
    print(
        f" [4/4] Biased Sampling Check (V_sampling)     : {'PASS' if v_sampling else 'FAIL'}"
    )
    print(f"       └── p_ks: {p_ks:.4f} | p_wald: {p_wald:.4f}")

    # Overall Decision
    p1_passed = v_id and v_os and v_sc and v_sampling

    print("\n====================================================")
    print(
        f" PILLAR I OVERALL AUDIT RESULT: "
        f"{'VERIFIED & TRUSTED [PASS]' if p1_passed else 'REJECTED [FAIL]'}"
    )
    print("====================================================\n")


if __name__ == "__main__":
    # Detect all generated submission files in the Submissions directory
    submission_files = [
        os.path.join(SUBMISSION_DIR, f"{AUDIT_TYPE}{DATASET_NAME}_shap_submission.pkl"),
        os.path.join(SUBMISSION_DIR, f"{AUDIT_TYPE}{DATASET_NAME}_lime_submission.pkl"),
    ]

    print("====================================================")
    print("      EXPLAINGUARD - FULL PILLAR I AUDIT SUITE      ")
    print("====================================================\n")

    for submission_path in submission_files:
        if os.path.exists(submission_path):
            run_pillar1_audit(submission_path)
            print("\n" + "=" * 52 + "\n")
        else:
            print(f"[!] Warning: File not found -> {submission_path}")