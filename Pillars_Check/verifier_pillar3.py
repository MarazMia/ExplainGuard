import os
import sys
from typing import Any, Dict, List, Tuple
import dill as pickle
import numpy as np
import pandas as pd
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
# 2. PILLAR III VERIFIER ENGINE
# ============================================================
class Pillar3Verifier:
    """
    Policy Decision Point (PDP) - Pillar III Engine.

    Enforces Feature Faithfulness via Top-k Deletion AUC vs. Random Baseline
    by extracting f' and background dataset B directly from embedded
    explainer objects.
    """

    def __init__(self, explainer_submission: Dict[str, Any]) -> None:
        """
        Parameters:
        -----------
        explainer_submission : dict
            Unpickled submission package dictionary containing 'explainer_object'.
        """
        self.submission = explainer_submission
        self.explainer_obj = explainer_submission.get(
            "explainer_object", explainer_submission
        )

        # 1. Extract f' directly from the explainer object
        f_prime_raw = self._extract_f_prime(self.explainer_obj)
        self.f_prime = lambda x: self._standardize_output(f_prime_raw(x))

        # 2. Extract background dataset B directly from the explainer object
        self.background_B = self._extract_background_dataset(self.explainer_obj)

        # 3. Extract evaluation targets & feature names
        self.targets = np.asarray(explainer_submission["explained_instances"])
        self.feature_names = explainer_submission.get("feature_names", [])

        # 4. Normalize attribution matrix
        self.attributions_matrix = self._extract_attributions_matrix()

    @staticmethod
    def _standardize_output(preds: np.ndarray) -> np.ndarray:
        """
        Ensures predictions are consistently formatted as a 1D numpy array
        representing P(Y=1).
        """
        preds = np.asarray(preds)
        if preds.ndim == 2:
            if preds.shape[1] == 2:
                return preds[:, 1]
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

        # 2. TabularLIME Explainer via `.predict_fn` property
        if hasattr(explainer, "predict_fn") and callable(explainer.predict_fn):
            return explainer.predict_fn

        raise AttributeError(
            f"Failed Security Inspection: Could not extract embedded model function f' "
            f"from explainer object of type '{type(explainer).__name__}'."
        )

    def _extract_background_dataset(self, explainer: Any) -> np.ndarray:
        """
        Extracts background dataset B directly from internal properties of the explainer.
        """
        # 1. SerializableKernelExplainer / SHAP KernelExplainer
        if hasattr(explainer, "background_data"):
            return np.asarray(explainer.background_data)
        if hasattr(explainer, "data") and hasattr(explainer.data, "data"):
            return np.asarray(explainer.data.data)

        # 2. LIME Explainer
        if hasattr(explainer, "background_data"):
            return np.asarray(explainer.background_data)
        if hasattr(explainer, "training_data"):
            return np.asarray(explainer.training_data)

        raise AttributeError(
            f"Failed Security Inspection: Could not extract embedded background dataset B "
            f"from explainer object of type '{type(explainer).__name__}'."
        )

    def _extract_attributions_matrix(self) -> np.ndarray:
        """
        Normalizes raw attributions from SHAP arrays or LIME dictionary lists
        into a consistent 2D numpy array of shape (N_samples, N_features).
        """
        raw_attrs = self.submission["attributions"]

        # LIME Dictionary List Format
        if isinstance(raw_attrs, list) and len(raw_attrs) > 0 and isinstance(raw_attrs[0], dict):
            matrix = np.zeros(self.targets.shape)
            for idx, attr_dict in enumerate(raw_attrs):
                for j, fname in enumerate(self.feature_names):
                    matrix[idx, j] = attr_dict.get(fname, 0.0)
            return matrix

        # SHAP List/Array Format
        if isinstance(raw_attrs, list):
            attrs = np.array(raw_attrs[1] if len(raw_attrs) > 1 else raw_attrs[0])
        else:
            attrs = np.array(raw_attrs)

        if attrs.ndim == 3:
            attrs = (
                attrs[:, :, 1]
                if attrs.shape[2] == 2
                else attrs[1, :, :]
            )

        return attrs

    def compute_ablation_instance(
        self, x: np.ndarray, ablated_indices: np.ndarray
    ) -> np.ndarray:
        """
        Generates ablated instance x^(k) by replacing ablated_indices
        with baseline mean values drawn from the background dataset B.
        """
        x_ablated = x.copy()
        for idx in ablated_indices:
            x_ablated[idx] = np.mean(self.background_B[:, idx])
        return x_ablated

    def evaluate_faithfulness_auc(
        self, K: int = 4, M: int = 10, epsilon: float = 0.01
    ) -> Tuple[bool, float, float]:
        """
        Equation (6): Evaluates AUC_A vs (AUC_M + epsilon) across all target instances.
        """
        d = self.targets.shape[1]
        K = min(K, d)

        auc_A_list = []
        auc_M_list = []

        for idx in range(len(self.targets)):
            x = self.targets[idx]
            phi = self.attributions_matrix[idx]
            y0 = self.f_prime(x.reshape(1, -1))[0]

            # ----------------------------------------------------
            # 1. Auditor Ranking AUC (AUC_A)
            # ----------------------------------------------------
            # Rank features by descending magnitude
            auditor_ranking = np.argsort(-np.abs(phi))

            delta_A_k = []
            for k in range(1, K + 1):
                top_k_indices = auditor_ranking[:k]
                x_k = self.compute_ablation_instance(x, top_k_indices)
                y_k = self.f_prime(x_k.reshape(1, -1))[0]
                delta_A_k.append(abs(y0 - y_k))

            auc_A = float(np.mean(delta_A_k))
            auc_A_list.append(auc_A)

            # ----------------------------------------------------
            # 2. Random Baseline AUC (AUC_M)
            # ----------------------------------------------------
            auc_M_j = []
            for _ in range(M):
                random_ranking = np.random.permutation(d)
                delta_M_k = []
                for k in range(1, K + 1):
                    random_k_indices = random_ranking[:k]
                    x_rand_k = self.compute_ablation_instance(x, random_k_indices)
                    y_rand_k = self.f_prime(x_rand_k.reshape(1, -1))[0]
                    delta_M_k.append(abs(y0 - y_rand_k))
                auc_M_j.append(np.mean(delta_M_k))

            auc_M = float(np.mean(auc_M_j))
            auc_M_list.append(auc_M)

        mean_auc_A = float(np.mean(auc_A_list))
        mean_auc_M = float(np.mean(auc_M_list))

        # Verification Rule: V_fidelity = I(AUC_A > AUC_M + epsilon)
        v_fidelity = mean_auc_A > (mean_auc_M + epsilon)

        return v_fidelity, mean_auc_A, mean_auc_M


# ============================================================
# 3. VERIFICATION EXECUTION PIPELINE
# ============================================================
def run_pillar3_audit() -> None:
    print("====================================================")
    print("      POLICY DECISION POINT (PDP) - PILLAR III     ")
    print("====================================================\n")

    shap_path = os.path.join(
        SUBMISSION_DIR, f"{AUDIT_TYPE}{DATASET_NAME}_shap_submission.pkl"
    )
    lime_path = os.path.join(
        SUBMISSION_DIR, f"{AUDIT_TYPE}{DATASET_NAME}_lime_submission.pkl"
    )

    p3_passed = True

    # ----------------------------------------------------
    # Verify SHAP Submission
    # ----------------------------------------------------
    if os.path.exists(shap_path):
        print(f"[+] Loading SHAP submission artifact:\n    {shap_path}")
        with open(shap_path, "rb") as f:
            shap_bundle = pickle.load(f)

        verifier_shap = Pillar3Verifier(shap_bundle)

        print("\n--- Verifying SHAP Feature Faithfulness ---")
        v_shap, shap_auc_A, shap_auc_M = verifier_shap.evaluate_faithfulness_auc(
            K=4, M=10, epsilon=0.01
        )
        print(f"  • SHAP Auditor AUC_A  : {shap_auc_A:.4f}")
        print(f"  • SHAP Random AUC_M   : {shap_auc_M:.4f}")
        print(
            f"  • SHAP Pillar III     : {'PASS' if v_shap else 'FAIL'}\n"
        )
        p3_passed = p3_passed and v_shap
    else:
        print(f"[!] File not found: {shap_path}\n")

    # ----------------------------------------------------
    # Verify LIME Submission
    # ----------------------------------------------------
    if os.path.exists(lime_path):
        print(f"[+] Loading LIME submission artifact:\n    {lime_path}")
        with open(lime_path, "rb") as f:
            lime_bundle = pickle.load(f)

        verifier_lime = Pillar3Verifier(lime_bundle)

        print("\n--- Verifying LIME Feature Faithfulness ---")
        v_lime, lime_auc_A, lime_auc_M = verifier_lime.evaluate_faithfulness_auc(
            K=4, M=10, epsilon=0.01
        )
        print(f"  • LIME Auditor AUC_A  : {lime_auc_A:.4f}")
        print(f"  • LIME Random AUC_M   : {lime_auc_M:.4f}")
        print(
            f"  • LIME Pillar III     : {'PASS' if v_lime else 'FAIL'}\n"
        )
        p3_passed = p3_passed and v_lime
    else:
        print(f"[!] File not found: {lime_path}\n")

    print("====================================================")
    print(
        f" PILLAR III OVERALL AUDIT RESULT: "
        f"{'VERIFIED & TRUSTED [PASS]' if p3_passed else 'REJECTED [FAIL]'}"
    )
    print("====================================================\n")


if __name__ == "__main__":
    run_pillar3_audit()