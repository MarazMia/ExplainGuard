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
AUDIT_TYPE = "attack_RandomRank_"  # Set to "clean_" or "attack_OOD_" or "attack_Shuffling_" or "attack_BiasedSampling_" or "attack_RandomRank_"
DATASET_NAME = "compas"

MODEL_DIR = os.path.join(PROJECT_ROOT, "Model")
AUDITOR_DATA_DIR = os.path.join(
    PROJECT_ROOT, "Data", "Auditor_End", DATASET_NAME
)
SUBMISSION_DIR = os.path.join(
    PROJECT_ROOT, "Data", "Auditor_End", "Submissions"
)


# ============================================================
# 2. PILLAR II VERIFIER ENGINE
# ============================================================
class Pillar2Verifier:
    """
    Policy Decision Point (PDP) - Pillar II Engine.

    Enforces Semantic Validity Axioms (Efficiency & Null-Player Axioms)
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

    def compute_expected_marginal_contribution(
        self, x: np.ndarray, feature_idx: int
    ) -> float:
        """
        Calculates expected marginal contribution sigma_i for feature i
        integrated across the background dataset B.
        """
        bg_data = self.background_B
        x_repeated = np.tile(x, (len(bg_data), 1))
        x_perturbed = x_repeated.copy()

        # Replace target feature with background baseline values
        x_perturbed[:, feature_idx] = bg_data[:, feature_idx]

        preds_orig = self.f_prime(x_repeated)
        preds_perturbed = self.f_prime(x_perturbed)

        return float(np.max(np.abs(preds_orig - preds_perturbed)))

    # ------------------------------------------------------------
    # CHECK 1: SHAP Semantic Validity (Efficiency & Null-Player)
    # ------------------------------------------------------------
    def verify_shap_semantic_validity(
        self, delta_1: float = 0.05, delta_null: float = 1e-1, delta_2: float = 0.05
    ) -> Tuple[bool, int, int]:
        """
        Equation (5) evaluation for SHAP attributions.
        """
        raw_attributions = self.submission["attributions"]

        if isinstance(raw_attributions, list):
            attributions = np.array(
                raw_attributions[1] if len(raw_attributions) > 1 else raw_attributions[0]
            )
        else:
            attributions = np.array(raw_attributions)

        if attributions.ndim == 3:
            attributions = (
                attributions[:, :, 1]
                if attributions.shape[2] == 2
                else attributions[1, :, :]
            )

        mx_efficiency_dif = 0.0
        mx_axiom_diff = 0.0

        phi_0 = float(np.mean(self.f_prime(self.background_B)))
        actual_preds = self.f_prime(self.targets)

        efficiency_violations = 0
        null_player_violations = 0

        for idx in range(len(self.targets)):
            x_i = self.targets[idx]
            phi_i = attributions[idx]
            y_i = actual_preds[idx]

            # 1. Efficiency Check: | phi_0 + sum(phi_i) - f'(x) | < delta_1
            reconstructed_pred = phi_0 + np.sum(phi_i)
            efficiency_diff = abs(reconstructed_pred - y_i)
            mx_efficiency_dif = max(efficiency_diff, mx_efficiency_dif)

            if efficiency_diff >= delta_1:
                efficiency_violations += 1

            # 2. Null-Player Check: if sigma_j < delta_null => |phi_j| < delta_2
            for j in range(len(x_i)):
                sigma_j = self.compute_expected_marginal_contribution(
                    x_i, feature_idx=j
                )
                if sigma_j < delta_null:
                    if abs(phi_i[j]) >= delta_2:
                        null_player_violations += 1
                        mx_axiom_diff = max(mx_axiom_diff, abs(phi_i[j]))

        v_efficiency = efficiency_violations == 0
        v_null = null_player_violations == 0
        v_axiom = v_efficiency and v_null

        return v_axiom, mx_efficiency_dif, mx_axiom_diff

    # ------------------------------------------------------------
    # CHECK 2: LIME Semantic Validity (Efficiency & Null-Player)
    # ------------------------------------------------------------
    def verify_lime_semantic_validity(
        self, delta_1: float = 0.4, delta_null: float = 1e-3, delta_2: float = 0.05
    ) -> Tuple[bool, int, int]:
        attributions_list = self.submission["attributions"]
        intercepts = self.submission.get("intercepts", None)
        actual_preds = self.f_prime(self.targets)

        mx_efficiency_dif = 0.0
        mx_axiom_diff = 0.0

        efficiency_violations = 0
        null_player_violations = 0

        for idx in range(len(self.targets)):
            x_i = self.targets[idx]
            attr_dict = attributions_list[idx]
            y_i = actual_preds[idx]

            phi_i = np.array(
                [attr_dict.get(fname, 0.0) for fname in self.feature_names]
            )

            # LIME local intercept baseline (phi_0)
            phi_0 = intercepts[idx] if intercepts is not None else float(np.mean(self.f_prime(self.background_B)))

            # Correct LIME local surrogate evaluation: phi_0 + sum(phi_i * x_i)
            reconstructed_pred = phi_0 + np.sum(phi_i * x_i)
            efficiency_diff = abs(reconstructed_pred - y_i)
            
            # Track peak residual across all instances
            mx_efficiency_dif = max(efficiency_diff, mx_efficiency_dif)

            if efficiency_diff >= delta_1:
                efficiency_violations += 1

            for j in range(len(x_i)):
                sigma_j = self.compute_expected_marginal_contribution(x_i, feature_idx=j)
                if sigma_j < delta_null:
                    if abs(phi_i[j]) >= delta_2:
                        null_player_violations += 1
                        # Track the magnitude of the offending null-player attribution
                        mx_axiom_diff = max(mx_axiom_diff, abs(phi_i[j]))

        v_efficiency = efficiency_violations == 0
        v_null = null_player_violations == 0
        v_axiom = v_efficiency and v_null

        return v_axiom, mx_efficiency_dif, mx_axiom_diff


# ============================================================
# 3. VERIFICATION EXECUTION PIPELINE
# ============================================================
def run_pillar2_audit() -> None:
    print("====================================================")
    print("      POLICY DECISION POINT (PDP) - PILLAR II      ")
    print("====================================================\n")

    shap_path = os.path.join(
        SUBMISSION_DIR, f"{AUDIT_TYPE}{DATASET_NAME}_shap_submission.pkl"
    )
    lime_path = os.path.join(
        SUBMISSION_DIR, f"{AUDIT_TYPE}{DATASET_NAME}_lime_submission.pkl"
    )

    p2_passed = True

    # ----------------------------------------------------
    # Verify SHAP Submission
    # ----------------------------------------------------
    if os.path.exists(shap_path):
        print(f"[+] Loading SHAP submission artifact:\n    {shap_path}")
        with open(shap_path, "rb") as f:
            shap_bundle = pickle.load(f)

        verifier_shap = Pillar2Verifier(shap_bundle)

        print("\n--- Verifying SHAP Semantic Validity ---")
        v_shap, shap_eff_fail, shap_null_fail = (
            verifier_shap.verify_shap_semantic_validity(
                delta_1=1e-3, delta_null=1e-4, delta_2=1e-3
            )
        )
        print(f"  • SHAP Efficiency Violations  : {shap_eff_fail}")
        print(f"  • SHAP Null-Player Violations : {shap_null_fail}")
        print(
            f"  • SHAP Pillar II Status       : {'PASS' if v_shap else 'FAIL'}\n"
        )
        p2_passed = p2_passed and v_shap
    else:
        print(f"[!] File not found: {shap_path}\n")

    # ----------------------------------------------------
    # Verify LIME Submission
    # ----------------------------------------------------
    if os.path.exists(lime_path):
        print(f"[+] Loading LIME submission artifact:\n    {lime_path}")
        with open(lime_path, "rb") as f:
            lime_bundle = pickle.load(f)

        verifier_lime = Pillar2Verifier(lime_bundle)

        print("\n--- Verifying LIME Semantic Validity ---")
        v_lime, lime_eff_fail, lime_null_fail = (
            verifier_lime.verify_lime_semantic_validity()
        )
        print(f"  • LIME Max Efficiency Violations  : {lime_eff_fail}")
        print(f"  • LIME Max Null-Player Violations : {lime_null_fail}")
        print(
            f"  • LIME Pillar II Status       : {'PASS' if v_lime else 'FAIL'}\n"
        )
        p2_passed = p2_passed and v_lime
    else:
        print(f"[!] File not found: {lime_path}\n")

    print("====================================================")
    print(
        f" PILLAR II OVERALL AUDIT RESULT: "
        f"{'VERIFIED & TRUSTED [PASS]' if p2_passed else 'REJECTED [FAIL]'}"
    )
    print("====================================================\n")


if __name__ == "__main__":
    run_pillar2_audit()