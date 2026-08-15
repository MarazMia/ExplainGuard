import os
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_and_partition_dataset(
    dataset_name,
    fetch_data_fn,
    target_col=None,
    cols_to_drop=None,
    test_size=0.20,
    seed_size=200,
    audit_pool_size=300,
    random_state=42
):
    """
    Generalized Data Partitioning Pipeline for ExplainGuard.

    Preprocessing:
    ----------------
    Continuous numerical features:
        -> StandardScaler

    Binary features:
        -> Preserved as 0/1
        -> NOT standardized

    This is particularly important for binary protected/categorical
    attributes such as `sex`, `race_African-American`, and
    `c_charge_degree_F`.

    Data partitions:
    ----------------
    Dev_End:
        X_train_full
        y_train_full
        X_test_full
        y_test_full

    Owner_End:
        X_seed_secret
        y_seed_secret

    Auditor_End:
        X_audit_test
        y_audit_test
    """

    if cols_to_drop is None:
        cols_to_drop = []

    # ========================================================
    # 0. PROJECT DIRECTORIES
    # ========================================================

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.join(SCRIPT_DIR, "Data")

    DEV_DIR = os.path.join(
        BASE_DIR,
        "Dev_End",
        dataset_name
    )

    OWNER_DIR = os.path.join(
        BASE_DIR,
        "Owner_End",
        dataset_name
    )

    AUDITOR_DIR = os.path.join(
        BASE_DIR,
        "Auditor_End",
        dataset_name
    )

    # Force directory creation
    for directory in [
        DEV_DIR,
        OWNER_DIR,
        AUDITOR_DIR
    ]:
        os.makedirs(directory, exist_ok=True)

    print("\n==================================================")
    print(f" PROCESSING DATASET: {dataset_name.upper()}")
    print(f" Base Data Path: {BASE_DIR}")
    print("==================================================")

    # ========================================================
    # 1. FETCH RAW DATA
    # ========================================================

    X, y = fetch_data_fn()

    if isinstance(y, pd.DataFrame):
        y = y.squeeze()

    y = y.astype(int)

    print("\n[+] Raw dataset loaded")
    print(f"    Samples  : {len(X)}")
    print(f"    Features : {X.shape[1]}")

    # ========================================================
    # 2. DROP TARGET / UNNECESSARY COLUMNS
    # ========================================================

    if target_col and target_col in X.columns:
        X = X.drop(
            columns=[target_col],
            errors="ignore"
        )

    if cols_to_drop:
        print(
            f"\n[+] Dropping specified columns: "
            f"{cols_to_drop}"
        )

        X = X.drop(
            columns=cols_to_drop,
            errors="ignore"
        )

    # Make sure all remaining features are numerical
    X = X.astype(float)

    feature_names = X.columns.tolist()

    print("\n[+] Final feature set:")
    for idx, feature in enumerate(feature_names):
        print(f"    [{idx}] {feature}")

    # ========================================================
    # 3. IDENTIFY BINARY VS CONTINUOUS FEATURES
    # ========================================================
    #
    # A binary feature is defined using the RAW data.
    #
    # This is intentionally done BEFORE scaling.
    #
    # Example:
    #
    #     sex                  -> {0, 1}
    #     race_African-American -> {0, 1}
    #     c_charge_degree_F    -> {0, 1}
    #
    # These remain unchanged.
    #
    # Continuous features are standardized.
    # ========================================================

    binary_features = []
    continuous_features = []

    for feature in feature_names:

        unique_values = X[feature].dropna().unique()

        if (
            len(unique_values) == 2
            and set(unique_values).issubset({0.0, 1.0})
        ):
            binary_features.append(feature)

        else:
            continuous_features.append(feature)

    print("\n==================================================")
    print(" FEATURE PREPROCESSING CONFIGURATION")
    print("==================================================")

    print("\n[+] Binary / categorical features")
    print("    These will NOT be scaled:")

    for feature in binary_features:
        print(f"      - {feature}")

    print("\n[+] Continuous numerical features")
    print("    These WILL be standardized:")

    for feature in continuous_features:
        print(f"      - {feature}")

    # ========================================================
    # 4. TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("\n==================================================")
    print(" TRAIN / TEST SPLIT")
    print("==================================================")

    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")

    # ========================================================
    # 5. FIT SCALER ONLY ON TRAINING DATA
    # ========================================================
    #
    # IMPORTANT:
    #
    # The scaler is fitted only using continuous features
    # from X_train.
    #
    # Binary features are never passed through StandardScaler.
    # ========================================================

    scaler = StandardScaler()

    if len(continuous_features) > 0:

        scaler.fit(
            X_train[continuous_features]
        )

        X_train_cont_scaled = pd.DataFrame(
            scaler.transform(
                X_train[continuous_features]
            ),
            columns=continuous_features,
            index=X_train.index
        )

        X_test_cont_scaled = pd.DataFrame(
            scaler.transform(
                X_test[continuous_features]
            ),
            columns=continuous_features,
            index=X_test.index
        )

    else:

        X_train_cont_scaled = pd.DataFrame(
            index=X_train.index
        )

        X_test_cont_scaled = pd.DataFrame(
            index=X_test.index
        )

    # ========================================================
    # 6. PRESERVE BINARY FEATURES AS 0/1
    # ========================================================

    X_train_binary = X_train[
        binary_features
    ].copy()

    X_test_binary = X_test[
        binary_features
    ].copy()

    # ========================================================
    # 7. RECOMBINE FEATURES
    # ========================================================
    #
    # IMPORTANT:
    #
    # We reconstruct the dataframe using the ORIGINAL feature
    # ordering.
    #
    # This guarantees that the model sees exactly the same
    # column order as the raw dataset.
    # ========================================================

    X_train_scaled = pd.DataFrame(
        index=X_train.index,
        columns=feature_names,
        dtype=float
    )

    X_test_scaled = pd.DataFrame(
        index=X_test.index,
        columns=feature_names,
        dtype=float
    )

    # Continuous features
    if len(continuous_features) > 0:

        X_train_scaled[
            continuous_features
        ] = X_train_cont_scaled

        X_test_scaled[
            continuous_features
        ] = X_test_cont_scaled

    # Binary features
    if len(binary_features) > 0:

        X_train_scaled[
            binary_features
        ] = X_train_binary

        X_test_scaled[
            binary_features
        ] = X_test_binary

    # Ensure exact feature ordering
    X_train_scaled = X_train_scaled[
        feature_names
    ]

    X_test_scaled = X_test_scaled[
        feature_names
    ]

    # Reset indices only after preprocessing
    X_train_scaled = X_train_scaled.reset_index(
        drop=True
    )

    X_test_scaled = X_test_scaled.reset_index(
        drop=True
    )

    y_train = y_train.reset_index(
        drop=True
    )

    y_test = y_test.reset_index(
        drop=True
    )

    # ========================================================
    # 8. VERIFY PREPROCESSING
    # ========================================================

    print("\n==================================================")
    print(" PREPROCESSING VERIFICATION")
    print("==================================================")

    print("\n[+] Binary feature values after preprocessing:")

    for feature in binary_features:

        values = np.sort(
            X_train_scaled[feature].unique()
        )

        print(
            f"    {feature:30s}: "
            f"{values}"
        )

    print("\n[+] Continuous feature statistics after preprocessing:")

    if len(continuous_features) > 0:

        means = X_train_scaled[
            continuous_features
        ].mean()

        stds = X_train_scaled[
            continuous_features
        ].std()

        for feature in continuous_features:

            print(
                f"    {feature:30s}: "
                f"mean={means[feature]: .6f}, "
                f"std={stds[feature]: .6f}"
            )

    # ========================================================
    # 9. SAVE TO DEV_END
    # ========================================================

    X_train_scaled.to_csv(
        os.path.join(
            DEV_DIR,
            f"{dataset_name}_X_train_full.csv"
        ),
        index=False
    )

    y_train.to_csv(
        os.path.join(
            DEV_DIR,
            f"{dataset_name}_y_train_full.csv"
        ),
        index=False
    )

    X_test_scaled.to_csv(
        os.path.join(
            DEV_DIR,
            f"{dataset_name}_X_test_full.csv"
        ),
        index=False
    )

    y_test.to_csv(
        os.path.join(
            DEV_DIR,
            f"{dataset_name}_y_test_full.csv"
        ),
        index=False
    )

    # ========================================================
    # 10. OWNER_END
    # ========================================================
    #
    # Stratified 200-sample secret seed set
    # ========================================================

    c0_idx = y_train[
        y_train == 0
    ].sample(
        n=seed_size // 2,
        random_state=random_state
    ).index

    c1_idx = y_train[
        y_train == 1
    ].sample(
        n=seed_size // 2,
        random_state=random_state
    ).index

    seed_indices = c0_idx.union(c1_idx)

    X_seed = X_train_scaled.loc[
        seed_indices
    ].reset_index(drop=True)

    y_seed = y_train.loc[
        seed_indices
    ].reset_index(drop=True)

    X_seed.to_csv(
        os.path.join(
            OWNER_DIR,
            f"{dataset_name}_X_seed_secret.csv"
        ),
        index=False
    )

    y_seed.to_csv(
        os.path.join(
            OWNER_DIR,
            f"{dataset_name}_y_seed_secret.csv"
        ),
        index=False
    )

    # ========================================================
    # 11. AUDITOR_END
    # ========================================================
    #
    # 300 samples from the test set
    # ========================================================

    if audit_pool_size > len(X_test_scaled):
        raise ValueError(
            f"audit_pool_size={audit_pool_size} exceeds "
            f"available test samples={len(X_test_scaled)}."
        )

    X_audit_test = (
        X_test_scaled
        .iloc[:audit_pool_size]
        .reset_index(drop=True)
    )

    y_audit_test = (
        y_test
        .iloc[:audit_pool_size]
        .reset_index(drop=True)
    )

    X_audit_test.to_csv(
        os.path.join(
            AUDITOR_DIR,
            f"{dataset_name}_X_audit_test.csv"
        ),
        index=False
    )

    y_audit_test.to_csv(
        os.path.join(
            AUDITOR_DIR,
            f"{dataset_name}_y_audit_test.csv"
        ),
        index=False
    )

    # ========================================================
    # 12. FINAL SUMMARY
    # ========================================================

    print("\n==================================================")
    print(" DATASET PREPARATION COMPLETED")
    print("==================================================")

    print("\n[+] Files written to disk:")

    print(
        f"  [Dev_End]     -> "
        f"{dataset_name}_X_train_full.csv "
        f"{X_train_scaled.shape}"
    )

    print(
        f"                   -> "
        f"{dataset_name}_y_train_full.csv "
        f"{y_train.shape}"
    )

    print(
        f"                   -> "
        f"{dataset_name}_X_test_full.csv "
        f"{X_test_scaled.shape}"
    )

    print(
        f"                   -> "
        f"{dataset_name}_y_test_full.csv "
        f"{y_test.shape}"
    )

    print(
        f"  [Owner_End]   -> "
        f"{dataset_name}_X_seed_secret.csv "
        f"{X_seed.shape}"
    )

    print(
        f"                   -> "
        f"{dataset_name}_y_seed_secret.csv "
        f"{y_seed.shape}"
    )

    print(
        f"  [Auditor_End] -> "
        f"{dataset_name}_X_audit_test.csv "
        f"{X_audit_test.shape}"
    )

    print(
        f"                   -> "
        f"{dataset_name}_y_audit_test.csv "
        f"{y_audit_test.shape}"
    )

    print("\n[+] Preprocessing policy:")
    print("    Continuous features : StandardScaler")
    print("    Binary features     : unchanged 0/1")

    print("\n[+] Binary features:")
    print(f"    {binary_features}")

    print("\n[+] Continuous features:")
    print(f"    {continuous_features}")

    print("\n==================================================\n")


# ============================================================
# EXECUTION
# ============================================================

def fetch_compas_raw():
    compas = fetch_openml(
        data_id=42193,
        as_frame=True,
        parser="auto"
    )

    return compas.data, compas.target


if __name__ == "__main__":

    prepare_and_partition_dataset(
        dataset_name="compas",
        fetch_data_fn=fetch_compas_raw,

        cols_to_drop=[
            "age_cat_25-45",
            "age_cat_Greaterthan45",
            "age_cat_Lessthan25",
            "c_charge_degree_M",
            "race_Caucasian"
        ],

        seed_size=200,
        audit_pool_size=300,
        random_state=42
    )