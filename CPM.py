
import argparse
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


# ======================================================================
# CONFIG
# ======================================================================

LANDMARK_DIR = "data/landmarks/"
EXCEL_FILE   = "data/UCLP_clinical.xlsx"
OUTPUT_DIR   = "outputs/"
RANDOM_STATE = 42

LANDMARK_NAMES = [
    "F-15", "F-7", "F-1-1", "F-2", "F-11", "F-1", "F-8",
    "F-10", "F-5", "F-9", "F-14", "F-3", "F-12", "F-6",
]

CLINICAL_COLS = {
    "Birth Weight (kg)": "birth_weight",
    "Birth Weight Percentile": "birth_pct",
    "Pre-Tx Weight at Impression Date (kg)": "pretx_weight",
    "Pre-Tx Percentile at Impression Date": "pretx_pct",
}
CLINICAL_NAMES = list(CLINICAL_COLS.values())

GROUP_COL = 1
FILE_COL  = 3

PATIENT_KEY_RE = re.compile(r"(UCLP_[A-Za-z]{3}\d{3})")
SCAN_RE        = re.compile(r"UpperJawScan_(\d)")


# ======================================================================
# Utilities
# ======================================================================

def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def patient_key_from_string(s):
    m = PATIENT_KEY_RE.search(str(s))
    return m.group(1) if m else None


def scan_number_from_filename(fname):
    m = SCAN_RE.search(str(fname))
    return int(m.group(1)) if m else None


# ======================================================================
# 1. Load landmarks
# ======================================================================

def load_landmarks(txt_path):
    coords = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    return np.asarray(coords, dtype=float)


# ======================================================================
# 2. Normalize landmark configuration
# ======================================================================

def normalize_landmarks(coords):
    if coords.shape[0] < 2:
        return None, np.nan
    centroid = np.mean(coords, axis=0)
    centered = coords - centroid
    size = np.sqrt(np.sum(centered ** 2))
    if size <= 0 or not np.isfinite(size):
        return None, np.nan
    return centered / size, size


# ======================================================================
# 3. Pairwise distances
# ======================================================================

def pairwise_distances(coords):
    n = coords.shape[0]
    if n < 2:
        return np.array([])
    idx_i, idx_j = np.triu_indices(n, k=1)
    diffs = coords[idx_i] - coords[idx_j]
    return np.linalg.norm(diffs, axis=1)


# ======================================================================
# 4. Feature names
# ======================================================================

def make_distance_names(n_landmarks, landmark_names=None):
    idx_i, idx_j = np.triu_indices(n_landmarks, k=1)
    if landmark_names is not None and len(landmark_names) == n_landmarks:
        names_i = [landmark_names[i] for i in idx_i]
        names_j = [landmark_names[j] for j in idx_j]
    else:
        names_i = [f"L{i}" for i in idx_i]
        names_j = [f"L{j}" for j in idx_j]
    return [f"d_{a}_{b}" for a, b in zip(names_i, names_j)]


# ======================================================================
# 5. Load clinical table
# ======================================================================

def load_clinical_table(excel_path):
    df = pd.read_excel(excel_path, header=1)
    df.columns = [str(c).strip() for c in df.columns]

    records = {}
    n_skipped = 0

    for _, row in df.iterrows():
        group = str(row.iloc[GROUP_COL]).strip().upper()
        if group not in {"C", "I"}:
            n_skipped += 1
            continue

        file_cell = str(row.iloc[FILE_COL])
        for piece in file_cell.split(","):
            key = patient_key_from_string(piece)
            if key is None:
                continue

            rec = {"label": 1 if group == "C" else 0, "group": group}
            for col, short in CLINICAL_COLS.items():
                if col not in df.columns:
                    continue
                val = row[col]
                if isinstance(val, str):
                    val = val.replace("kg", "").strip()
                rec[short] = pd.to_numeric(val, errors="coerce")

            records[key] = rec

    print(f"Parsed {len(records)} unique patient keys "
          f"(skipped {n_skipped} non-C/I rows)")
    return records


# ======================================================================
# 6. Index landmark files by patient key + scan number
# ======================================================================

def index_landmark_files(landmark_dir):
    by_patient = defaultdict(dict)
    issues = Counter()

    for fname in sorted(os.listdir(landmark_dir)):
        if not fname.lower().endswith(".txt"):
            continue

        key = patient_key_from_string(fname)
        if key is None:
            issues["no_key"] += 1
            continue

        scan = scan_number_from_filename(fname)
        if scan not in (1, 2):
            issues["no_scan_number"] += 1
            continue

        if scan in by_patient[key]:
            issues["duplicate_scan"] += 1
            print(f"WARNING: duplicate Scan_{scan} for {key}:\n"
                  f"  kept    : {by_patient[key][scan]}\n"
                  f"  ignored : {fname}")
            continue

        by_patient[key][scan] = os.path.join(landmark_dir, fname)

    pairs = {}
    for key, scans in by_patient.items():
        if 1 in scans and 2 in scans:
            pairs[key] = {"pre": scans[1], "post": scans[2]}
        elif 1 in scans:
            issues["pre_only"] += 1
        elif 2 in scans:
            issues["post_only"] += 1

    print()
    print("=" * 70)
    print("LANDMARK FILE INDEXING")
    print("=" * 70)
    print(f"Patient keys with BOTH pre+post : {len(pairs)}")
    print(f"Pre-only                        : {issues['pre_only']}")
    print(f"Post-only                       : {issues['post_only']}")
    print(f"Duplicate scans                 : {issues['duplicate_scan']}")
    print(f"No UCLP key found               : {issues['no_key']}")
    print(f"No Scan_N in name               : {issues['no_scan_number']}")

    return pairs, issues


# ======================================================================
# 7. Build dataset (pre + post + delta + clinical)
# ======================================================================

def build_dataset(pairs, clinical_dict, landmark_names):
    expected_n = len(landmark_names)
    distance_names = make_distance_names(expected_n, landmark_names)

    pre_names   = [f"pre_{n}"   for n in distance_names]
    post_names  = [f"post_{n}"  for n in distance_names]
    delta_names = [f"delta_{n}" for n in distance_names]
    feature_names = pre_names + post_names + delta_names + CLINICAL_NAMES

    X_rows, y, keys_used = [], [], []
    n_missing_clinical = 0
    n_bad_landmarks = 0

    for key, scans in sorted(pairs.items()):
        if key not in clinical_dict:
            n_missing_clinical += 1
            continue

        pre_coords  = load_landmarks(scans["pre"])
        post_coords = load_landmarks(scans["post"])

        if (pre_coords.shape[0] != expected_n
                or post_coords.shape[0] != expected_n):
            n_bad_landmarks += 1
            print(f"Skipping {key}: pre has {pre_coords.shape[0]} landmarks, "
                  f"post has {post_coords.shape[0]}, expected {expected_n}")
            continue

        pre_norm,  _ = normalize_landmarks(pre_coords)
        post_norm, _ = normalize_landmarks(post_coords)
        if pre_norm is None or post_norm is None:
            n_bad_landmarks += 1
            continue

        pre_d   = pairwise_distances(pre_norm)
        post_d  = pairwise_distances(post_norm)
        delta_d = post_d - pre_d

        rec = clinical_dict[key]
        clinical_vec = [rec.get(c, np.nan) for c in CLINICAL_NAMES]

        X_rows.append(np.concatenate(
            [pre_d, post_d, delta_d, clinical_vec]))
        y.append(rec["label"])
        keys_used.append(key)

    if not X_rows:
        return np.empty((0, 0)), np.array([]), [], []

    X = np.vstack(X_rows)
    y = np.asarray(y)

    print()
    print("=" * 70)
    print("DATASET")
    print("=" * 70)
    print(f"Patients (pre+post+clinical): {len(y)}")
    print(f"Features:                     {X.shape[1]}")
    print(f"  PRE distances:              {len(pre_names)}")
    print(f"  POST distances:             {len(post_names)}")
    print(f"  DELTA distances:            {len(delta_names)}")
    print(f"  Clinical:                   {len(CLINICAL_NAMES)}")
    print(f"Class distribution:           {Counter(y)}")
    print(f"Missing values:               {np.isnan(X).sum()}")
    print(f"Missing clinical matches:     {n_missing_clinical}")
    print(f"Invalid landmark configs:     {n_bad_landmarks}")

    return X, y, keys_used, feature_names


# ======================================================================
# 8. Train / test split
# ======================================================================

def make_split(X, y, keys, seed):
    (X_train, X_test, y_train, y_test,
     keys_train, keys_test) = train_test_split(
        X, y, keys,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    print()
    print("=" * 70)
    print("80/20 PATIENT-LEVEL SPLIT")
    print("=" * 70)
    print(f"Training patients: {len(y_train)}")
    print(f"Test patients:     {len(y_test)}")
    print(f"Training classes:  {Counter(y_train)}")
    print(f"Test classes:      {Counter(y_test)}")
    print("\nTEST PATIENTS:")
    for key, label in sorted(zip(keys_test, y_test)):
        print(f"  {key:20s} {'C' if label == 1 else 'I'}")

    return X_train, X_test, y_train, y_test, keys_train, keys_test


# ======================================================================
# 9. Preprocessing
# ======================================================================

def preprocess_train_test(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    return X_train_scaled, X_test_scaled, imputer, scaler


# ======================================================================
# 10. Neural network
# ======================================================================

def build_model(input_dim, units=128, dropout=0.30, l2_reg=1e-3):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units, activation="relu", kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(units // 2, activation="relu", kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ======================================================================
# 11. Train + evaluate NN
# ======================================================================

def train_and_evaluate_nn(X_train, X_test, y_train, y_test, keys_test, seed):
    print()
    print("=" * 70)
    print("NEURAL NETWORK")
    print("=" * 70)

    min_class = min(Counter(y_train).values())
    k_neighbors = max(1, min(3, min_class - 1))

    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"Before SMOTE: {Counter(y_train)}")
    print(f"After SMOTE:  {Counter(y_train_smote)}")

    model = build_model(X_train_smote.shape[1])

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=30,
        restore_best_weights=True, verbose=0,
    )

    history = model.fit(
        X_train_smote, y_train_smote,
        validation_split=0.20,
        epochs=400,
        batch_size=8,
        callbacks=[early_stopping],
        verbose=0,
        shuffle=True,
    )

    proba = model.predict(X_test, verbose=0).ravel()
    preds = (proba >= 0.5).astype(int)

    print()
    print("--- TEST SET ---")
    for key, p, true in sorted(zip(keys_test, proba, y_test)):
        print(f"{key:20s} P(C)={p:.3f} "
              f"pred={'C' if p >= 0.5 else 'I'} "
              f"true={'C' if true == 1 else 'I'}")

    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    print()
    print(classification_report(
        y_test, preds, target_names=["I", "C"],
        digits=3, zero_division=0,
    ))
    print(f"Accuracy:          {accuracy_score(y_test, preds):.3f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, preds):.3f}")
    if len(np.unique(y_test)) > 1:
        print(f"ROC-AUC:           {roc_auc_score(y_test, proba):.3f}")
    try:
        print(f"F1:                {f1_score(y_test, preds):.3f}")
    except ValueError:
        pass

    return model, history, proba


# ======================================================================
# 12. Feature importance analysis
# ======================================================================

def feature_importance_analysis(
    X_train, X_test, y_train, y_test,
    feature_names, output_dir, seed,
):
    print()
    print("=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    rf = RandomForestClassifier(
        n_estimators=1000, max_depth=None, random_state=seed,
        class_weight="balanced", n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    perm = permutation_importance(
        rf, X_test, y_test, n_repeats=30, random_state=seed,
        n_jobs=-1, scoring="roc_auc",
    )

    lr = LogisticRegression(
        max_iter=5000, class_weight="balanced",
        random_state=seed, solver="liblinear",
    )
    lr.fit(X_train, y_train)
    lr_coef = lr.coef_.ravel()

    results = pd.DataFrame({
        "feature":              feature_names,
        "rf_importance":        rf.feature_importances_,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std":  perm.importances_std,
        "lr_coef":              lr_coef,
        "lr_abs_coef":          np.abs(lr_coef),
    })

    def block_of(name):
        if name.startswith("pre_"):   return "PRE"
        if name.startswith("post_"):  return "POST"
        if name.startswith("delta_"): return "DELTA"
        if name in CLINICAL_NAMES:    return "CLINICAL"
        return "OTHER"

    results["block"] = results["feature"].apply(block_of)
    results = (results
               .sort_values("perm_importance_mean", ascending=False)
               .reset_index(drop=True))
    results["rank"] = results.index + 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    full_path = os.path.join(output_dir,
                             f"feature_importance_full_{timestamp}.csv")
    results.to_csv(full_path, index=False)

    top20 = results.head(20)
    top20_path = os.path.join(output_dir,
                              f"feature_importance_top20_{timestamp}.csv")
    top20.to_csv(top20_path, index=False)

    block_summary = (results
                     .groupby("block")["perm_importance_mean"]
                     .sum()
                     .sort_values(ascending=False)
                     .reset_index())
    block_path = os.path.join(output_dir,
                              f"feature_importance_by_block_{timestamp}.csv")
    block_summary.to_csv(block_path, index=False)

    clinical = results[results["block"] == "CLINICAL"]
    clinical_path = os.path.join(output_dir,
                                 f"clinical_feature_importance_{timestamp}.csv")
    clinical.to_csv(clinical_path, index=False)

    summary_path = os.path.join(
        output_dir, f"feature_importance_summary_{timestamp}.txt"
    )
    with open(summary_path, "w") as f:
        f.write("UCLP CLOSURE PREDICTION  —  PRE / POST / DELTA\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(f"Training patients: {len(y_train)}\n")
        f.write(f"Test patients:     {len(y_test)}\n")
        f.write(f"Training classes:  {Counter(y_train)}\n")
        f.write(f"Test classes:      {Counter(y_test)}\n\n")

        f.write("=" * 80 + "\n")
        f.write("PERMUTATION IMPORTANCE BY BLOCK (sum)\n")
        f.write("=" * 80 + "\n\n")
        for _, row in block_summary.iterrows():
            f.write(f"{row['block']:<10s} "
                    f"{row['perm_importance_mean']:.5f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 20 FEATURES OVERALL\n")
        f.write("=" * 80 + "\n\n")
        for _, row in top20.iterrows():
            f.write(f"{int(row['rank']):3d}  "
                    f"{row['feature']:<30s} "
                    f"[{row['block']:<8s}] "
                    f"perm={row['perm_importance_mean']:.5f} "
                    f"+/- {row['perm_importance_std']:.5f}  "
                    f"RF={row['rf_importance']:.5f}  "
                    f"|LR|={row['lr_abs_coef']:.5f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("CLINICAL FEATURES\n")
        f.write("=" * 80 + "\n\n")
        for _, row in clinical.iterrows():
            f.write(f"{row['feature']:<20s} "
                    f"rank={int(row['rank']):3d} "
                    f"perm={row['perm_importance_mean']:.5f} "
                    f"RF={row['rf_importance']:.5f} "
                    f"|LR|={row['lr_abs_coef']:.5f}\n")

    print()
    print("--- IMPORTANCE BY BLOCK (sum) ---")
    print(block_summary.to_string(index=False))
    print()
    print("--- TOP 20 FEATURES ---")
    print(top20[[
        "rank", "block", "feature",
        "perm_importance_mean", "perm_importance_std",
        "rf_importance", "lr_abs_coef",
    ]].to_string(index=False))
    print()
    print(f"Full table:    {full_path}")
    print(f"Top 20:        {top20_path}")
    print(f"By block:      {block_path}")
    print(f"Clinical:      {clinical_path}")
    print(f"Summary:       {summary_path}")

    return results, top20, block_summary


# ======================================================================
# 13. Plots
# ======================================================================

def save_plots(y_test, proba, preds, top20, output_dir, timestamp):
    # ROC curve
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve (test set)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"roc_curve_{timestamp}.png"),
                    dpi=150)
        plt.close(fig)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["I", "C"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["I", "C"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"confusion_matrix_{timestamp}.png"),
                dpi=150)
    plt.close(fig)

    # Top-20 feature importance
    fig, ax = plt.subplots(figsize=(7, 6))
    y_pos = np.arange(len(top20))
    ax.barh(y_pos, top20["perm_importance_mean"][::-1],
            xerr=top20["perm_importance_std"][::-1],
            color="#4C72B0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20["feature"][::-1], fontsize=7)
    ax.set_xlabel("Permutation importance (ROC-AUC drop)")
    ax.set_title("Top 20 features")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"top_features_{timestamp}.png"),
                dpi=150)
    plt.close(fig)


# ======================================================================
# 14. Main
# ======================================================================

def main(args):
    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("CPM: Clinical Prediction Model")
    print("=" * 70)
    print(f"Landmark dir : {args.landmark_dir}")
    print(f"Clinical     : {args.excel}")
    print(f"Output       : {args.output_dir}")
    print(f"Seed         : {args.seed}")
    print()

    clinical = load_clinical_table(args.excel)
    pairs, _ = index_landmark_files(args.landmark_dir)

    X, y, keys, feature_names = build_dataset(
        pairs, clinical, LANDMARK_NAMES)

    if len(X) == 0:
        raise SystemExit("No matched patients found.")

    (X_train, X_test, y_train, y_test,
     keys_train, keys_test) = make_split(X, y, keys, args.seed)

    (X_train_scaled, X_test_scaled,
     imputer, scaler) = preprocess_train_test(X_train, X_test)

    model, history, proba = train_and_evaluate_nn(
        X_train_scaled, X_test_scaled,
        y_train, y_test, keys_test, args.seed,
    )

    preds = (proba >= 0.5).astype(int)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model + preprocessor
    model_path = os.path.join(args.output_dir, "uclp_classifier.keras")
    model.save(model_path)
    prep_path = os.path.join(args.output_dir, "preprocessor.joblib")
    joblib.dump({"imputer": imputer, "scaler": scaler}, prep_path)
    print(f"\nSaved model:        {model_path}")
    print(f"Saved preprocessor: {prep_path}")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"metrics_{timestamp}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Training patients: {len(y_train)}\n")
        f.write(f"Test patients:     {len(y_test)}\n")
        f.write(f"Training classes:  {Counter(y_train)}\n")
        f.write(f"Test classes:      {Counter(y_test)}\n\n")
        f.write("Per-patient test predictions:\n")
        for key, p, true in sorted(zip(keys_test, proba, y_test)):
            f.write(f"{key:20s} P(C)={p:.3f} "
                    f"pred={'C' if p >= 0.5 else 'I'} "
                    f"true={'C' if true == 1 else 'I'}\n")
        f.write("\n")
        f.write(classification_report(
            y_test, preds, target_names=["I", "C"],
            digits=3, zero_division=0))
        f.write(f"\nAccuracy:          {accuracy_score(y_test, preds):.3f}\n")
        f.write(f"Balanced accuracy: {balanced_accuracy_score(y_test, preds):.3f}\n")
        if len(np.unique(y_test)) > 1:
            f.write(f"ROC-AUC:           {roc_auc_score(y_test, proba):.3f}\n")
        try:
            f.write(f"F1:                {f1_score(y_test, preds):.3f}\n")
        except ValueError:
            pass
    print(f"Saved metrics:      {metrics_path}")

    # Feature importance
    results, top20, _ = feature_importance_analysis(
        X_train_scaled, X_test_scaled, y_train, y_test,
        feature_names, args.output_dir, args.seed,
    )

    # Plots
    save_plots(y_test, proba, preds, top20, args.output_dir, timestamp)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(description="CPM: Clinical Prediction Model")
    p.add_argument("--landmark-dir", default=LANDMARK_DIR,
                   help="Directory containing per-scan landmark .txt files")
    p.add_argument("--excel", default=EXCEL_FILE,
                   help="Clinical Excel workbook with labels and covariates")
    p.add_argument("--output-dir", default=OUTPUT_DIR,
                   help="Where models, tables, and plots are written")
    p.add_argument("--seed", type=int, default=RANDOM_STATE,
                   help="Random seed for splits, models, SMOTE, and TF")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())