"""Data loading, quality review and preprocessing pipeline.

Run:  python -m src.data
Output: data/processed/sign_mnist.npz and reports/data_quality.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __package__ in (None, ""):  # allow running this file directly (e.g. VS Code Run / F5)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C


# ---------------------------------------------------------------- loading
def load_raw():
    """Return (train_df, test_df) exactly as downloaded from Kaggle."""
    if not C.TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"{C.TRAIN_CSV} not found. Download Sign Language MNIST from "
            "https://www.kaggle.com/datasets/datamunge/sign-language-mnist "
            "and put the two CSV files in data/raw/."
        )
    return pd.read_csv(C.TRAIN_CSV), pd.read_csv(C.TEST_CSV)


# ---------------------------------------------------------- quality review
def quality_report(train_df, test_df):
    """Missing values, duplicates, label coverage, class balance, pixel range."""
    px = [c for c in train_df.columns if c != "label"]
    counts = train_df["label"].value_counts().sort_index()
    report = {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "missing_values": {"train": int(train_df.isna().sum().sum()),
                           "test": int(test_df.isna().sum().sum())},
        "duplicate_rows": {"train": int(train_df.duplicated().sum()),
                           "test": int(test_df.duplicated().sum()),
                           "train_test_overlap": int(len(pd.merge(train_df, test_df, how="inner")))},
        "labels_present": sorted(int(x) for x in train_df["label"].unique()),
        "labels_missing": [9, 25],
        "pixel_range": [int(train_df[px].values.min()), int(train_df[px].values.max())],
        "class_count_train": {C.LETTERS[C.LABEL_TO_INDEX[k]]: int(v) for k, v in counts.items()},
        "class_imbalance_ratio": round(float(counts.max() / counts.min()), 3),
        "out_of_range_labels": int((~train_df["label"].isin(C.ORIGINAL_LABELS)).sum()
                                   + (~test_df["label"].isin(C.ORIGINAL_LABELS)).sum()),
    }
    return report


# ------------------------------------------------------------- transforms
def to_arrays(df):
    """DataFrame -> (X float32 in [0,1] with shape (N,28,28), y int64 in 0..23)."""
    df = df.dropna().drop_duplicates()                     # cleaning (no-op on this dataset, kept for safety)
    df = df[df["label"].isin(C.ORIGINAL_LABELS)]           # drop impossible labels
    y = df["label"].map(C.LABEL_TO_INDEX).to_numpy(np.int64)
    X = df.drop(columns="label").to_numpy(np.float32) / 255.0  # min-max scale to [0,1]
    return X.reshape(-1, C.IMG_SIZE, C.IMG_SIZE), y


def pseudo_groups(X, y, clusters_per_class=C.CLUSTERS_PER_CLASS):
    """Recover approximate 'source photo' groups.

    The Kaggle training set was built by augmenting ~1,704 original photos, so
    it contains many near-copies of each photo. We cluster images of each
    letter (10x10 thumbnails, per-image normalised, Ward linkage) so that the
    copies of one photo land in the same group.
    """
    from skimage.transform import resize
    from sklearn.cluster import AgglomerativeClustering

    small = np.stack([resize(x, (10, 10), anti_aliasing=True) for x in X]).reshape(len(X), -1)
    small = (small - small.mean(1, keepdims=True)) / (small.std(1, keepdims=True) + 1e-6)
    groups = np.zeros(len(y), dtype=np.int64)
    offset = 0
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        k = min(clusters_per_class, len(idx))
        groups[idx] = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(small[idx]) + offset
        offset += k
    return groups


def split(X, y, val_size=C.VAL_SIZE, seed=C.SEED, strategy=C.SPLIT_STRATEGY):
    """Train/validation split of the official training set.

    strategy="group":  per letter, hold out val_size of the near-duplicate
                       groups (stratified by class AND leakage-free).
    strategy="random": stratified random split (kept only for comparison).
    The official test set is never touched during model selection.
    """
    if strategy == "random":
        return train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed)
    groups = pseudo_groups(X, y)
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        g = np.unique(groups[y == c])
        held = rng.choice(g, max(1, int(round(len(g) * val_size))), replace=False)
        is_val |= np.isin(groups, held)
    return X[~is_val], X[is_val], y[~is_val], y[is_val]


def standardize(X_train, *others):
    """Per-pixel z-score using training statistics only (avoids leakage)."""
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    return [(X_train - mean) / std] + [(x - mean) / std for x in others], (mean, std)


def build_dataset(save=True):
    train_df, test_df = load_raw()
    report = quality_report(train_df, test_df)
    X_full, y_full = to_arrays(train_df)
    X_test, y_test = to_arrays(test_df)
    X_tr, X_val, y_tr, y_val = split(X_full, y_full)
    report["split_strategy"] = C.SPLIT_STRATEGY
    report["split_sizes"] = {"train": int(len(y_tr)), "val": int(len(y_val)), "test": int(len(y_test))}

    if save:
        C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(C.PROCESSED_DIR / "sign_mnist.npz",
                            X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val,
                            X_test=X_test, y_test=y_test)
        (C.REPORTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2))
    return (X_tr, y_tr), (X_val, y_val), (X_test, y_test), report


def load_processed():
    path = C.PROCESSED_DIR / "sign_mnist.npz"
    if not path.exists():
        build_dataset()
    d = np.load(path)
    return (d["X_train"], d["y_train"]), (d["X_val"], d["y_val"]), (d["X_test"], d["y_test"])


if __name__ == "__main__":
    *_, rep = build_dataset()
    print(json.dumps({k: v for k, v in rep.items() if k != "class_count_train"}, indent=2))
