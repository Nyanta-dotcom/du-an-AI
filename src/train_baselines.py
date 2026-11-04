"""Train and evaluate week-3 baseline models.

Run:  python -m src.train_baselines
Output: reports/baseline_metrics.json, reports/baseline_metrics.md, figures
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

if __package__ in (None, ""):  # allow running this file directly (e.g. VS Code Run / F5)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src.data import load_processed
from src.features import hog_features, raw_pixels


def models():
    s = C.SEED
    return {
        "Majority class (floor)": ("raw", DummyClassifier(strategy="most_frequent")),
        "Logistic Regression (pixels)": ("raw", make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1))),
        "Random Forest (pixels)": ("raw", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=s)),
        "MLP 256-128 (pixels)": ("raw", make_pipeline(StandardScaler(), MLPClassifier((256, 128), alpha=1e-3, early_stopping=True,
                                                                                    max_iter=60, random_state=s))),
        "Linear SVM (HOG)": ("hog", make_pipeline(StandardScaler(), LinearSVC(C=0.01, max_iter=5000, random_state=s))),
        "MLP 256-128 (HOG)": ("hog", make_pipeline(StandardScaler(), MLPClassifier((256, 128), alpha=1e-3, early_stopping=True,
                                                                                  max_iter=60, random_state=s))),
    }


def evaluate(model, X, y):
    t = time.perf_counter()
    pred = model.predict(X)
    ms = (time.perf_counter() - t) / len(X) * 1000
    return pred, {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"), "ms_per_image": ms}


def plot_confusion(y, pred, title, path):
    cm = confusion_matrix(y, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(C.NUM_CLASSES), C.LETTERS); ax.set_yticks(range(C.NUM_CLASSES), C.LETTERS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return cm


def top_confusions(cm, k=8):
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:k], off.shape))[0]
    return [{"true": C.LETTERS[i], "predicted": C.LETTERS[j], "rate": round(float(off[i, j]), 3)} for i, j in pairs]


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    (Xtr, ytr), (Xval, yval), (Xte, yte) = load_processed()
    t = time.perf_counter()
    feats = {
        "raw": tuple(raw_pixels(x) for x in (Xtr, Xval, Xte)),
        "hog": tuple(hog_features(x) for x in (Xtr, Xval, Xte)),
    }
    print(f"features ready ({time.perf_counter() - t:.1f}s): raw={feats['raw'][0].shape[1]}, hog={feats['hog'][0].shape[1]}")

    results, preds = [], {}
    for name, (fkey, model) in models().items():
        ftr, fval, fte = feats[fkey]
        t = time.perf_counter(); model.fit(ftr, ytr); fit_s = time.perf_counter() - t
        _, val = evaluate(model, fval, yval)
        pte, test = evaluate(model, fte, yte)
        preds[name] = pte
        row = {"model": name, "features": fkey, "train_seconds": round(fit_s, 1),
               "val_accuracy": round(val["accuracy"], 4), "val_macro_f1": round(val["macro_f1"], 4),
               "test_accuracy": round(test["accuracy"], 4), "test_macro_f1": round(test["macro_f1"], 4),
               "ms_per_image": round(test["ms_per_image"], 4)}
        row["val_test_gap"] = round(row["val_accuracy"] - row["test_accuracy"], 4)
        results.append(row); print(row)

    # model selection uses VALIDATION score only; test is reported, not optimised
    best = max(results[1:], key=lambda r: r["val_macro_f1"])
    best_test = max(results[1:], key=lambda r: r["test_macro_f1"])
    cm = plot_confusion(yte, preds[best_test["model"]], f"{best_test['model']} - test set (row-normalised)",
                        C.FIG_DIR / "confusion_best_baseline.png")
    per_class_f1 = f1_score(yte, preds[best_test["model"]], average=None)

    # comparison chart
    fig, ax = plt.subplots(figsize=(9, 3.8))
    names = [r["model"] for r in results]; y = np.arange(len(names))
    ax.barh(y - 0.2, [r["val_accuracy"] for r in results], 0.4, label=f"validation ({C.SPLIT_STRATEGY} split)")
    ax.barh(y + 0.2, [r["test_accuracy"] for r in results], 0.4, label="official test")
    ax.set_yticks(y, names); ax.invert_yaxis(); ax.set_xlim(0, 1.05); ax.set_xlabel("accuracy")
    ax.set_title(f"Baseline models: validation ({C.SPLIT_STRATEGY} split) vs test accuracy"); ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "baseline_comparison.png", dpi=150); plt.close(fig)

    # per-class F1 chart
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.bar(C.LETTERS, per_class_f1); ax.set_ylim(0, 1.05); ax.set_ylabel("F1")
    ax.set_title(f"Per-letter F1 on test - {best_test['model']}")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "per_class_f1.png", dpi=150); plt.close(fig)

    out = {"results": results, "best_by_validation": best["model"], "best_by_test": best_test["model"],
           "top_confusions_test": top_confusions(cm),
           "per_class_f1_test": {l: round(float(f), 3) for l, f in zip(C.LETTERS, per_class_f1)}}
    (C.REPORTS_DIR / "baseline_metrics.json").write_text(json.dumps(out, indent=2))

    md = ["| Model | Features | Val acc | Val macro-F1 | Test acc | Test macro-F1 | Val-test gap | Train (s) | ms/img |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['model']} | {r['features']} | {r['val_accuracy']:.3f} | {r['val_macro_f1']:.3f} | {r['test_accuracy']:.3f} | "
           f"{r['test_macro_f1']:.3f} | {r['val_test_gap']:+.3f} | {r['train_seconds']} | {r['ms_per_image']:.3f} |" for r in results]
    (C.REPORTS_DIR / "baseline_metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md)); print(json.dumps(out["top_confusions_test"], indent=1))


if __name__ == "__main__":
    main()
