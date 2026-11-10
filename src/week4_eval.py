"""Week 4 evaluation: baseline vs CNN, error analysis, figures.

Run after src.train_cnn_numpy:  python -m src.week4_eval
Output: reports/week4/*.png, reports/week4/week4_summary.json
"""
import json

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.data import load_processed
from src.features import hog_features

OUT = C.REPORTS_DIR / "week4"
BLUE, ORANGE, GREY = "#2E75B6", "#E07A2F", "#9AA3AF"


def main():
    (Xtr, ytr), (Xv, yv), (Xte, yte) = load_processed()
    runs = json.loads((OUT / "cnn_runs.json").read_text())
    best = max(runs, key=lambda r: r["best_val_macro_f1"])           # chosen on VALIDATION only
    P = np.load(OUT / f"test_proba_{best['name']}.npy"); pc = P.argmax(1)

    # week-3 best baseline (MLP + HOG), re-trained with the same settings to get its test predictions
    mlp = make_pipeline(StandardScaler(), MLPClassifier((256, 128), alpha=1e-3, early_stopping=True, max_iter=60, random_state=C.SEED))
    mlp.fit(hog_features(Xtr), ytr); pb = mlp.predict(hog_features(Xte))

    base = json.loads((C.REPORTS_DIR / "baseline_metrics.json").read_text())["results"]
    rows = [{"model": r["model"], "val_f1": r["val_macro_f1"], "test_acc": r["test_accuracy"], "test_f1": r["test_macro_f1"]} for r in base[1:]]
    names = {"noaug_lr1e-3_do0.3": "CNN (không augment)", "aug_lr1e-3_do0.3": "CNN + augment", "aug_lr3e-3_do0.3": "CNN + augment, lr 3e-3", "aug_lr1e-3_do0.5": "CNN + augment, dropout 0.5"}
    for r in runs:
        rows.append({"model": names[r["name"]], "val_f1": round(r["best_val_macro_f1"], 4), "test_acc": round(r["test_acc"], 4), "test_f1": round(r["test_macro_f1"], 4)})

    # 1. comparison chart (test macro-F1)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(rows))
    cols = [ORANGE if r["model"] == names[best["name"]] else (BLUE if r["model"].startswith("CNN") else GREY) for r in rows]
    ax.barh(y, [r["test_f1"] for r in rows], color=cols)
    for i, r in enumerate(rows): ax.text(r["test_f1"] + .005, i, f'{r["test_f1"]:.3f}', va="center", fontsize=9)
    ax.set_yticks(y, [r["model"] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0.6, 1.02)
    ax.axvline(0.899, ls="--", color="#555", lw=1); ax.text(0.9, len(rows) - .4, "mốc tuần 3 (0,899)", fontsize=8, color="#555")
    ax.set_xlabel("macro-F1 trên tập test"); ax.set_title("Baseline (xám) và CNN (xanh); cam = CNN chọn theo validation")
    fig.tight_layout(); fig.savefig(OUT / "comparison.png", dpi=150); plt.close(fig)

    # 2. training curves
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for r in runs:
        e = [h["epoch"] for h in r["history"]]
        axs[0].plot(e, [h["train_loss"] for h in r["history"]], marker="o", ms=3, label=names[r["name"]])
        axs[1].plot(e, [h["val_macro_f1"] for h in r["history"]], marker="o", ms=3, label=names[r["name"]])
    axs[0].set_title("Loss huấn luyện"); axs[1].set_title("Macro-F1 trên validation (chia theo nhóm)")
    for a in axs: a.set_xlabel("epoch"); a.grid(alpha=.3)
    axs[1].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=150); plt.close(fig)

    # 3. per-letter F1, CNN vs baseline
    fc = f1_score(yte, pc, average=None); fb = f1_score(yte, pb, average=None)
    fig, ax = plt.subplots(figsize=(10, 3.4)); x = np.arange(24)
    ax.bar(x - .2, fb, .4, color=GREY, label="MLP + HOG (tuần 3)"); ax.bar(x + .2, fc, .4, color=BLUE, label=names[best["name"]])
    ax.set_xticks(x, C.LETTERS); ax.set_ylim(0, 1.18); ax.set_ylabel("F1 trên test"); ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.set_title("F1 từng chữ cái"); fig.tight_layout(); fig.savefig(OUT / "per_letter_f1.png", dpi=150); plt.close(fig)

    # 4. confusion matrix
    cm = confusion_matrix(yte, pc, normalize="true"); cmb = confusion_matrix(yte, pb, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7.2)); im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(24), C.LETTERS); ax.set_yticks(range(24), C.LETTERS); ax.set_xlabel("dự đoán"); ax.set_ylabel("thật")
    ax.set_title(f"Confusion matrix – {names[best['name']]} (test)"); fig.colorbar(im, fraction=.046)
    fig.tight_layout(); fig.savefig(OUT / "confusion_cnn.png", dpi=150); plt.close(fig)
    off = cm.copy(); np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:8], off.shape))[0]
    pairs_cnn = [{"true": C.LETTERS[i], "pred": C.LETTERS[j], "cnn": round(float(cm[i, j]), 3), "baseline": round(float(cmb[i, j]), 3)} for i, j in top]
    old = [("N", "S"), ("V", "W"), ("U", "K"), ("R", "S"), ("S", "E"), ("U", "R")]
    L = C.LETTERS
    pairs_week3 = [{"true": a, "pred": b, "baseline": round(float(cmb[L.index(a), L.index(b)]), 3), "cnn": round(float(cm[L.index(a), L.index(b)]), 3)} for a, b in old]

    # 5. misclassified gallery
    wrong = np.where(pc != yte)[0]
    rng = np.random.default_rng(0); pick = rng.choice(wrong, min(16, len(wrong)), replace=False)
    fig, axs = plt.subplots(2, 8, figsize=(11, 3.4))
    for a, i in zip(axs.flat, pick):
        a.imshow(Xte[i], cmap="gray"); a.set_title(f"{L[yte[i]]}→{L[pc[i]]} ({P[i, pc[i]]:.0%})", fontsize=9); a.axis("off")
    fig.suptitle("Ảnh test CNN đoán sai (thật → đoán, độ tin cậy)"); fig.tight_layout(); fig.savefig(OUT / "misclassified.png", dpi=150); plt.close(fig)

    # 6. landmark MLP (web app) results, from reports/landmark/metrics.json
    lm_path = C.LANDMARK_METRICS
    lm = json.loads(lm_path.read_text(encoding="utf-8")) if lm_path.exists() else None

    summary = {"chosen": best["name"], "chosen_label": names[best["name"]], "rows": rows,
               "runs": [{k: v for k, v in r.items() if k != "history"} | {"epochs_run": len(r["history"]), "sec_per_epoch": round(float(np.mean([h["sec"] for h in r["history"]])), 1)} for r in runs],
               "baseline_retrained_test_f1": round(float(f1_score(yte, pb, average="macro")), 4),
               "per_letter_f1_cnn": {l: round(float(v), 3) for l, v in zip(L, fc)},
               "per_letter_f1_base": {l: round(float(v), 3) for l, v in zip(L, fb)},
               "top_pairs_cnn": pairs_cnn, "week3_pairs": pairs_week3, "n_wrong": int(len(wrong)), "landmark": lm}
    (OUT / "week4_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_letter_f1_cnn", "per_letter_f1_base", "landmark")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
