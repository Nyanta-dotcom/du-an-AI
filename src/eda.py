"""Figures for data understanding & preprocessing. Run: python -m src.eda"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):  # allow running this file directly (e.g. VS Code Run / F5)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src.augment import augment_batch
from src.data import load_processed


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    (Xtr, ytr), (Xval, yval), (Xte, yte) = load_processed()

    # 1. class distribution per split
    fig, ax = plt.subplots(figsize=(11, 3.6))
    w = 0.28
    idx = np.arange(C.NUM_CLASSES)
    for i, (y, name) in enumerate([(ytr, "train"), (yval, "val"), (yte, "test")]):
        ax.bar(idx + (i - 1) * w, np.bincount(y, minlength=C.NUM_CLASSES), w, label=name)
    ax.set_xticks(idx, C.LETTERS)
    ax.set_ylabel("images")
    ax.set_title("Class distribution per split (J and Z absent)")
    ax.set_ylim(0, 1350)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "class_distribution.png", dpi=150); plt.close(fig)

    # 2. one sample per letter
    fig, axes = plt.subplots(3, 8, figsize=(10, 4.2))
    for k, ax in enumerate(axes.flat):
        ax.imshow(Xtr[np.where(ytr == k)[0][0]], cmap="gray"); ax.set_title(C.LETTERS[k]); ax.axis("off")
    fig.suptitle("One training sample per letter (28x28 grayscale)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "samples_per_letter.png", dpi=150); plt.close(fig)

    # 3. mean image per letter
    fig, axes = plt.subplots(3, 8, figsize=(10, 4.2))
    for k, ax in enumerate(axes.flat):
        ax.imshow(Xtr[ytr == k].mean(0), cmap="magma"); ax.set_title(C.LETTERS[k]); ax.axis("off")
    fig.suptitle("Mean image per letter")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "mean_per_letter.png", dpi=150); plt.close(fig)

    # 4. augmentation preview
    src = Xtr[:6]
    aug = [augment_batch(src, seed=s) for s in range(4)]
    fig, axes = plt.subplots(6, 5, figsize=(6, 7.5))
    for r in range(6):
        axes[r, 0].imshow(src[r], cmap="gray", vmin=0, vmax=1)
        for c in range(4):
            axes[r, c + 1].imshow(aug[c][r], cmap="gray", vmin=0, vmax=1)
    for a in axes.flat: a.axis("off")
    axes[0, 0].set_title("original"); [axes[0, c].set_title(f"aug {c}") for c in range(1, 5)]
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "augmentation_preview.png", dpi=150); plt.close(fig)

    # 5. pixel intensity distribution train vs test
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(Xtr.mean(axis=(1, 2)), bins=50, alpha=.6, density=True, label="train")
    ax.hist(Xte.mean(axis=(1, 2)), bins=50, alpha=.6, density=True, label="test")
    ax.set_xlabel("mean brightness per image"); ax.set_title("Brightness distribution"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "brightness_train_vs_test.png", dpi=150); plt.close(fig)
    print("figures saved to", C.FIG_DIR)


if __name__ == "__main__":
    main()
