"""Shared paths and constants for the Sign Language MNIST project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"



def _find_csv(name):
    """Find a Kaggle CSV directly in data/raw/ or inside a sub-folder (the zip unpacks both ways)."""
    direct = RAW_DIR / name
    if direct.exists():
        return direct
    hits = sorted(RAW_DIR.rglob(name)) if RAW_DIR.exists() else []
    return hits[0] if hits else direct


TRAIN_CSV = _find_csv("sign_mnist_train.csv")
TEST_CSV = _find_csv("sign_mnist_test.csv")

IMG_SIZE = 28
SEED = 42
VAL_SIZE = 0.15  # fraction of the official training set held out for validation
# "group": hold out whole clusters of near-duplicate images (recommended, see reports/week3_report.md)
# "random": plain stratified random split (leaks near-duplicates -> ~100% val accuracy)
SPLIT_STRATEGY = "group"
CLUSTERS_PER_CLASS = 70  # ~1704 original photos / 24 letters

# Original Kaggle labels are 0-25 (A-Z) but 9 (J) and 25 (Z) are missing
# because those letters need motion. We remap the 24 remaining labels to 0..23.
ORIGINAL_LABELS = [i for i in range(26) if i not in (9, 25)]
LETTERS = [chr(ord("A") + i) for i in ORIGINAL_LABELS]
LABEL_TO_INDEX = {orig: idx for idx, orig in enumerate(ORIGINAL_LABELS)}
NUM_CLASSES = len(ORIGINAL_LABELS)  # 24
