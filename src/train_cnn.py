"""WEEK 4 STARTER - CNN in PyTorch (prepared in week 3, not yet benchmarked).

Run:  pip install torch onnx
      python -m src.train_cnn --epochs 15 --augment
Output: models/cnn_best.pt, models/cnn.onnx (for the web app), reports/cnn_metrics.json
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):  # allow running this file directly (e.g. VS Code Run / F5)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src.augment import augment_image
from src.data import load_processed


class SignDataset(Dataset):
    def __init__(self, X, y, augment=False, seed=C.SEED):
        self.X, self.y, self.augment = X, y, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = augment_image(x, self.rng)
        return torch.from_numpy(np.ascontiguousarray(x))[None], int(self.y[i])


class SmallCNN(nn.Module):
    """3 conv blocks (32-64-128) + dense head. ~0.4M params, fast enough for the browser."""

    def __init__(self, n_classes=C.NUM_CLASSES, p_drop=0.4):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))  # 28->14->7->3
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(p_drop), nn.Linear(128 * 3 * 3, 128),
                                  nn.ReLU(), nn.Dropout(p_drop), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); out = []
    for x, _ in loader:
        out.append(model(x.to(device)).argmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (Xtr, ytr), (Xval, yval), (Xte, yte) = load_processed()
    tr = DataLoader(SignDataset(Xtr, ytr, args.augment), batch_size=args.batch, shuffle=True, num_workers=0)
    val = DataLoader(SignDataset(Xval, yval), batch_size=512)
    te = DataLoader(SignDataset(Xte, yte), batch_size=512)

    model = SmallCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    models_dir = C.ROOT / "models"; models_dir.mkdir(exist_ok=True)

    best, wait, history = -1, 0, []
    for ep in range(1, args.epochs + 1):
        model.train(); t = time.perf_counter(); tot = 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
            tot += loss.item() * len(y)
        sched.step()
        vf1 = f1_score(yval, predict(model, val, device), average="macro")
        history.append({"epoch": ep, "train_loss": tot / len(ytr), "val_macro_f1": vf1})
        print(f"epoch {ep:02d} loss {tot / len(ytr):.4f} val_f1 {vf1:.4f} ({time.perf_counter() - t:.0f}s)")
        if vf1 > best:
            best, wait = vf1, 0; torch.save(model.state_dict(), models_dir / "cnn_best.pt")
        else:
            wait += 1
            if wait >= args.patience:
                print("early stopping"); break

    model.load_state_dict(torch.load(models_dir / "cnn_best.pt", map_location=device))
    pte = predict(model, te, device)
    metrics = {"augment": args.augment, "best_val_macro_f1": best,
               "test_accuracy": accuracy_score(yte, pte), "test_macro_f1": f1_score(yte, pte, average="macro"),
               "params": sum(p.numel() for p in model.parameters()), "history": history}
    (C.REPORTS_DIR / f"cnn_metrics{'_aug' if args.augment else ''}.json").write_text(json.dumps(metrics, indent=2))
    print({k: v for k, v in metrics.items() if k != "history"})

    model.cpu().eval()
    torch.onnx.export(model, torch.zeros(1, 1, 28, 28), models_dir / "cnn.onnx", input_names=["image"],
                      output_names=["logits"], dynamic_axes={"image": {0: "batch"}}, opset_version=17)
    print("exported", models_dir / "cnn.onnx")


if __name__ == "__main__":
    main()
