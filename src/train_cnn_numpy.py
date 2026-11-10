"""Week 4: train the NumPy CNN on the group split, tune on validation, report test.

Run:  python -m src.train_cnn_numpy            (all 4 configs, ~30 min on 2 CPU cores)
      python -m src.train_cnn_numpy --only aug_lr1e-3_do0.3
Output: models/cnn_numpy_<name>.npz, reports/week4/cnn_runs.json
"""
import argparse, json, time

if __package__ in (None, ""):  # allow running this file directly (e.g. VS Code Run / F5)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src import config as C
from src.augment import augment_image
from src.cnn_numpy import CNN, Adam, softmax_ce
from src.data import load_processed

CONFIGS = {
    "noaug_lr1e-3_do0.3": dict(augment=False, lr=1e-3, dropout=0.3),
    "aug_lr1e-3_do0.3": dict(augment=True, lr=1e-3, dropout=0.3),
    "aug_lr3e-3_do0.3": dict(augment=True, lr=3e-3, dropout=0.3),
    "aug_lr1e-3_do0.5": dict(augment=True, lr=1e-3, dropout=0.5),
}
EPOCHS, PATIENCE, BATCH, SMOOTH, WD = 15, 3, 128, 0.05, 1e-4


def augment_epoch(X, rng, p=0.7):
    out = X.copy()
    for i in np.where(rng.random(len(X)) < p)[0]:
        out[i] = augment_image(X[i], rng)
    return out


def run(name, cfg, data, log):
    (Xtr, ytr), (Xv, yv), _ = data
    rng = np.random.default_rng(C.SEED)
    net = CNN(dropout=cfg["dropout"], seed=C.SEED)
    opt = Adam(net.p, lr=cfg["lr"], wd=WD)
    best, best_state, wait, hist = -1, None, 0, []
    for ep in range(1, EPOCHS + 1):
        t = time.time()
        lr = cfg["lr"] * 0.5 * (1 + np.cos(np.pi * (ep - 1) / EPOCHS))  # cosine schedule
        opt.lr = lr
        X = augment_epoch(Xtr, rng) if cfg["augment"] else Xtr
        idx = rng.permutation(len(X)); tot = 0.0
        for s in range(0, len(X), BATCH):
            b = idx[s:s + BATCH]
            loss, d = softmax_ce(net.forward(X[b], train=True), ytr[b], SMOOTH)
            opt.step(net.p, net.backward(d)); tot += loss * len(b)
        pv = net.predict_proba(Xv).argmax(1)
        vf1 = f1_score(yv, pv, average="macro"); vacc = accuracy_score(yv, pv)
        hist.append({"epoch": ep, "lr": lr, "train_loss": tot / len(X), "val_acc": vacc, "val_macro_f1": vf1, "sec": time.time() - t})
        log(f"[{name}] ep {ep:2d} loss {tot / len(X):.4f} val_acc {vacc:.4f} val_f1 {vf1:.4f} ({time.time() - t:.0f}s)")
        if vf1 > best:
            best, best_state, wait = vf1, net.state(), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                log(f"[{name}] early stopping"); break
    net.load(best_state)
    return net, {"name": name, **cfg, "best_val_macro_f1": best, "best_epoch": int(np.argmax([h["val_macro_f1"] for h in hist]) + 1),
                 "params": net.n_params(), "history": hist}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only"); a = ap.parse_args()
    data = load_processed()
    out_dir = C.REPORTS_DIR / "week4"; out_dir.mkdir(parents=True, exist_ok=True)
    (C.ROOT / "models").mkdir(exist_ok=True)
    log = lambda s: print(s, flush=True)
    runs = []
    for name, cfg in CONFIGS.items():
        if a.only and name != a.only: continue
        net, info = run(name, cfg, data, log)
        np.savez_compressed(C.ROOT / "models" / f"cnn_numpy_{name}.npz", **net.p)
        Xte, yte = data[2]
        t = time.perf_counter(); P = net.predict_proba(Xte); ms = (time.perf_counter() - t) / len(Xte) * 1000
        pt = P.argmax(1)
        info.update(test_acc=accuracy_score(yte, pt), test_macro_f1=f1_score(yte, pt, average="macro"), ms_per_image=ms)
        np.save(out_dir / f"test_proba_{name}.npy", P.astype(np.float32))
        runs.append(info)
        (out_dir / "cnn_runs.json").write_text(json.dumps(runs, indent=2, default=float))
        log(f"[{name}] best val F1 {info['best_val_macro_f1']:.4f} | test acc {info['test_acc']:.4f} F1 {info['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
