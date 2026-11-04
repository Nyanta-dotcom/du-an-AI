"""Week 4-5: MLP on MediaPipe hand landmarks for the web app.

Data : keypoint.csv (MIT, Akram Adrane et al., github.com/AkramOM606/American-Sign-Language-Detection)
       + optional CSV files exported from the web app's "Dữ liệu" page (same format).
       Row = label(0-25, A-Z), then 21 x (x, y) relative to the wrist, divided by max |value|,
       measured on a horizontally mirrored camera image. J (9) and Z (25) are dropped (need motion).
Run  : python train_landmark_mlp.py [extra.csv ...]
Out  : model_bundle.json (weights + reference poses + samples) and metrics.json
"""
import sys, json, base64, csv
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
KEEP = [c for c in range(26) if c not in (9, 25)]
LETTERS = [chr(65 + c) for c in KEEP]
REMAP = {o: i for i, o in enumerate(KEEP)}
rng = np.random.default_rng(42)

def load(paths):
    X, y = [], []
    for p in paths:
        for r in csv.reader(open(p, encoding='utf-8-sig')):
            if len(r) != 43: continue
            lab = int(float(r[0]))
            if lab in REMAP:
                X.append([float(v) for v in r[1:]]); y.append(REMAP[lab])
    return np.array(X, np.float32), np.array(y)

def renorm(X): return (X / (np.abs(X).max(1, keepdims=True) + 1e-9)).astype(np.float32)
def mirror(X): M = X.copy(); M[:, 0::2] *= -1; return M
def rotate(X, deg):
    P = X.reshape(-1, 21, 2); a = np.radians(rng.uniform(-deg, deg, len(P)))[:, None]
    c, s = np.cos(a), np.sin(a)
    return renorm(np.stack([P[..., 0]*c - P[..., 1]*s, P[..., 0]*s + P[..., 1]*c], -1).reshape(-1, 42))
def augment(X, y):
    B = np.concatenate([X, mirror(X)]); yb = np.concatenate([y, y])
    return np.concatenate([B, rotate(B, 15)]), np.concatenate([yb, yb])
def block_split(y, frac=.8):
    tr = np.zeros(len(y), bool)
    for c in range(24):
        idx = np.where(y == c)[0]; tr[idx[:int(len(idx)*frac)]] = True
    return tr
def mlp(): return MLPClassifier((128, 64), alpha=1e-3, max_iter=150, early_stopping=True, random_state=0)

paths = [HERE / 'keypoint.csv'] + [Path(p) for p in sys.argv[1:]]
X, y = load(paths); print('samples', X.shape)

# honest evaluation: hold out the LAST 20% of each letter (later recording sessions)
tr = block_split(y)
m = mlp().fit(*augment(X[tr], y[tr])); p = m.predict(X[~tr])
cm = confusion_matrix(y[~tr], p, normalize='true'); off = cm.copy(); np.fill_diagonal(off, 0)
top = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:8], off.shape))[0]
Xa, Xb, ya, yb = train_test_split(X, y, test_size=.2, stratify=y, random_state=0)
rand_acc = accuracy_score(yb, mlp().fit(Xa, ya).predict(Xb))
metrics = {'samples': int(len(y)), 'block_split_accuracy': round(float(accuracy_score(y[~tr], p)), 4),
           'block_split_macro_f1': round(float(f1_score(y[~tr], p, average='macro')), 4),
           'random_split_accuracy_no_aug': round(float(rand_acc), 4),
           'per_letter_f1': {L: round(float(f), 3) for L, f in zip(LETTERS, f1_score(y[~tr], p, average=None))},
           'top_confusions': [{'true': LETTERS[i], 'pred': LETTERS[j], 'rate': round(float(off[i, j]), 3)} for i, j in top]}
print(json.dumps({k: v for k, v in metrics.items() if k != 'per_letter_f1'}, indent=1))

# final model on all data
final = mlp().fit(*augment(X, y))
q8 = lambda A: base64.b64encode(np.clip(np.round(A * 127), -127, 127).astype(np.int8).tobytes()).decode()
ref, demo, base = [], [], []
for c in range(24):
    idx = np.where(y == c)[0]
    mean = X[idx].mean(0); med = idx[np.argmin(np.linalg.norm(X[idx] - mean, axis=1))]
    ref.append([round(float(v), 4) for v in X[med]])
    held = np.where((y == c) & ~tr)[0]; demo.append(q8(X[rng.choice(held, 20, replace=False)]))
    base.append(q8(X[rng.choice(idx, 120, replace=False)]))
bundle = {'letters': LETTERS, 'metrics': metrics,
          'layers': [{'W': np.round(W, 5).tolist(), 'b': np.round(b, 5).tolist()} for W, b in zip(final.coefs_, final.intercepts_)],
          'ref': ref, 'demo': demo, 'base': base}
json.dump(bundle, open(HERE / 'model_bundle.json', 'w'), separators=(',', ':'))
json.dump(metrics, open(HERE / 'metrics.json', 'w'), indent=2, ensure_ascii=False)
np.save(HERE / 'check_in.npy', X[:40]); np.save(HERE / 'check_out.npy', final.predict_proba(X[:40]))
print('bundle KB', (HERE / 'model_bundle.json').stat().st_size // 1024)
