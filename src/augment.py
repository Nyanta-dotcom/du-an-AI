"""On-the-fly image augmentation (used for CNN training in week 4).

Simulates what the webcam will see: small rotations, shifts, zoom and lighting
changes. Horizontal flip is OFF by default because it turns a right-hand sign
into a left-hand sign; enable it only if left-handed users must be supported.
"""
import numpy as np
from scipy import ndimage


def augment_image(img, rng, max_rot=10, max_shift=2, zoom=(0.9, 1.1),
                  brightness=0.15, contrast=0.15, flip=False):
    out = ndimage.rotate(img, rng.uniform(-max_rot, max_rot), reshape=False, mode="nearest")
    z = rng.uniform(*zoom)
    zoomed = ndimage.zoom(out, z, order=1, mode="nearest")
    n = img.shape[0]
    if z >= 1:  # crop centre
        s = (zoomed.shape[0] - n) // 2
        out = zoomed[s:s + n, s:s + n]
    else:       # pad back to n x n
        p = n - zoomed.shape[0]
        out = np.pad(zoomed, ((p // 2, p - p // 2),) * 2, mode="edge")
    out = ndimage.shift(out, rng.integers(-max_shift, max_shift + 1, size=2), mode="nearest")
    out = (out - 0.5) * (1 + rng.uniform(-contrast, contrast)) + 0.5 + rng.uniform(-brightness, brightness)
    if flip and rng.random() < 0.5:
        out = out[:, ::-1]
    return np.clip(out, 0, 1).astype(np.float32)


def augment_batch(X, seed=None, **kw):
    rng = np.random.default_rng(seed)
    return np.stack([augment_image(x, rng, **kw) for x in X])
