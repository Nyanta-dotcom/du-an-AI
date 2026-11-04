"""Feature engineering for classical baselines.

- raw:  flattened 784 pixels (z-scored)
- hog:  Histogram of Oriented Gradients, captures finger edges/shape and is
        less sensitive to lighting than raw intensities
"""
import numpy as np
from skimage.feature import hog


def raw_pixels(X):
    return X.reshape(len(X), -1)


def hog_features(X, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
    return np.stack([
        hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block, block_norm="L2-Hys", feature_vector=True)
        for img in X
    ]).astype(np.float32)
