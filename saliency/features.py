import cv2
import numpy as np


CENTER_SCALES = [2, 3, 4]
DELTA_SCALES = [3, 4]


def center_surround_diff(pyramid, c, s):
    center = pyramid[c]
    surround = cv2.resize(
        pyramid[s],
        (center.shape[1], center.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.absdiff(center, surround)


def normalize_map(feature_map):
    fmap = feature_map.copy()
    fmap = cv2.normalize(fmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    mean_val = np.mean(fmap)
    max_val = np.max(fmap)
    return fmap * ((max_val - mean_val) ** 2)


def generate_feature_maps(pyramid):
    maps = []
    for c in CENTER_SCALES:
        for delta in DELTA_SCALES:
            s = c + delta
            fmap = center_surround_diff(pyramid, c, s)
            maps.append(normalize_map(fmap))
    return maps
