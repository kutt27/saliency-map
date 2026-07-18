import cv2
import numpy as np


def build_conspicuity_map(feature_maps, base_shape=None):
    if base_shape is None:
        base_shape = feature_maps[0].shape

    conspicuity = np.zeros(base_shape, dtype=np.float32)
    for fmap in feature_maps:
        resized = cv2.resize(fmap, (base_shape[1], base_shape[0]))
        conspicuity += resized

    return cv2.normalize(conspicuity, None, 0, 1, cv2.NORM_MINMAX)
