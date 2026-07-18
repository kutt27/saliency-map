import cv2
import numpy as np


def extract_intensity(img_rgb):
    return np.mean(img_rgb, axis=2)


def compute_opponent_channels(img_rgb):
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    r = np.clip(R - (G + B) / 2, 0, None)
    g = np.clip(G - (R + B) / 2, 0, None)
    b = np.clip(B - (R + G) / 2, 0, None)
    y = np.clip((R + G) / 2 - np.abs(R - G) / 2 - B, 0, None)

    return r, g, b, y


def compute_color_opponency(r, g, b, y):
    RG = cv2.absdiff(r, g)
    BY = cv2.absdiff(b, y)
    return RG, BY
