import cv2
import numpy as np


ORIENTATIONS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def apply_gabor_filter(image, theta, ksize=9, sigma=4.0, lambd=10.0, gamma=0.5):
    kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F
    )
    return cv2.filter2D(image, cv2.CV_32F, kernel)
