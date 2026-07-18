import cv2
import numpy as np

from .pyramid import build_gaussian_pyramid
from .channels import extract_intensity, compute_opponent_channels, compute_color_opponency
from .gabor import apply_gabor_filter, ORIENTATIONS
from .features import generate_feature_maps, normalize_map
from .conspicuity import build_conspicuity_map


def compute_saliency(img_rgb):
    intensity = extract_intensity(img_rgb)
    intensity_pyramid = build_gaussian_pyramid(intensity)
    intensity_maps = generate_feature_maps(intensity_pyramid)
    intensity_conspicuity = build_conspicuity_map(intensity_maps)

    r, g, b, y = compute_opponent_channels(img_rgb)
    RG, BY = compute_color_opponency(r, g, b, y)

    RG_pyramid = build_gaussian_pyramid(RG)
    BY_pyramid = build_gaussian_pyramid(BY)
    color_maps = generate_feature_maps(RG_pyramid) + generate_feature_maps(BY_pyramid)
    color_conspicuity = build_conspicuity_map(color_maps)

    orientation_pyramids = [
        build_gaussian_pyramid(apply_gabor_filter(intensity, theta))
        for theta in ORIENTATIONS
    ]
    orientation_maps = []
    for pyramid in orientation_pyramids:
        orientation_maps.extend(generate_feature_maps(pyramid))
    orientation_conspicuity = build_conspicuity_map(orientation_maps)

    I_bar = normalize_map(intensity_conspicuity)
    C_bar = normalize_map(color_conspicuity)
    O_bar = normalize_map(orientation_conspicuity)

    saliency = (I_bar + C_bar + O_bar) / 3.0
    return cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)


def create_heatmap(saliency_map, img_shape):
    resized = cv2.resize(saliency_map, (img_shape[1], img_shape[0]))
    saliency_uint8 = np.uint8(resized * 255)
    heatmap = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def create_overlay(img_rgb, saliency_map, alpha=0.6):
    heatmap = create_heatmap(saliency_map, img_rgb.shape)
    return cv2.addWeighted(np.uint8(img_rgb * 255), alpha, heatmap, 1 - alpha, 0)
