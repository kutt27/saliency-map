import cv2
import numpy as np
import matplotlib.pyplot as plt

from saliency import compute_saliency, create_overlay

img_bgr = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

saliency = compute_saliency(img_rgb)
overlay = create_overlay(img_rgb, saliency)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(saliency, cmap="hot")
axes[1].set_title("Saliency Map")
axes[1].axis("off")

axes[2].imshow(overlay)
axes[2].set_title("Attention Heatmap Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.show()
