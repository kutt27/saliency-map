# Saliency Map

Itti-Koch saliency model implementation using intensity, color opponency, and orientation channels with center-surround differences across scales.

For indepth overview, refer to this blog: [Blog Link](https://www.amals.xyz/blog/how-your-brain-decides-what-to-look-at)

## Usage

```bash
pip install -r requirements.txt
python main.py
```

Place an `image.jpg` in the project root, or change the path in `main.py`.

## Modules

| Module | Purpose |
|---|---|
| `saliency.pyramid` | Gaussian pyramid at multiple scales |
| `saliency.channels` | Intensity extraction, RG/BY opponent channels |
| `saliency.gabor` | Gabor filters at 0°, 45°, 90°, 135° |
| `saliency.features` | Center-surround difference, feature map normalization |
| `saliency.conspicuity` | Across-scale fusion into conspicuity maps |
| `saliency.pipeline` | `compute_saliency()`, `create_heatmap()`, `create_overlay()` |

## API

```python
from saliency import compute_saliency, create_heatmap, create_overlay

saliency = compute_saliency(img_rgb)           # -> (H, W) float32 [0, 1]
heatmap  = create_heatmap(saliency, img_shape) # -> (H, W, 3) uint8
overlay  = create_overlay(img_rgb, saliency)    # -> (H, W, 3) uint8
```
