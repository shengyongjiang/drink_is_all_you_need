# Testing the AI Vision Pipeline

## Setup

```bash
cd backend
```

## Usage

```bash
# Run all 7 test images (both approaches)
uv run python tests/test_pipeline.py

# Run a single image
uv run python tests/test_pipeline.py --image test_pipelin_images/25.JPG
```

Both approaches run on each image. Output is a comparison table:

```
Image           Expected  SAM1+CV     SAM2
------------------------------------------
25.JPG               25%     30%     30%   (34.4s + 1.1s)
50.JPG               50%     48%     45%   (34.2s + 1.4s)
...
```

## Two Detection Approaches

### Approach 1: YOLO + SAM1 + OpenCV (Multi-strategy)

Pipeline: YOLO11n cup detection → SAM ViT-B segmentation → water level estimation via 3 strategies (SAM region boundaries, Sobel-y edges, brightness split).

Debug outputs (saved next to input image):
- `_yolo.jpg` — YOLO bounding box overlay
- `_cup.jpg` — SAM cup mask overlay
- `_regions.jpg` — SAM sub-region segmentation (colored zones)
- `_split.jpg` — split score profile chart
- `_level.jpg` — final result: water line, ruler, fill percentage

### Approach 2: SAM2 Fine-tuned for Fill Level

Uses SAM2 (Hiera-Small) fine-tuned on vessel/fill-level datasets. Outputs 3 semantic masks: vessel, filled, transparent. Fill ratio = filled_pixels / vessel_pixels.

Debug output:
- `_sam2fill.jpg` — green=vessel, blue=filled, yellow=transparent

## Test Images

All test images are in `test_pipelin_images/`. Two categories:

### Colored Liquid (tea in transparent cup)

| File | Fill Level | Description |
|------|-----------|-------------|
| `25.JPG` | ~25% | Low fill, Lipton tea bag visible |
| `50.JPG` | ~50% | Half full |
| `80.JPG` | ~80% | Nearly full, Lipton tea bag visible |

### Transparent Water (clear water in transparent cup)

| File | Fill Level | Description |
|------|-----------|-------------|
| `pure_10.JPG` | ~10% | Very low fill, water barely visible |
| `pure_20.JPG` | ~20% | Low fill |
| `pure_70.JPG` | ~70% | High fill |
| `pure_80.JPG` | ~80% | Nearly full |

## Accuracy Comparison

| Image | Actual | SAM1+CV | SAM2 |
|-------|--------|---------|------|
| 25.JPG (colored) | 25% | 30% | 30% |
| 50.JPG (colored) | 50% | 48% | 45% |
| 80.JPG (colored) | 80% | 71% | 68% |
| pure_10 (transparent) | 10% | 25% | 22% |
| pure_20 (transparent) | 20% | 28% | 32% |
| pure_70 (transparent) | 70% | 54% | 62% |
| pure_80 (transparent) | 80% | 75% | 76% |

### Observations

- **SAM1 + OpenCV**: Better for colored liquids at mid-range fills (50%, 80%). Uses SAM sub-region boundaries when available, falls back to Sobel-y edge detection.
- **SAM2 Fill Level**: Better for transparent water (pure_70: 54%→62%, pure_80: 75%→76%). Directly segments vessel/filled regions without edge heuristics. Tends to underestimate high fills because vessel mask sometimes extends beyond the cup.
- **Common weakness**: Both overestimate low fill levels (pure_10: actual 10%, both predict ~22-25%).

## Photo Requirements

- Cup or bottle with water, photographed from the **side** (not top-down)
- Camera should see the full height of the cup
- Water level should be visible (transparent or semi-transparent cup)
- Good lighting, minimal background clutter helps
- No reference card needed — outputs relative water level (0.0~1.0)

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No cup/bottle detected` | Reduce background objects, ensure cup is the largest object |
| Water level = 0.0 | Use a transparent/translucent cup so water line is visible |
| Very slow (>30s) | Normal for first run on CPU, subsequent runs reuse loaded model |
| SAM2 import error | Ensure `sam2_fill_level/` is cloned and checkpoints downloaded |
