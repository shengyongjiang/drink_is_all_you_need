# Testing the AI Vision Pipeline

## Setup

```bash
cd ai
```

## Usage

```bash
# Run all test images
uv run python tests/test_pipeline.py

# Run a single test directory
uv run python tests/test_pipeline.py --dir tests/images/test_25
```

Output is a comparison table:

```
Test            Expected  SAM1+CV     SAM2      Avg
----------------------------------------------------
test_25              25%     30%     30%     30%   (35.5s)
test_50              50%     48%     45%     46%   (35.8s)
...
```

## Two Detection Approaches

### Approach 1: YOLO + SAM1 + OpenCV (Multi-strategy)

Pipeline: YOLO11n cup detection → SAM ViT-B segmentation → water level estimation via 3 strategies (SAM region boundaries, Sobel-y edges, brightness split).

Debug outputs (saved in each test directory):
- `yolo.jpg` — YOLO bounding box overlay
- `cup.jpg` — SAM cup mask overlay
- `regions.jpg` — SAM sub-region segmentation (colored zones)
- `split.jpg` — split score profile chart
- `level.jpg` — final result: water line, ruler, fill percentage

### Approach 2: SAM2 Fine-tuned for Fill Level

Uses SAM2 (Hiera-Small) fine-tuned on vessel/fill-level datasets. Outputs 3 semantic masks: vessel, filled, transparent. Fill ratio = filled_pixels / vessel_pixels.

Debug output:
- `sam2fill.jpg` — green=vessel, blue=filled, yellow=transparent

## Test Images

Each test case is a directory under `tests/images/` containing `original.jpg`. Debug images are saved alongside it.

### Colored Liquid (tea in transparent cup)

| Directory | Fill Level | Description |
|-----------|-----------|-------------|
| `test_25` | ~25% | Low fill, Lipton tea bag visible |
| `test_50` | ~50% | Half full |
| `test_80` | ~80% | Nearly full, Lipton tea bag visible |

### Transparent Water (clear water in transparent cup)

| Directory | Fill Level | Description |
|-----------|-----------|-------------|
| `test_pure_10` | ~10% | Very low fill, water barely visible |
| `test_pure_20` | ~20% | Low fill |
| `test_pure_70` | ~70% | High fill |
| `test_pure_80` | ~80% | Nearly full |

## Accuracy Comparison

| Test | Actual | SAM1+CV | SAM2 |
|------|--------|---------|------|
| test_25 (colored) | 25% | 30% | 30% |
| test_50 (colored) | 50% | 48% | 45% |
| test_80 (colored) | 80% | 71% | 68% |
| test_pure_10 (transparent) | 10% | 25% | 22% |
| test_pure_20 (transparent) | 20% | 28% | 32% |
| test_pure_70 (transparent) | 70% | 54% | 62% |
| test_pure_80 (transparent) | 80% | 75% | 76% |

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
