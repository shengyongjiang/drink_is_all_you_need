# Testing the AI Vision Pipeline

## Setup

Make sure you're in the `backend/` directory:

```bash
cd backend
```

## How to Test

```bash
uv run test_pipeline.py --image path/to/your_photo.jpg
```

### What the Photo Should Look Like

- A **cup or bottle with water** in it
- Photo from the **side** (not top-down) — camera should see the full height of the cup
- Water level should be visible (transparent or semi-transparent cup works best)
- Good lighting, minimal background clutter helps
- **No reference card needed** — the algorithm outputs relative water level (0.0~1.0)

### Example

```bash
uv run test_pipeline.py --image ~/Desktop/mug_test.jpg
```

## Expected Output

```
Loading image: ~/Desktop/mug_test.jpg
Image size: 1920x1080
Running SAM segmentation...
Found 12 masks in 8.5s
Detecting cup...
Detecting water level...

=== Results ===
Relative water level: 0.76
(Cup height = 1.0, water fills 76% of the cup)

Overlay saved to: ~/Desktop/mug_test_result.jpg
```

The `_result.jpg` overlay shows:
- **Orange overlay** = detected cup outline
- **Blue overlay** = detected water region
- **Yellow line** = water surface line
- **White text** = relative water level percentage

## How to Calculate Actual Volume Later

```
actual_water_height = cup_height_mm × relative_level
actual_volume_ml = cup_total_capacity_ml × relative_level
```

## Custom Output Path

```bash
uv run test_pipeline.py --image photo.jpg --output result.jpg
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No cup/bottle detected` | Reduce background objects, ensure cup is the largest object |
| Water level = 0.0 | Use a transparent/translucent cup so water line is visible |
| Very slow (>30s) | Normal for first run on CPU, subsequent runs reuse loaded model |
