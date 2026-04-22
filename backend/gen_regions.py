#!/usr/bin/env python3
"""Generate SAM sub-segmentation regions within the YOLO-detected cup bbox."""

import argparse
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps
from segment_anything import SamAutomaticMaskGenerator

from vision import resize_image, detect_cup_yolo, _get_sam_predictor, SAM_MODEL_TYPE, SAM_CHECKPOINT, DEVICE
from segment_anything import sam_model_registry


def load_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    return np.array(img.convert("RGB"))


def main(image_path: str):
    print(f"Loading: {image_path}")
    image = load_image(image_path)
    resized, scale = resize_image(image)
    print(f"Resized: {resized.shape[1]}x{resized.shape[0]}")

    detections = detect_cup_yolo(resized)
    if not detections:
        print("No cup detected!")
        sys.exit(1)

    best = detections[0]
    x1, y1, x2, y2 = best["bbox"]
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(resized.shape[1], x2 + pad)
    y2 = min(resized.shape[0], y2 + pad)
    cropped = resized[y1:y2, x1:x2]
    print(f"Cup bbox: ({x1},{y1})-({x2},{y2}), cropped: {cropped.shape[1]}x{cropped.shape[0]}")

    print("Running SamAutomaticMaskGenerator on cropped cup...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )
    masks = mask_gen.generate(cropped)
    masks.sort(key=lambda m: m["area"], reverse=True)
    print(f"Found {len(masks)} sub-masks")

    overlay = cropped.copy()
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(len(masks), 3))
    for i, m in enumerate(masks):
        seg = m["segmentation"]
        color = colors[i]
        overlay[seg] = (overlay[seg] * 0.4 + color * 0.6).astype(np.uint8)
        ys_m = np.where(seg)[0]
        if len(ys_m) > 0:
            cy = int(np.mean(ys_m))
            xs_m = np.where(seg)[1]
            cx = int(np.mean(xs_m))
            area_pct = m["area"] / (cropped.shape[0] * cropped.shape[1]) * 100
            cv2.putText(overlay, f"{i}:{area_pct:.1f}%", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    base = image_path.rsplit(".", 1)[0]
    out_path = f"{base}_regions.jpg"
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    main(args.image)