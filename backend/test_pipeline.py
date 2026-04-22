#!/usr/bin/env python3
"""Test pipeline: YOLO cup detection -> SAM segmentation -> water level detection."""

import argparse
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from vision import (
    resize_image,
    detect_cup_yolo,
    segment_cup_sam,
    segment_cup_regions,
    draw_yolo_detections,
    draw_cup_mask,
    draw_regions_overlay,
)
from volume_estimator import detect_water_level, draw_split_debug, draw_level_overlay


def load_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    return np.array(img.convert("RGB"))


def save_debug(image_rgb: np.ndarray, path: str):
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
    print(f"  -> saved: {path}")


def get_output_path(image_path: str, suffix: str) -> str:
    base = image_path.rsplit(".", 1)[0]
    return f"{base}_{suffix}.jpg"


def run_pipeline(image_path: str):
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    print(f"Original size: {image.shape[1]}x{image.shape[0]}")

    resized, scale = resize_image(image)
    print(f"Resized: {resized.shape[1]}x{resized.shape[0]} (scale={scale:.3f})")
    save_debug(resized, get_output_path(image_path, "norm"))

    # Step 1: YOLO detection
    print("\n[Step 1] YOLO cup detection...")
    t0 = time.time()
    detections = detect_cup_yolo(resized)
    t1 = time.time()
    print(f"Found {len(detections)} cup/bottle detections in {t1 - t0:.1f}s")
    for i, det in enumerate(detections):
        print(f"  [{i}] {det['class_name']} conf={det['confidence']:.2f} bbox={det['bbox']}")

    yolo_overlay = draw_yolo_detections(resized, detections)
    save_debug(yolo_overlay, get_output_path(image_path, "yolo"))

    if not detections:
        print("ERROR: No cup/bottle detected by YOLO!")
        sys.exit(1)

    best = detections[0]
    print(f"\nUsing: {best['class_name']} (conf={best['confidence']:.2f})")

    # Step 2: SAM segmentation within YOLO bbox
    print("\n[Step 2] SAM segmentation within bbox...")
    t0 = time.time()
    mask = segment_cup_sam(resized, best["bbox"])
    t1 = time.time()
    print(f"SAM segmentation done in {t1 - t0:.1f}s")
    print(f"Mask pixels: {mask.sum()} / {mask.size} ({mask.sum()/mask.size:.1%})")

    cup_overlay = draw_cup_mask(resized, mask, best["bbox"])
    save_debug(cup_overlay, get_output_path(image_path, "cup"))

    # Step 3: SAM sub-segmentation within cup
    print("\n[Step 3] SAM cup region sub-segmentation...")
    t0 = time.time()
    sub_masks = segment_cup_regions(resized, best["bbox"], mask)
    t1 = time.time()
    print(f"Found {len(sub_masks)} sub-masks in {t1 - t0:.1f}s")

    regions_overlay = draw_regions_overlay(resized, mask, sub_masks)
    save_debug(regions_overlay, get_output_path(image_path, "regions"))

    # Step 4: Water level detection
    print("\n[Step 4] Water level detection...")
    level, water_line_y = detect_water_level(resized, mask, sub_masks)

    print(f"\n{'='*30}")
    print(f"  Relative water level: {level}")
    print(f"  Water fills {level:.0%} of the cup")
    print(f"  Water line at y={water_line_y}")
    print(f"{'='*30}")

    split_debug = draw_split_debug(resized, mask, sub_masks, water_line_y)
    save_debug(split_debug, get_output_path(image_path, "split"))

    level_overlay = draw_level_overlay(resized, mask, level, water_line_y, best["bbox"])
    save_debug(level_overlay, get_output_path(image_path, "level"))

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLO + SAM + water level pipeline")
    parser.add_argument("--image", required=True, help="Path to test image")
    args = parser.parse_args()
    run_pipeline(args.image)
