#!/usr/bin/env python3
"""AI processor: detects water level from captured images."""

import argparse
import json
import os
import sys
import time
from datetime import datetime

AI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(AI_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "captures")

sys.path.insert(0, AI_DIR)
sys.path.insert(0, os.path.join(AI_DIR, "sam2_fill_level"))

import cv2
import numpy as np
import torch
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
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM2_DIR = os.path.join(AI_DIR, "sam2_fill_level")
SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints", "sam2_hiera_small.pt")
SAM2_FINETUNED = os.path.join(SAM2_DIR, "checkpoints", "SAM2_For_VesselAndFillLevel", "model_small.torch")
SAM2_CFG = "sam2_hiera_s.yaml"

_sam2_predictor = None


def get_sam2_predictor() -> SAM2ImagePredictor:
    global _sam2_predictor
    if _sam2_predictor is None:
        sam2_model = build_sam2(SAM2_CFG, SAM2_CHECKPOINT, device="cpu")
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.model.load_state_dict(torch.load(SAM2_FINETUNED, map_location="cpu"))
        predictor.model.eval()
        _sam2_predictor = predictor
    return _sam2_predictor


def load_image(path: str) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return np.array(img.convert("RGB"))


def save_debug(image_rgb: np.ndarray, path: str):
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def run_approach1(capture_dir: str, resized: np.ndarray) -> tuple[float, dict]:
    detections = detect_cup_yolo(resized)
    save_debug(draw_yolo_detections(resized, detections), os.path.join(capture_dir, "yolo.jpg"))

    if not detections:
        return 0.0, {}

    best = detections[0]
    mask = segment_cup_sam(resized, best["bbox"])
    save_debug(draw_cup_mask(resized, mask, best["bbox"]), os.path.join(capture_dir, "cup.jpg"))

    sub_masks = segment_cup_regions(resized, best["bbox"], mask)
    save_debug(draw_regions_overlay(resized, mask, sub_masks), os.path.join(capture_dir, "regions.jpg"))

    level, water_line_y = detect_water_level(resized, mask, sub_masks)
    save_debug(draw_split_debug(resized, mask, sub_masks, water_line_y), os.path.join(capture_dir, "split.jpg"))
    save_debug(draw_level_overlay(resized, mask, level, water_line_y, best["bbox"]), os.path.join(capture_dir, "level.jpg"))

    return level, {
        "cup_class": best["class_name"],
        "cup_confidence": round(best["confidence"], 3),
        "cup_bbox": [int(x) for x in best["bbox"]],
    }


def run_approach2(capture_dir: str, resized: np.ndarray) -> float:
    predictor = get_sam2_predictor()
    predictor.set_image_batch([resized])

    input_label = np.ones([1, 1])
    _, unnorm_coords, labels, _ = predictor._prep_prompts(
        np.ones([1, 1, 2]), input_label, box=None, mask_logits=None, normalize_coords=True
    )
    sparse_emb, dense_emb = predictor.model.sam_prompt_encoder(
        points=(unnorm_coords, labels), boxes=None, masks=None
    )
    high_res_features = [f[-1].unsqueeze(0) for f in predictor._features["high_res_feats"]]
    low_res_masks, _, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=True, repeat_image=False, high_res_features=high_res_features,
    )
    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
    prd_mask = (torch.sigmoid(prd_masks).cpu().detach().numpy() > 0.5)[0]

    vessel_mask, filled_mask, transparent_mask = prd_mask[0], prd_mask[1], prd_mask[2]
    vessel_pixels = vessel_mask.sum()
    fill_ratio = float(filled_mask.sum() / vessel_pixels) if vessel_pixels > 0 else 0.0

    overlay = resized.copy()
    overlay[vessel_mask] = (overlay[vessel_mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    overlay[filled_mask] = (overlay[filled_mask] * 0.4 + np.array([50, 120, 255]) * 0.6).astype(np.uint8)
    overlay[transparent_mask] = (overlay[transparent_mask] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)
    h, w = overlay.shape[:2]
    fs = max(0.5, min(h, w) / 800)
    lw = max(1, min(h, w) // 400)
    cv2.putText(overlay, f"Fill: {fill_ratio:.0%}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.5, (0, 255, 255), lw + 1)
    save_debug(overlay, os.path.join(capture_dir, "sam2fill.jpg"))

    return fill_ratio


def process_image(capture_dir: str) -> dict:
    original_path = os.path.join(capture_dir, "original.jpg")
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"No original.jpg in {capture_dir}")

    timestamp = os.path.basename(capture_dir)
    image = load_image(original_path)
    resized, _ = resize_image(image)

    sam1_level, cup_info = run_approach1(capture_dir, resized)
    sam2_level = run_approach2(capture_dir, resized)

    result = {
        "timestamp": timestamp,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "cup_detected": bool(cup_info),
        **cup_info,
        "sam1_level": round(sam1_level, 3),
        "sam2_level": round(sam2_level, 3),
        "level": round((sam1_level + sam2_level) / 2, 3),
    }

    with open(os.path.join(capture_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


def find_unprocessed() -> list[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    dirs = []
    for name in sorted(os.listdir(DATA_DIR)):
        d = os.path.join(DATA_DIR, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "original.jpg")) and not os.path.exists(os.path.join(d, "result.json")):
            dirs.append(d)
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Process captured images for water level detection")
    parser.add_argument("capture_dir", nargs="?", help="Path to a capture directory")
    parser.add_argument("--all", action="store_true", help="Process all unprocessed captures")
    args = parser.parse_args()

    if args.all:
        dirs = find_unprocessed()
        if not dirs:
            print("No unprocessed captures found.")
            return
        print(f"Found {len(dirs)} unprocessed capture(s)")
    elif args.capture_dir:
        dirs = [os.path.abspath(args.capture_dir)]
    else:
        parser.print_help()
        sys.exit(1)

    print("Loading SAM2 model...")
    t0 = time.time()
    get_sam2_predictor()
    print(f"SAM2 model loaded in {time.time() - t0:.1f}s\n")

    for d in dirs:
        name = os.path.basename(d)
        print(f"Processing {name}...", end=" ", flush=True)
        t0 = time.time()
        result = process_image(d)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s) — level: {result['level']:.0%}")

    print("\nDone!")


if __name__ == "__main__":
    main()
