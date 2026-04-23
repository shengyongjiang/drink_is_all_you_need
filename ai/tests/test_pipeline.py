#!/usr/bin/env python3
"""Test pipeline: runs both detection approaches on test images."""

import argparse
import glob
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.join(SCRIPT_DIR, "..")
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

IMAGE_DIR = os.path.join(SCRIPT_DIR, "images")
SAM2_DIR = os.path.join(AI_DIR, "sam2_fill_level")
SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints", "sam2_hiera_small.pt")
SAM2_FINETUNED = os.path.join(SAM2_DIR, "checkpoints", "SAM2_For_VesselAndFillLevel", "model_small.torch")
SAM2_CFG = "sam2_hiera_s.yaml"

EXPECTED = {
    "25.JPG": 25, "50.JPG": 50, "80.JPG": 80,
    "pure_10.JPG": 10, "pure_20.JPG": 20, "pure_70.JPG": 70, "pure_80.JPG": 80,
}


def load_image(path: str) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return np.array(img.convert("RGB"))


def save_debug(image_rgb: np.ndarray, path: str):
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def out_path(image_path: str, suffix: str) -> str:
    base = image_path.rsplit(".", 1)[0]
    return f"{base}_{suffix}.jpg"


# ── Approach 1: YOLO + SAM1 + OpenCV ─────────────────────────────────────────

def run_approach1(image_path: str, resized: np.ndarray) -> float:
    detections = detect_cup_yolo(resized)
    save_debug(draw_yolo_detections(resized, detections), out_path(image_path, "yolo"))

    if not detections:
        return 0.0

    best = detections[0]
    mask = segment_cup_sam(resized, best["bbox"])
    save_debug(draw_cup_mask(resized, mask, best["bbox"]), out_path(image_path, "cup"))

    sub_masks = segment_cup_regions(resized, best["bbox"], mask)
    save_debug(draw_regions_overlay(resized, mask, sub_masks), out_path(image_path, "regions"))

    level, water_line_y = detect_water_level(resized, mask, sub_masks)
    save_debug(draw_split_debug(resized, mask, sub_masks, water_line_y), out_path(image_path, "split"))
    save_debug(draw_level_overlay(resized, mask, level, water_line_y, best["bbox"]), out_path(image_path, "level"))

    return level


# ── Approach 2: SAM2 Fine-tuned for Fill Level ───────────────────────────────

def load_sam2_predictor() -> SAM2ImagePredictor:
    sam2_model = build_sam2(SAM2_CFG, SAM2_CHECKPOINT, device="cpu")
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.load_state_dict(torch.load(SAM2_FINETUNED, map_location="cpu"))
    predictor.model.eval()
    return predictor


def run_approach2(predictor: SAM2ImagePredictor, image_path: str, resized: np.ndarray) -> float:
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
    save_debug(overlay, out_path(image_path, "sam2fill"))

    return fill_ratio


# ── Main ─────────────────────────────────────────────────────────────────────

DEBUG_SUFFIXES = {"_norm", "_yolo", "_cup", "_regions", "_split", "_level", "_sam2fill"}


def find_images(image_arg: str | None) -> list[str]:
    if image_arg:
        return [os.path.abspath(image_arg)]
    paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.JPG")) + glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    return [p for p in paths if not any(os.path.basename(p).rsplit(".", 1)[0].endswith(s) for s in DEBUG_SUFFIXES)]


def main():
    parser = argparse.ArgumentParser(description="Test water level detection pipeline")
    parser.add_argument("--image", default=None, help="Path to a single test image (default: all images in test_pipelin_images/)")
    args = parser.parse_args()

    images = find_images(args.image)
    if not images:
        print("No test images found.")
        sys.exit(1)

    print(f"Loading SAM2 model...")
    t0 = time.time()
    sam2_predictor = load_sam2_predictor()
    print(f"SAM2 model loaded in {time.time() - t0:.1f}s\n")

    header = f"{'Image':<15} {'Expected':>8} {'SAM1+CV':>8} {'SAM2':>8}"
    print(header)
    print("-" * len(header))

    for path in images:
        name = os.path.basename(path)
        image = load_image(path)
        resized, _ = resize_image(image)

        t0 = time.time()
        level1 = run_approach1(path, resized)
        t1 = time.time()
        level2 = run_approach2(sam2_predictor, path, resized)
        t2 = time.time()

        exp = EXPECTED.get(name)
        exp_str = f"{exp}%" if exp is not None else "  ?"
        print(f"{name:<15} {exp_str:>8} {level1:>7.0%} {level2:>7.0%}   ({t1-t0:.1f}s + {t2-t1:.1f}s)")

    print("\nDone!")


if __name__ == "__main__":
    main()
