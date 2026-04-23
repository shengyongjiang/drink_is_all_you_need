import os

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from ultralytics import YOLO

MAX_SIDE = 1024

SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT = os.path.join(os.path.dirname(__file__), "models", "sam_vit_b.pth")
DEVICE = "cpu"

# COCO class IDs: 41=cup, 39=bottle, 40=wine glass
CUP_CLASS_IDS = {39, 40, 41}

_yolo_model = None
_sam_model = None
_sam_predictor = None
_sam_mask_generator = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolo11n.pt")
    return _yolo_model


def _get_sam_model():
    global _sam_model
    if _sam_model is None:
        _sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        _sam_model.to(device=DEVICE)
    return _sam_model


def _get_sam_predictor():
    global _sam_predictor
    if _sam_predictor is None:
        _sam_predictor = SamPredictor(_get_sam_model())
    return _sam_predictor


def _get_sam_mask_generator():
    global _sam_mask_generator
    if _sam_mask_generator is None:
        _sam_mask_generator = SamAutomaticMaskGenerator(
            _get_sam_model(),
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100,
        )
    return _sam_mask_generator


def resize_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_SIDE:
        return image, 1.0
    scale = MAX_SIDE / long_side
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def detect_cup_yolo(image: np.ndarray) -> list[dict]:
    model = _get_yolo()
    results = model(image, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in CUP_CLASS_IDS:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(box.conf[0]),
                    "class_id": cls_id,
                    "class_name": model.names[cls_id],
                })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def segment_cup_sam(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    predictor = _get_sam_predictor()
    predictor.set_image(image)

    box_array = np.array(bbox)
    masks, scores, _ = predictor.predict(
        box=box_array,
        multimask_output=True,
    )
    best_idx = np.argmax(scores)
    return masks[best_idx]


def draw_yolo_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    overlay = image.copy()
    line_w = max(2, min(image.shape[:2]) // 300)
    font_scale = max(0.5, min(image.shape[:2]) / 1000)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), line_w)
        cv2.putText(overlay, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), line_w)

    return overlay


def draw_cup_mask(image: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    overlay = image.copy()
    line_w = max(2, min(image.shape[:2]) // 300)

    overlay[mask] = (overlay[mask] * 0.5 + np.array([180, 0, 255]) * 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (180, 0, 255), line_w * 2)

    x1, y1, x2, y2 = bbox
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), line_w)

    return overlay


def segment_cup_regions(image: np.ndarray, bbox: tuple[int, int, int, int],
                        cup_mask: np.ndarray) -> list[dict]:
    x1, y1, x2, y2 = bbox
    pad = 10
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(image.shape[1], x2 + pad)
    cy2 = min(image.shape[0], y2 + pad)
    cropped = image[cy1:cy2, cx1:cx2]

    mask_gen = _get_sam_mask_generator()
    raw_masks = mask_gen.generate(cropped)

    regions = []
    for m in raw_masks:
        seg_full = np.zeros(image.shape[:2], dtype=bool)
        seg_full[cy1:cy2, cx1:cx2] = m["segmentation"]
        seg_full = seg_full & cup_mask
        area = int(seg_full.sum())
        if area < 100:
            continue
        regions.append({"segmentation": seg_full, "area": area})

    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions


def draw_regions_overlay(image: np.ndarray, cup_mask: np.ndarray,
                         regions: list[dict]) -> np.ndarray:
    overlay = image.copy()
    lw = max(2, min(image.shape[:2]) // 300)
    fs = max(0.4, min(image.shape[:2]) / 1200)

    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(len(regions), 3))
    cup_area = int(cup_mask.sum())

    for i, r in enumerate(regions):
        seg = r["segmentation"]
        overlay[seg] = (overlay[seg] * 0.4 + colors[i] * 0.6).astype(np.uint8)
        ys_r = np.where(seg)[0]
        if len(ys_r) > 0:
            cy = int(np.mean(ys_r))
            cx = int(np.mean(np.where(seg)[1]))
            pct = r["area"] / cup_area * 100 if cup_area > 0 else 0
            cv2.putText(overlay, f"{i}:{pct:.1f}%", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (255, 255, 255), lw)

    mask_u8 = cup_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), lw)

    return overlay
