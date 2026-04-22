import cv2
import numpy as np


def _sam_region_detection(cup_mask: np.ndarray, sub_masks: list[dict],
                          scan_top: int, scan_bottom: int) -> tuple[int, float, np.ndarray]:
    """Find water line via SAM sub-mask horizontal boundary detection.

    For each row, counts how many columns have a mask transition from the row
    above. The row with the most transitions = strongest horizontal boundary
    = water surface.

    Returns (best_row_index_within_scan, best_score, score_profile).
    """
    n = scan_bottom - scan_top
    num_masks = len(sub_masks)
    if num_masks < 2:
        return n // 2, 0.0, np.zeros(n)

    h, w = cup_mask.shape
    # Assign each pixel to its smallest covering mask (most specific)
    dom_mask = np.full((h, w), -1, dtype=np.int16)
    for i in range(len(sub_masks) - 1, -1, -1):
        dom_mask[sub_masks[i]["segmentation"]] = i

    score_profile = np.zeros(n, dtype=float)
    for i in range(1, n):
        y = scan_top + i
        row_cup = cup_mask[y, :]
        cup_cols = np.where(row_cup)[0]
        if len(cup_cols) < 5:
            continue
        changes = np.sum(dom_mask[y, cup_cols] != dom_mask[y - 1, cup_cols])
        score_profile[i] = changes / len(cup_cols)

    k = max(3, n // 15)
    if k % 2 == 0:
        k += 1
    smoothed = np.convolve(score_profile, np.ones(k) / k, mode="same")

    inner_margin = n // 10
    inner = smoothed[inner_margin : n - inner_margin]
    if len(inner) == 0:
        return n // 2, 0.0, smoothed
    peak_local = int(np.argmax(inner))
    peak_idx = peak_local + inner_margin
    peak_val = smoothed[peak_idx]

    return peak_idx, peak_val, smoothed


def _edge_based_detection(gray: np.ndarray, cup_mask: np.ndarray,
                           scan_top: int, scan_bottom: int) -> tuple[int, float]:
    """Find water line via horizontal edge detection (Sobel-y).
    Returns (best_row_index_within_scan, peak_score).
    """
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
    n = scan_bottom - scan_top

    edge_profile = np.zeros(n, dtype=float)
    for i, y in enumerate(range(scan_top, scan_bottom)):
        row_mask = cup_mask[y, :]
        mask_px = np.where(row_mask)[0]
        if len(mask_px) < 5:
            continue
        vals = sobel_y[y, row_mask]
        strong = vals[vals > 20]
        if len(strong) < 3:
            continue
        coverage = len(strong) / len(mask_px)
        edge_profile[i] = float(np.mean(strong)) * coverage

    k = max(3, n // 12)
    if k % 2 == 0:
        k += 1
    smoothed = np.convolve(edge_profile, np.ones(k) / k, mode="same")

    inner_margin = n // 8
    inner = smoothed[inner_margin : n - inner_margin]
    if len(inner) == 0:
        return n // 2, 0.0
    peak_local = int(np.argmax(inner))
    peak_idx = peak_local + inner_margin
    peak_val = smoothed[peak_idx]
    median_val = float(np.median(smoothed[smoothed > 0])) if np.any(smoothed > 0) else 1.0
    relative_peak = peak_val / max(median_val, 0.01)

    return peak_idx, relative_peak


def _brightness_split_detection(gray: np.ndarray, cup_mask: np.ndarray,
                                 scan_top: int, scan_bottom: int) -> tuple[int, float]:
    """Find water line via max brightness split difference.
    Returns (best_row_index_within_scan, best_score).
    """
    n = scan_bottom - scan_top
    row_brightness = np.zeros(n, dtype=float)
    for i, y in enumerate(range(scan_top, scan_bottom)):
        row_mask = cup_mask[y, :]
        pixels = gray[y, row_mask]
        if len(pixels) < 5:
            continue
        row_brightness[i] = float(np.mean(pixels))

    k = max(3, n // 20)
    if k % 2 == 0:
        k += 1
    smoothed = np.convolve(row_brightness, np.ones(k) / k, mode="same")

    min_split = n // 10
    best_idx = n // 2
    best_score = 0.0
    cumsum = np.cumsum(smoothed)
    for i in range(min_split, n - min_split):
        top_mean = cumsum[i] / (i + 1)
        bottom_mean = (cumsum[-1] - cumsum[i]) / (n - i - 1)
        score = abs(top_mean - bottom_mean)
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, best_score


def detect_water_level(image: np.ndarray, cup_mask: np.ndarray,
                       sub_masks: list[dict] = None) -> tuple[float, int]:
    """Detect relative water level (0.0~1.0) within the cup mask.

    Uses three strategies (priority order):
    1. SAM region split: mask coverage change (best for transparent water)
    2. Edge-based: horizontal edge detection (fallback)
    3. Brightness split: brightness change (fallback for colored liquids)

    Returns (level, water_line_y).
    """
    ys = np.where(cup_mask)[0]
    cup_top = int(ys.min())
    cup_bottom = int(ys.max())
    cup_height = cup_bottom - cup_top
    if cup_height < 20:
        return 0.0, cup_bottom

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    margin = int(cup_height * 0.10)
    scan_top = cup_top + margin
    scan_bottom = cup_bottom - margin
    n = scan_bottom - scan_top
    if n < 10:
        return 0.0, cup_bottom

    sam_idx, sam_score = 0, 0.0
    if sub_masks and len(sub_masks) >= 2:
        sam_idx, sam_score, _ = _sam_region_detection(cup_mask, sub_masks, scan_top, scan_bottom)
        print(f"  SAM regions: idx={sam_idx} score={sam_score:.3f}")

    edge_idx, edge_score = _edge_based_detection(gray, cup_mask, scan_top, scan_bottom)
    bright_idx, bright_score = _brightness_split_detection(gray, cup_mask, scan_top, scan_bottom)

    print(f"  Edge: idx={edge_idx} relative_peak={edge_score:.2f}")
    print(f"  Brightness: idx={bright_idx} split_score={bright_score:.1f}")

    if sam_score > 0.05:
        water_idx = sam_idx
        print(f"  -> Using SAM region split")
    elif edge_score > 2.0:
        water_idx = edge_idx
        print(f"  -> Using edge detection (strong peak)")
    elif bright_score > 15 and edge_score < 1.5:
        water_idx = bright_idx
        print(f"  -> Using brightness split")
    elif edge_score > 1.5:
        water_idx = edge_idx
        print(f"  -> Using edge detection (moderate peak)")
    elif bright_score > 10:
        water_idx = bright_idx
        print(f"  -> Using brightness split (weak)")
    else:
        return 0.0, cup_bottom

    water_line_y = scan_top + water_idx
    level = (cup_bottom - water_line_y) / cup_height
    level = round(min(max(level, 0.0), 1.0), 2)

    return level, water_line_y


def draw_split_debug(image: np.ndarray, cup_mask: np.ndarray,
                     sub_masks: list[dict], water_line_y: int) -> np.ndarray:
    """Debug image: SAM sub-masks colored + split score profile chart."""
    ys = np.where(cup_mask)[0]
    cup_top = int(ys.min())
    cup_bottom = int(ys.max())
    cup_height = cup_bottom - cup_top
    margin = int(cup_height * 0.10)
    scan_top = cup_top + margin
    scan_bottom = cup_bottom - margin
    n = scan_bottom - scan_top

    _, _, score_profile = _sam_region_detection(cup_mask, sub_masks, scan_top, scan_bottom)

    overlay = image.copy()
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(len(sub_masks), 3))
    for i, m in enumerate(sub_masks):
        seg = m["segmentation"]
        overlay[seg] = (overlay[seg] * 0.4 + colors[i] * 0.6).astype(np.uint8)

    h, w = overlay.shape[:2]
    lw = max(2, min(h, w) // 300)
    fs = max(0.4, min(h, w) / 1200)

    mask_u8 = cup_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), lw)

    xs = np.where(cup_mask)[1]
    cup_left = int(xs.min())
    cup_right = int(xs.max())

    cv2.line(overlay, (cup_left, scan_top), (cup_right, scan_top), (255, 255, 0), lw)
    cv2.line(overlay, (cup_left, scan_bottom), (cup_right, scan_bottom), (255, 255, 0), lw)

    wl_row = min(water_line_y, cup_mask.shape[0] - 1)
    water_xs = np.where(cup_mask[wl_row, :])[0]
    if len(water_xs) > 0:
        xl, xr = int(water_xs.min()), int(water_xs.max())
    else:
        xl, xr = cup_left, cup_right
    cv2.line(overlay, (xl, water_line_y), (xr, water_line_y), (0, 255, 255), lw * 3)
    cv2.putText(overlay, f"water line", (xr + 5, water_line_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), lw)

    chart_w = 200
    chart = np.zeros((h, chart_w, 3), dtype=np.uint8)
    chart[:] = (30, 30, 30)

    if n > 0 and score_profile.max() > 0:
        max_val = score_profile.max()
        for i in range(n):
            y = scan_top + i
            bar_len = int(score_profile[i] / max_val * (chart_w - 20))
            color = (0, 255, 255) if y == water_line_y else (0, 200, 0)
            cv2.line(chart, (10, y), (10 + bar_len, y), color, 1)

    cv2.putText(chart, "Split Score", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(chart, "(mask coverage)", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 180, 0), 1)
    cv2.line(chart, (10, scan_top), (chart_w - 10, scan_top), (255, 255, 0), 1)
    cv2.line(chart, (10, scan_bottom), (chart_w - 10, scan_bottom), (255, 255, 0), 1)
    if scan_top <= water_line_y <= scan_bottom:
        cv2.line(chart, (10, water_line_y), (chart_w - 10, water_line_y), (0, 255, 255), 2)

    return np.hstack([overlay, chart])


def draw_level_overlay(
    image: np.ndarray,
    cup_mask: np.ndarray,
    water_level: float,
    water_line_y: int,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    overlay = image.copy()
    h, w = overlay.shape[:2]
    lw = max(2, min(h, w) // 300)
    fs = max(0.4, min(h, w) / 1200)

    # Cup contour (purple)
    mask_u8 = cup_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (180, 0, 255), lw)

    ys = np.where(cup_mask)[0]
    xs = np.where(cup_mask)[1]
    cup_top = int(ys.min())
    cup_bottom = int(ys.max())
    cup_height = cup_bottom - cup_top
    cup_left = int(xs.min())
    cup_right = int(xs.max())

    # Blue tint on water region
    water_region = cup_mask.copy()
    water_region[:water_line_y, :] = False
    overlay[water_region] = (overlay[water_region] * 0.5 + np.array([50, 120, 255]) * 0.5).astype(np.uint8)

    # Cup top / bottom lines (green)
    cv2.line(overlay, (cup_left, cup_top), (cup_right, cup_top), (0, 255, 0), lw)
    cv2.line(overlay, (cup_left, cup_bottom), (cup_right, cup_bottom), (0, 255, 0), lw)
    cv2.putText(overlay, "1.0", (cup_right + 8, cup_top + 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), lw)
    cv2.putText(overlay, "0.0", (cup_right + 8, cup_bottom + 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), lw)

    # Water line (yellow, thick)
    wl_row = min(water_line_y, cup_mask.shape[0] - 1)
    water_xs = np.where(cup_mask[wl_row, :])[0]
    if len(water_xs) > 0:
        x_left, x_right = int(water_xs.min()), int(water_xs.max())
    else:
        x_left, x_right = cup_left, cup_right
    cv2.line(overlay, (x_left, water_line_y), (x_right, water_line_y), (0, 255, 255), lw * 3)

    # Water level label
    label = f"{water_level:.2f} ({water_level:.0%})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.5, lw)
    lx = cup_right + 8
    ly = water_line_y + th // 2
    cv2.rectangle(overlay, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), (0, 0, 0), -1)
    cv2.putText(overlay, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.5, (0, 255, 255), lw)

    # Ruler on left
    rx = max(0, cup_left - 25)
    cv2.line(overlay, (rx, cup_top), (rx, cup_bottom), (255, 255, 255), lw)
    for tick in [0.25, 0.5, 0.75]:
        ty = int(cup_bottom - tick * cup_height)
        cv2.line(overlay, (rx - 8, ty), (rx + 8, ty), (255, 255, 255), lw)
        cv2.putText(overlay, f"{tick:.0%}", (rx - 45, ty + 4), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, (255, 255, 255), lw)

    # YOLO bbox (green, thin)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), lw)

    return overlay
