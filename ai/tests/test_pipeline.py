#!/usr/bin/env python3
"""Test pipeline: runs both approaches on test images and compares with expected values."""

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, AI_DIR)

from process import process_image, get_sam2_predictor

IMAGE_DIR = os.path.join(SCRIPT_DIR, "images")

EXPECTED = {
    "test_25": 25, "test_50": 50, "test_80": 80,
    "test_pure_10": 10, "test_pure_20": 20, "test_pure_70": 70, "test_pure_80": 80,
}


def find_test_dirs(test_arg: str | None) -> list[str]:
    if test_arg:
        return [os.path.abspath(test_arg)]
    dirs = []
    for name in sorted(os.listdir(IMAGE_DIR)):
        d = os.path.join(IMAGE_DIR, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "original.jpg")):
            dirs.append(d)
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Test water level detection pipeline")
    parser.add_argument("--dir", default=None, help="Path to a single test directory (default: all in images/)")
    args = parser.parse_args()

    test_dirs = find_test_dirs(args.dir)
    if not test_dirs:
        print("No test directories found.")
        sys.exit(1)

    print("Loading SAM2 model...")
    t0 = time.time()
    get_sam2_predictor()
    print(f"SAM2 model loaded in {time.time() - t0:.1f}s\n")

    header = f"{'Test':<15} {'Expected':>8} {'SAM1+CV':>8} {'SAM2':>8} {'Avg':>8}"
    print(header)
    print("-" * len(header))

    for test_dir in test_dirs:
        name = os.path.basename(test_dir)

        t0 = time.time()
        result = process_image(test_dir)
        elapsed = time.time() - t0

        exp = EXPECTED.get(name)
        exp_str = f"{exp}%" if exp is not None else "  ?"
        print(f"{name:<15} {exp_str:>8} {result['sam1_level']:>7.0%} {result['sam2_level']:>7.0%} {result['level']:>7.0%}   ({elapsed:.1f}s)")

    print("\nDone!")


if __name__ == "__main__":
    main()
