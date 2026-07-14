"""
compare_single.py
------------------
Quick CLI to compare a single reference image against a single distorted
(e.g. compressed) image and print PSNR / SSIM / MSE.

Usage:
    python compare_single.py --reference original.png --distorted compressed.jpg
"""

import argparse
import sys

import cv2

from metrics import evaluate_pair


def main():
    parser = argparse.ArgumentParser(
        description="Compare two images using PSNR and SSIM."
    )
    parser.add_argument(
        "--reference", required=True, help="Path to reference (original) image"
    )
    parser.add_argument(
        "--distorted", required=True, help="Path to distorted (compressed) image"
    )
    args = parser.parse_args()

    ref_img = cv2.imread(args.reference, cv2.IMREAD_COLOR)
    dist_img = cv2.imread(args.distorted, cv2.IMREAD_COLOR)

    if ref_img is None:
        sys.exit(f"Could not read reference image: {args.reference}")
    if dist_img is None:
        sys.exit(f"Could not read distorted image: {args.distorted}")

    if ref_img.shape != dist_img.shape:
        print(
            f"Note: image sizes differ ({ref_img.shape[:2]} vs {dist_img.shape[:2]}); "
            f"resizing distorted image to match reference for comparison."
        )

    metrics = evaluate_pair(ref_img, dist_img)

    print(f"\nReference: {args.reference}")
    print(f"Distorted: {args.distorted}")
    print("-" * 40)
    print(f"MSE:  {metrics['mse']:.4f}")
    psnr_str = (
        "inf (identical images)"
        if metrics["psnr"] == float("inf")
        else f"{metrics['psnr']:.2f} dB"
    )
    print(f"PSNR: {psnr_str}")
    print(f"SSIM: {metrics['ssim']:.4f}")


if __name__ == "__main__":
    main()
