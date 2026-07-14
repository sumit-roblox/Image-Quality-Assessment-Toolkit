"""
batch_compare.py
-----------------
Automates comparison of compressed/distorted images against reference images
across a directory, and generates a CSV + text summary report plus an
optional bar-chart visualization of PSNR/SSIM per image.

Matching strategy: a reference image `foo.png` in --reference-dir is matched
to a distorted image with the SAME FILENAME in --distorted-dir. This covers
the common workflow of compressing a folder of images and comparing each
output back to its source.
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import cv2
import numpy as np

from metrics import evaluate_pair

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def find_image_files(directory: str) -> list:
    files = []
    for name in sorted(os.listdir(directory)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(name)
    return files


def compare_directories(reference_dir: str, distorted_dir: str) -> list:
    """
    Compare every image in reference_dir against the same-named image in
    distorted_dir. Returns a list of result dicts (one per matched pair).
    Files that exist only in one directory are reported as skipped.
    """
    ref_files = find_image_files(reference_dir)
    dist_files = set(find_image_files(distorted_dir))

    results = []
    for filename in ref_files:
        if filename not in dist_files:
            print(f"  [skip] {filename}: no matching file in distorted dir")
            continue

        ref_path = os.path.join(reference_dir, filename)
        dist_path = os.path.join(distorted_dir, filename)

        ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        dist_img = cv2.imread(dist_path, cv2.IMREAD_COLOR)

        if ref_img is None or dist_img is None:
            print(f"  [skip] {filename}: failed to read image")
            continue

        metrics = evaluate_pair(ref_img, dist_img)
        metrics["filename"] = filename
        metrics["ref_size_kb"] = os.path.getsize(ref_path) / 1024
        metrics["dist_size_kb"] = os.path.getsize(dist_path) / 1024
        metrics["compression_ratio"] = (
            metrics["ref_size_kb"] / metrics["dist_size_kb"]
            if metrics["dist_size_kb"] > 0
            else float("inf")
        )
        results.append(metrics)
        print(
            f"  [ok]   {filename}: PSNR={metrics['psnr']:.2f} dB, SSIM={metrics['ssim']:.4f}"
        )

    return results


def write_csv_report(results: list, output_path: str) -> None:
    fieldnames = [
        "filename",
        "mse",
        "psnr",
        "ssim",
        "ref_size_kb",
        "dist_size_kb",
        "compression_ratio",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})


def write_summary_report(results: list, output_path: str) -> None:
    psnr_values = [r["psnr"] for r in results if r["psnr"] != float("inf")]
    ssim_values = [r["ssim"] for r in results]
    ratio_values = [
        r["compression_ratio"]
        for r in results
        if r["compression_ratio"] != float("inf")
    ]

    lines = []
    lines.append("Image Quality Assessment Report")
    lines.append("=" * 40)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Images compared: {len(results)}")
    lines.append("")

    if psnr_values:
        lines.append(
            f"PSNR (dB)  -  mean: {np.mean(psnr_values):.2f}  "
            f"min: {np.min(psnr_values):.2f}  max: {np.max(psnr_values):.2f}  "
            f"std: {np.std(psnr_values):.2f}"
        )
    if ssim_values:
        lines.append(
            f"SSIM       -  mean: {np.mean(ssim_values):.4f}  "
            f"min: {np.min(ssim_values):.4f}  max: {np.max(ssim_values):.4f}  "
            f"std: {np.std(ssim_values):.4f}"
        )
    if ratio_values:
        lines.append(
            f"Compression ratio (ref size / dist size) - mean: {np.mean(ratio_values):.2f}x"
        )

    lines.append("")
    lines.append("Per-image results:")
    lines.append("-" * 40)
    for r in sorted(results, key=lambda x: x["psnr"]):
        psnr_str = "inf" if r["psnr"] == float("inf") else f"{r['psnr']:.2f}"
        lines.append(
            f"  {r['filename']:<30} PSNR={psnr_str:>8} dB   SSIM={r['ssim']:.4f}"
        )

    report_text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report_text)
    print("\n" + report_text)


def generate_chart(results: list, output_path: str) -> None:
    """Bar chart of PSNR and SSIM per image (matplotlib, saved as PNG)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available, skipping chart generation")
        return

    names = [r["filename"] for r in results]
    psnr_vals = [min(r["psnr"], 100) for r in results]  # cap inf for display
    ssim_vals = [r["ssim"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(names) * 0.6), 8))

    ax1.bar(names, psnr_vals, color="#4C72B0")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("PSNR per Image")
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(names, ssim_vals, color="#55A868")
    ax2.set_ylabel("SSIM")
    ax2.set_ylim(0, 1)
    ax2.set_title("SSIM per Image")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Batch-compare reference and compressed images using PSNR/SSIM."
    )
    parser.add_argument(
        "--reference-dir", required=True, help="Directory of original/reference images"
    )
    parser.add_argument(
        "--distorted-dir",
        required=True,
        help="Directory of compressed/distorted images",
    )
    parser.add_argument(
        "--output-dir", default="./iqa_report", help="Directory to write reports into"
    )
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    args = parser.parse_args()

    if not os.path.isdir(args.reference_dir):
        sys.exit(f"Reference directory not found: {args.reference_dir}")
    if not os.path.isdir(args.distorted_dir):
        sys.exit(f"Distorted directory not found: {args.distorted_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(
        f"Comparing images:\n  reference: {args.reference_dir}\n  distorted: {args.distorted_dir}\n"
    )
    results = compare_directories(args.reference_dir, args.distorted_dir)

    if not results:
        sys.exit("No matched image pairs found. Nothing to report.")

    csv_path = os.path.join(args.output_dir, "iqa_results.csv")
    summary_path = os.path.join(args.output_dir, "iqa_summary.txt")
    chart_path = os.path.join(args.output_dir, "iqa_chart.png")

    write_csv_report(results, csv_path)
    write_summary_report(results, summary_path)
    if not args.no_chart:
        generate_chart(results, chart_path)

    print(f"\nReports written to: {args.output_dir}")


if __name__ == "__main__":
    main()
