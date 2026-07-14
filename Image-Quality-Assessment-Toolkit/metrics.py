"""
metrics.py
----------
Core image quality metrics: PSNR and SSIM, implemented with OpenCV + NumPy only
(no scikit-image dependency).

Both functions accept images as NumPy arrays (as loaded by cv2.imread), either
grayscale (H, W) or color (H, W, 3), and images of the same shape. Pixel values
are assumed to be in the 0-255 uint8 range (standard for cv2.imread).
"""

import cv2
import numpy as np


def _to_grayscale_if_color(img: np.ndarray) -> np.ndarray:
    """Convert a color image to grayscale; pass grayscale through unchanged."""
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def compute_psnr(
    reference: np.ndarray, distorted: np.ndarray, max_pixel_value: float = 255.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR = 10 * log10(MAX^2 / MSE)

    Returns:
        PSNR in dB. Returns float('inf') if the images are identical (MSE == 0).
    """
    if reference.shape != distorted.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs distorted {distorted.shape}"
        )

    reference = reference.astype(np.float64)
    distorted = distorted.astype(np.float64)

    mse = np.mean((reference - distorted) ** 2)
    if mse == 0:
        return float("inf")

    psnr = 10 * np.log10((max_pixel_value**2) / mse)
    return float(psnr)


def compute_ssim(
    reference: np.ndarray,
    distorted: np.ndarray,
    max_pixel_value: float = 255.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images using a
    Gaussian-weighted sliding window (matches the standard Wang et al. 2004
    formulation). Computed on grayscale; for color images, each image is
    converted to grayscale first.

    Returns:
        Mean SSIM index in [-1, 1] (1.0 = identical images).
    """
    if reference.shape != distorted.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs distorted {distorted.shape}"
        )

    ref_gray = _to_grayscale_if_color(reference).astype(np.float64)
    dist_gray = _to_grayscale_if_color(distorted).astype(np.float64)

    C1 = (0.01 * max_pixel_value) ** 2
    C2 = (0.03 * max_pixel_value) ** 2

    kernel = cv2.getGaussianKernel(window_size, sigma)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(ref_gray, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(dist_gray, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(ref_gray**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(dist_gray**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(ref_gray * dist_gray, -1, window)[5:-5, 5:-5] - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(ssim_map.mean())


def compute_mse(reference: np.ndarray, distorted: np.ndarray) -> float:
    """Compute Mean Squared Error between two images."""
    if reference.shape != distorted.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs distorted {distorted.shape}"
        )
    reference = reference.astype(np.float64)
    distorted = distorted.astype(np.float64)
    return float(np.mean((reference - distorted) ** 2))


def evaluate_pair(reference: np.ndarray, distorted: np.ndarray) -> dict:
    """Convenience wrapper: compute MSE, PSNR, and SSIM for one image pair."""
    if reference.shape != distorted.shape:
        # Resize distorted to match reference so comparison is still possible
        distorted = cv2.resize(distorted, (reference.shape[1], reference.shape[0]))

    return {
        "mse": compute_mse(reference, distorted),
        "psnr": compute_psnr(reference, distorted),
        "ssim": compute_ssim(reference, distorted),
    }
