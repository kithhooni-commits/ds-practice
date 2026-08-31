"""PSNR / SSIM — 과제 평가 지표. 데이터 범위를 명시적으로 넘긴다."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

__all__ = ["psnr", "ssim", "scores"]


def psnr(ref: np.ndarray, est: np.ndarray, data_range: float = 1.0) -> float:
    return float(peak_signal_noise_ratio(ref, est, data_range=data_range))


def ssim(ref: np.ndarray, est: np.ndarray, data_range: float = 1.0) -> float:
    return float(structural_similarity(ref, est, data_range=data_range))


def scores(ref: np.ndarray, est: np.ndarray, data_range: float = 1.0) -> dict[str, float]:
    return {"psnr": psnr(ref, est, data_range), "ssim": ssim(ref, est, data_range)}
