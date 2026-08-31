"""잡음 세기 추정 — 과제가 n 의 성질을 알려주지 않으므로 g 에서 직접 뽑아낸다.

세 가지 독립 추정을 두고 서로 교차 검증한다.
  1) wavelet MAD  — 최고주파 대역 계수의 중앙절대편차 (가우시안 가정, 표준 방법)
  2) 평탄 영역    — 국소 분산이 가장 낮은 타일들의 표준편차
  3) 고주파 코너  — dipole 커널이 신호를 거의 통과시키지 않는 k 영역의 잔여 파워
"""

from __future__ import annotations

import numpy as np

from dipole import dipole_kernel

__all__ = ["sigma_mad", "sigma_flat_patches", "sigma_null_space", "estimate_noise"]


def sigma_mad(g: np.ndarray) -> float:
    """1-레벨 Haar diagonal detail 계수의 MAD 기반 추정.

    diagonal detail 은 매끄러운 신호를 거의 통과시키지 않으므로 사실상 잡음만 남는다.
    MAD/0.6745 는 가우시안에서 표준편차의 강건 추정량이다. (PyWavelets 없이 직접 계산)
    """
    a = g[: g.shape[0] // 2 * 2, : g.shape[1] // 2 * 2]
    b = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2)
    dd = (b[:, 0, :, 0] - b[:, 0, :, 1] - b[:, 1, :, 0] + b[:, 1, :, 1]) / 2.0
    return float(np.median(np.abs(dd)) / 0.6745)


def sigma_flat_patches(g: np.ndarray, patch: int = 16, q: float = 0.05) -> float:
    """가장 평탄한 하위 q 비율 타일의 표준편차 중앙값.

    평탄한 곳에서는 신호 변동이 거의 없으므로 남는 것은 잡음뿐이라는 가정.
    구조가 조금이라도 남아 있으면 과대추정 쪽으로 치우친다.
    """
    ny, nx = g.shape
    ty, tx = ny // patch, nx // patch
    tiles = (
        g[: ty * patch, : tx * patch]
        .reshape(ty, patch, tx, patch)
        .transpose(0, 2, 1, 3)
        .reshape(-1, patch * patch)
    )
    stds = tiles.std(axis=1)
    k = max(1, int(len(stds) * q))
    return float(np.median(np.sort(stds)[:k]))


def sigma_null_space(g: np.ndarray, theta_deg: float, tol: float = 0.02) -> float:
    """|D(k)| < tol 인 k 에 남은 파워로 추정.

    그 자리에는 신호가 (거의) 실려 올 수 없으므로 관측된 것은 잡음이다.
    dipole 커널을 알고 있을 때만 쓸 수 있는, 이 과제에 특화된 추정이다.
    """
    G = np.fft.fft2(g)
    m = np.abs(dipole_kernel(g.shape, theta_deg)) < tol
    m[0, 0] = False
    if not m.any():
        return float("nan")
    # 파세발: E[|G|^2] = N * sigma^2  (백색 가우시안, 실수 신호)
    power = np.mean(np.abs(G[m]) ** 2) / g.size
    return float(np.sqrt(power))


def estimate_noise(g: np.ndarray, theta_deg: float | None = None) -> dict[str, float]:
    """세 추정을 모두 계산하고, 합의값(중앙값)을 함께 돌려준다."""
    out = {"mad": sigma_mad(g), "flat": sigma_flat_patches(g)}
    if theta_deg is not None:
        out["null"] = sigma_null_space(g, theta_deg)
    vals = [v for v in out.values() if np.isfinite(v)]
    out["consensus"] = float(np.median(vals))
    return out
