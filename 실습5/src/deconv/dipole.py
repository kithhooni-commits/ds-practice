"""2D dipole 커널과 forward 열화 모델.

과제의 열화 모델은  g = h * f + n  이고, h 는 dipole 커널이다.
푸리에 영역에서 dipole 은 곱셈으로 작용한다.

    D(k) = 1/3 - (k · b̂)^2 / |k|^2

b̂ 은 주자장(B0) 방향의 단위벡터다. D(k) = 0 이 되는 곳은

    (k · b̂)^2 / |k|^2 = 1/3   →   k 와 b̂ 사이 각이 54.7°(magic angle)

3D 에서는 원뿔(cone), 2D 에서는 원점을 지나는 두 직선이다. 이 선 위에서 D 가 0 이므로
G/D 를 그대로 계산하면 그 방향으로 줄무늬가 폭발한다. 이 모듈은 그 커널을 만들고,
학습 데이터를 만들 때 쓸 forward 시뮬레이터를 제공한다.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "dipole_kernel",
    "zero_cone_mask",
    "forward",
    "make_orientations",
]


def _k_grid(shape: tuple[int, int], voxel: tuple[float, float] = (1.0, 1.0)):
    """fftshift 되지 않은(=np.fft.fft2 와 같은 배치) 주파수 격자."""
    ny, nx = shape
    ky = np.fft.fftfreq(ny, d=voxel[0])[:, None]
    kx = np.fft.fftfreq(nx, d=voxel[1])[None, :]
    return ky, kx


def dipole_kernel(
    shape: tuple[int, int],
    theta_deg: float = 0.0,
    voxel: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """주파수 영역 2D dipole 커널 D(k).

    theta_deg 는 B0 방향을 +y 축에서 반시계로 돌린 각도(도).
    orientation 을 바꾼다는 것은 이 각도를 바꾸는 것이고, 그러면 0 이 되는 직선의
    위치도 같이 돈다. 여러 orientation 을 합치면 각자의 0 자리가 서로 달라
    k-space 의 빈칸이 메워진다 — COSMOS 가 성립하는 이유다.

    반환값은 실수 배열, 범위는 [-2/3, 1/3]. DC(k=0)에는 0 을 넣는다.
    """
    ky, kx = _k_grid(shape, voxel)
    t = np.deg2rad(theta_deg)
    by, bx = np.cos(t), np.sin(t)

    k2 = ky ** 2 + kx ** 2
    kb = ky * by + kx * bx

    with np.errstate(divide="ignore", invalid="ignore"):
        d = 1.0 / 3.0 - (kb ** 2) / k2
    d[0, 0] = 0.0
    return d


def zero_cone_mask(
    shape: tuple[int, int],
    theta_deg: float = 0.0,
    tol: float = 0.1,
    voxel: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """|D(k)| < tol 인 영역 — 정보가 사실상 사라진 자리."""
    return np.abs(dipole_kernel(shape, theta_deg, voxel)) < tol


def forward(
    f: np.ndarray,
    theta_deg: float = 0.0,
    sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    voxel: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """g = h * f + n. 컨볼루션은 FFT 로, 잡음은 가우시안(공간영역 가산).

    sigma 는 f 의 값 범위와 같은 단위다 (f 를 [0,1] 로 정규화해 쓰면 sigma 도 그 척도).
    """
    F = np.fft.fft2(f)
    D = dipole_kernel(f.shape, theta_deg, voxel)
    g = np.real(np.fft.ifft2(F * D))
    if sigma > 0:
        rng = np.random.default_rng() if rng is None else rng
        g = g + rng.normal(0.0, sigma, size=g.shape)
    return g


def make_orientations(n: int = 6, span_deg: float = 180.0) -> list[float]:
    """orientation 각도 목록. 규칙상 clean 1장당 최대 6개.

    dipole 은 180° 주기이므로 [0, 180) 을 균등 분할하는 것이 0 영역을 가장 넓게
    흩어 놓는다.
    """
    return [i * span_deg / n for i in range(n)]
