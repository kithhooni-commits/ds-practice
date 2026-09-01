"""2일차 deconvolution — 배포 코드와 동일한 dipole 커널·연산자.

1일차와 달리 **노이즈가 없다.** `ForwardSimulator` 는 `g = ifft2(fft2(f) · D)` 만 한다.
그래서 문제의 성격이 완전히 다르다 — 노이즈가 없으면 역산이 거의 정확해서
고전 기법(Wiener)이 딥러닝을 크게 앞선다. 배포 예시 로그에서 이미 그렇다.

    입력(blur)        7.89 dB
    U-Net 30 epoch   25.59 dB
    TKD              31.19 dB
    Wiener K=1e-4    42.25 dB     <- 딥러닝보다 +16.7 dB

내가 1일차 초반에 만든 `dipole.py` 와 커널 정의가 미묘하게 다르다. 배포 구현을
그대로 옮겨 채점과 어긋나지 않게 한다. 특히:

  * DC 가 0 이 아니라 **1/3** 이다. 분모에 1e-8 을 더해 0 나눗셈을 피하는데,
    그 결과 D(0) = 1/3 - 0/1e-8 = 1/3 이 된다. 평균이 보존되므로 1일차 README 에
    적었던 "DC 는 복원 불가능" 은 이 과제에는 해당하지 않는다.
  * `np.arange(-N/2, N/2)` 격자에 `fftshift` 를 걸어 곧바로 OTF 로 쓴다.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = ["dipole_otf", "forward", "wiener", "tkd", "tikhonov"]


@lru_cache(maxsize=16)
def dipole_otf(
    shape: tuple[int, int],
    voxel: tuple[float, float] = (1.0, 1.0),
    b0: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """배포 코드 `dipole_kernel` 과 동일. fftshift 가 들어 있어 반환값이 곧 OTF 다."""
    y = np.arange(-shape[1] / 2, shape[1] / 2, 1)
    x = np.arange(-shape[0] / 2, shape[0] / 2, 1)
    Y, X = np.meshgrid(y, x)

    X = X / (shape[0] * voxel[0])
    Y = Y / (shape[1] * voxel[1])

    D = 1 / 3 - (X * b0[0] + Y * b0[1]) ** 2 / (X**2 + Y**2 + 1e-8)
    return np.fft.fftshift(D)


def forward(img: np.ndarray) -> np.ndarray:
    """measure = ifft2(fft2(img) · D). 노이즈 없음 — 배포 ForwardSimulator 와 동일."""
    D = dipole_otf(img.shape)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * D))


def wiener(measure: np.ndarray, K: float) -> np.ndarray:
    """W = (1/D) · |D|² / (|D|² + K). 배포 `wiener_filter` 와 동일.

    K → 0 이면 직접 역산이 된다. 노이즈가 없으므로 K 를 얼마나 작게 둘 수 있는지가
    성능을 지배한다. 유한한 하한이 존재하는 이유는 dipole 의 0 영역 때문이다 —
    거기서는 |D| 가 부동소수점 정밀도까지 내려가 1/D 가 폭발한다.
    """
    D = dipole_otf(measure.shape)
    h2 = np.abs(D) ** 2
    safe = np.where(np.abs(D) < 1e-12, 1e-12, D)
    W = (1.0 / safe) * (h2 / (h2 + K))
    return np.real(np.fft.ifft2(W * np.fft.fft2(measure)))


def tkd(measure: np.ndarray, t: float) -> np.ndarray:
    """|D| < t 인 자리에서 분모를 ±t 로 고정한 절단 역산."""
    D = dipole_otf(measure.shape)
    Dt = np.where(np.abs(D) < t, np.sign(D) * t, D)
    Dt = np.where(Dt == 0, t, Dt)
    return np.real(np.fft.ifft2(np.fft.fft2(measure) / Dt))


def tikhonov(measure: np.ndarray, lam: float) -> np.ndarray:
    """W = D/(D² + λ|k|²). 고주파를 더 세게 눌러 줄무늬를 억제한다."""
    ny, nx = measure.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    k2 = ky**2 + kx**2
    D = dipole_otf(measure.shape)
    den = D**2 + lam * k2
    W = np.where(den > 0, D / np.where(den == 0, 1.0, den), 0.0)
    return np.real(np.fft.ifft2(W * np.fft.fft2(measure)))
