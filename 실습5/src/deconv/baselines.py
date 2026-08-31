"""A안 — 학습 없는 기준선. 여기서 나온 숫자가 이후 모든 방법의 바닥이 된다.

전부 같은 형태다: 주파수 영역에서 G 에 어떤 필터 W(k) 를 곱해 F̂ = W·G 를 만든다.
차이는 D(k) ≈ 0 인 자리를 무엇으로 메우느냐 하나뿐이다.

  direct    W = 1/D                      → 메우지 않는다. 발산한다 (반례용)
  tkd       W = 1/D  단, |D|<t 이면 1/(±t) → 잘라낸다. 가장 단순
  wiener    W = D/(D²+λ)                 → 잡음 대비 신호 비로 부드럽게 누른다
  tikhonov  W = D/(D²+λ|k|²)             → 매끄러움을 선호하도록 가중
  cosmos    W_i = D_i/(Σ D_j²)           → 다른 orientation 이 그 자리를 대신 채운다
"""

from __future__ import annotations

import numpy as np

from dipole import dipole_kernel

__all__ = ["direct_inverse", "tkd", "wiener", "tikhonov", "cosmos", "METHODS"]


def _apply(G: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft2(G * W))


def direct_inverse(g: np.ndarray, theta_deg: float = 0.0, eps: float = 1e-12) -> np.ndarray:
    """naive 역연산. 왜 정규화가 필요한지 보여주기 위한 반례."""
    D = dipole_kernel(g.shape, theta_deg)
    W = np.where(np.abs(D) > eps, 1.0 / np.where(D == 0, 1.0, D), 0.0)
    return _apply(np.fft.fft2(g), W)


def tkd(g: np.ndarray, theta_deg: float = 0.0, t: float = 0.15) -> np.ndarray:
    """Truncated K-space Division. |D|<t 인 자리에서 분모를 ±t 로 고정한다."""
    D = dipole_kernel(g.shape, theta_deg)
    Dt = np.where(np.abs(D) < t, np.sign(D) * t, D)
    Dt = np.where(Dt == 0, t, Dt)  # D 가 정확히 0 이면 sign 도 0 이 되므로
    W = 1.0 / Dt
    W[0, 0] = 0.0
    return _apply(np.fft.fft2(g), W)


def wiener(g: np.ndarray, theta_deg: float = 0.0, lam: float = 1e-2) -> np.ndarray:
    """W = D/(D²+λ). λ ≈ 잡음파워/신호파워 로 두면 MSE 최적에 가깝다."""
    D = dipole_kernel(g.shape, theta_deg)
    W = D / (D ** 2 + lam)
    return _apply(np.fft.fft2(g), W)


def tikhonov(g: np.ndarray, theta_deg: float = 0.0, lam: float = 1e-3) -> np.ndarray:
    """W = D/(D²+λ|k|²). 고주파일수록 더 세게 눌러 줄무늬를 억제한다."""
    ny, nx = g.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    k2 = ky ** 2 + kx ** 2
    D = dipole_kernel(g.shape, theta_deg)
    den = D ** 2 + lam * k2
    W = np.where(den > 0, D / np.where(den == 0, 1.0, den), 0.0)
    W[0, 0] = 0.0
    return _apply(np.fft.fft2(g), W)


def cosmos(
    gs: list[np.ndarray] | np.ndarray,
    thetas: list[float],
    lam: float = 0.0,
) -> np.ndarray:
    """여러 orientation 을 한 번에 푸는 최소제곱 해.

        F̂ = Σ_i D_i* G_i / (Σ_i D_i² + λ)

    orientation 이 3개 이상이면 어떤 k 에서도 최소 하나의 D_i 가 0 에서 충분히 떨어져
    있어 분모가 살아난다. 이것이 규칙의 'orientation ≤ 6' 이 열어 둔 길이다.
    """
    gs = list(gs)
    assert len(gs) == len(thetas) and len(gs) > 0
    shape = gs[0].shape
    num = np.zeros(shape, dtype=complex)
    den = np.zeros(shape, dtype=float)
    for g, th in zip(gs, thetas):
        D = dipole_kernel(shape, th)
        num += D * np.fft.fft2(g)
        den += D ** 2
    den = den + lam
    F = np.where(den > 0, num / np.where(den == 0, 1.0, den), 0.0)
    F[0, 0] = 0.0
    return np.real(np.fft.ifft2(F))


METHODS = {
    "direct": direct_inverse,
    "tkd": tkd,
    "wiener": wiener,
    "tikhonov": tikhonov,
}
