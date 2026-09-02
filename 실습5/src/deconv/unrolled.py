"""Unrolled network — 데이터 정합과 사전지식을 번갈아 N번.

## 왜 이 구조인가

지금까지 본 것을 그대로 이어 붙인 결과다.

    측정치를 신경망에 그대로   19.8 dB   역연산은 CNN 이 못 한다 (조건수)
    Wiener 단독               32.1 dB   역연산은 되지만 잔여 오차가 남는다
    Wiener → 신경망 1회       41.1 dB   각자 잘하는 일을 시켰다
    전개형 (N회 반복)         ?         고치고, 물리로 되돌리고, 다시 고친다

Wiener + 신경망은 **한 번에** 다 지워야 한다. 전개형은 조금씩 고치되 매번
물리 제약으로 되돌리므로 오차가 누적되지 않는다.

## 한 단계

    z_k = 신경망(x_{k-1})                      사전지식. 학습된다
    x_k = argmin ‖D·x − g‖² + λ_k‖x − z_k‖²    데이터 정합. 학습 파라미터 없음

dipole 은 주파수 영역에서 **대각 연산자**라 두 번째 줄이 닫힌 형태로 정확히 풀린다.

    X_k = (D*·G + λ_k·Z_k) / (|D|² + λ_k)

반복마다 FFT 두 번이면 끝이고 역행렬도, 켤레기울기도 필요 없다. λ_k 는 단계마다
따로 학습해서 "이번엔 측정치를 얼마나 믿을지"를 스스로 정하게 한다.

노이즈가 크면 λ 가 커져 사전지식 쪽으로, 작으면 λ 가 작아져 데이터 쪽으로 기운다 —
2일차에서 K 가 모델 오차를 흡수하던 것과 같은 역할을 학습으로 얻는다.

## 3일차와의 관계

3일차는 `g = h*f + n` 이다. 이 구조가 그대로 답이 된다. 데이터 정합 단계는 그대로
두고, 사전지식 자리에 1일차에서 학습한 디노이저(DRUNet)를 넣으면 plug-and-play 가
되고, 그 디노이저까지 같이 학습하면 전개형이 된다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "denoise"))
from models import build_model  # noqa: E402

from challenge import dipole_otf  # noqa: E402

__all__ = ["UnrolledNet", "data_consistency", "estimate_sigma", "self_ensemble"]

_CACHE: dict = {}


def estimate_sigma(measure: Tensor, tol: float = 0.02,
                   b0: tuple[float, float] = (0.0, 1.0)) -> Tensor:
    """측정치만 보고 σ 를 읽는다. 정답이 필요 없다.

    `|D| < tol` 인 주파수에는 신호가 실려올 수 없다 — dipole 이 그리로 아무것도
    보내지 않기 때문이다. 거기 남은 것은 전부 노이즈다. 파세발로

        E|G|² = N·σ²

    3일차 σ 는 이미지마다 0.0007 ~ 0.13 으로 **200배** 차이가 난다. 하나의 blind
    모델로 그 범위를 다 덮으려면 평균적으로 타협해야 하지만, σ 를 알려주면 매 장에
    맞는 세기로 지울 수 있다. val 40장에서 상대오차 중앙값 1.9%.

    커널을 알기에 쓸 수 있는 추정기다. MAD 와 달리 이미지의 고주파를 노이즈로
    착각하지 않는다 — 1일차 σ 게이트가 실패한 원인이 그것이었다.
    """
    D = _otf(tuple(measure.shape[-2:]), b0, measure.device, torch.float32)
    mask = D.abs() < tol
    G = torch.fft.fft2(measure.float())
    n_pix = measure.shape[-1] * measure.shape[-2]
    return ((G.abs() ** 2)[..., mask].mean(-1) / n_pix).sqrt().view(-1)


# dipole 이 견디는 대칭. transpose·90도 회전은 B0 방향을 돌려버려서 못 쓴다
# (직접 확인: flip x/y·180도는 오차 1e-16, transpose·rot90 은 3.17)
_SYMS = [
    (lambda a: a, lambda a: a),
    (lambda a: a.flip(-1), lambda a: a.flip(-1)),
    (lambda a: a.flip(-2), lambda a: a.flip(-2)),
    (lambda a: a.flip(-1, -2), lambda a: a.flip(-1, -2)),
]


@torch.no_grad()
def self_ensemble(fn, measure: Tensor) -> Tensor:
    """4× self-ensemble. 학습 없이 먹는 점수다.

    1일차엔 8× (뒤집기 + 90도 회전) 를 썼지만 3일차엔 4개만 유효하다. dipole 은
    B0 방향을 가지므로 회전하면 **연산자 자체가 바뀐다** — 돌린 입력에 안 돌린
    커널을 적용하는 꼴이 되어 오히려 손해다.
    """
    return torch.stack([inv(fn(t(measure))) for t, inv in _SYMS]).mean(0)


def _otf(shape: tuple[int, int], b0: tuple[float, float], device, dtype) -> Tensor:
    key = (shape, b0, device, dtype)
    if key not in _CACHE:
        _CACHE[key] = torch.from_numpy(dipole_otf(shape, b0=b0)).to(device=device, dtype=dtype)
    return _CACHE[key]


def data_consistency(z: Tensor, measure: Tensor, lam: Tensor,
                     b0: tuple[float, float] = (0.0, 1.0)) -> Tensor:
    """argmin_x ‖D·x − g‖² + λ‖x − z‖² 의 닫힌 해.

        X = (D*·G + λ·Z) / (|D|² + λ)

    D 가 실수 대각이므로 D* = D 다. λ 는 양수여야 하고, 배치마다 다를 수 있다.
    """
    D = _otf(tuple(z.shape[-2:]), b0, z.device, torch.float32)
    lam = lam.view(-1, 1, 1, 1).to(z.dtype)
    G = torch.fft.fft2(measure.float())
    Z = torch.fft.fft2(z.float())
    X = (D * G + lam.float() * Z) / (D**2 + lam.float())
    return torch.fft.ifft2(X).real.to(z.dtype)


class UnrolledNet(nn.Module):
    """N 단계 전개형.

    `share_weights=True` 면 모든 단계가 같은 신경망을 쓴다 (파라미터 N배 절약,
    반복 횟수를 추론 때 바꿀 수 있다). False 면 단계마다 따로 학습한다.

    λ 는 `log_lam` 으로 저장해 항상 양수가 되게 한다.
    """

    def __init__(
        self,
        n_iter: int = 4,
        model: str = "dncnn",
        features: int = 64,
        share_weights: bool = True,
        init_lam: float = 0.05,
        sigma_map: bool = False,
    ) -> None:
        super().__init__()
        self.n_iter = n_iter
        n_net = 1 if share_weights else n_iter
        self.share_weights = share_weights
        # σ 조건화는 DRUNet 만 받는다 (여분 채널로 σ 를 넣는 구조)
        self.sigma_map = sigma_map and model == "drunet"
        kw = {"sigma_map": True} if self.sigma_map else {}
        self.nets = nn.ModuleList(
            [build_model(model, features=features, num_of_layers=17, **kw) for _ in range(n_net)]
        )
        self.log_lam = nn.Parameter(torch.full((n_iter,), float(torch.log(torch.tensor(init_lam)))))

    def net(self, k: int) -> nn.Module:
        return self.nets[0] if self.share_weights else self.nets[k]

    def forward(self, measure: Tensor, x0: Tensor | None = None,
                b0: tuple[float, float] = (0.0, 1.0)) -> Tensor:
        # 시작점은 Wiener 결과를 받거나, 없으면 첫 데이터 정합으로 만든다
        x = x0 if x0 is not None else data_consistency(
            torch.zeros_like(measure), measure, self.log_lam[0].exp().expand(measure.shape[0]), b0)

        # σ 는 측정치에서 읽는다 — 정답도, 메타데이터도 쓰지 않는다
        sigma = estimate_sigma(measure, b0=b0) if self.sigma_map else None

        for k in range(self.n_iter):
            z = self.net(k)(x, sigma) if self.sigma_map else self.net(k)(x)
            lam = self.log_lam[k].exp().expand(measure.shape[0])
            x = data_consistency(z, measure, lam, b0)
        return x
