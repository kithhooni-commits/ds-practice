"""DC-Net — 신뢰 가능한 주파수는 정확히 역산하고, 신경망에게는 null cone 만 맡긴다.

## 왜 이 구조인가

디컨볼루션이 어려운 이유는 **전 대역이 어려워서가 아니다.** |D| 가 큰 대부분의
주파수는 나눗셈 한 번으로 정확히 되돌아온다. 문제는 magic-angle cone 근처의 얇은
X자 영역뿐이다 — 거기서만 1/D 가 수천~수만 배로 폭발한다.

그런데 end-to-end 신경망에게 흐린 이미지를 통째로 주면, **쉬운 부분까지 전부 학습으로
풀라고 시키는 셈**이다. 배포 U-Net 이 25.59 dB 에 그친 이유가 그것이다.

DC-Net 은 일을 나눈다.

    |D| > τ  인 주파수   →  X = G/D 로 정확히 복원. **네트워크가 건드리지 못한다**
    |D| ≤ τ  인 주파수   →  정보가 사라진 자리. 신경망이 채운다

`hard data consistency` 라 부른다. 관측으로 결정되는 것은 관측이 정하고, 관측이
결정하지 못하는 것만 사전지식(학습)이 정한다.

## 왜 이게 학습을 쉽게 만드나

네트워크가 1/D 의 극단적 이득을 배울 필요가 없어진다. 출력이 X자 영역에만 반영되고
그 폭이 좁아서, 문제가 "이미지 전체를 복원하라"에서 "이미지의 빠진 줄무늬를 메워라"로
바뀐다. 후자는 CNN 이 잘하는 종류의 일이다 (inpainting 에 가깝다).

## τ 의 선택

작을수록 신경망 몫이 줄지만, |D|=τ 에서 1/τ 배 증폭이 일어나 측정 노이즈·양자화
오차가 커진다. 노이즈가 없는 2일차에서는 작게, 노이즈가 있으면 크게 잡는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "denoise"))
from models import build_model  # noqa: E402

from challenge import dipole_otf  # noqa: E402

__all__ = ["DCNet"]

_CACHE: dict = {}


def _otf(shape, b0, device) -> Tensor:
    key = (shape, b0, device)
    if key not in _CACHE:
        _CACHE[key] = torch.from_numpy(dipole_otf(shape, b0=b0)).float().to(device)
    return _CACHE[key]


class DCNet(nn.Module):
    """hard data consistency + null cone 을 채우는 신경망.

    forward 흐름:

        X_known = G/D   (|D| > τ 인 곳만, 나머지는 0)
        x0      = ifft2(X_known)                 관측이 정하는 부분
        z       = net(x0)                        신경망의 전체 이미지 추정
        X_final = X_known + fft2(z)·(|D| ≤ τ)    빠진 대역만 z 로 채운다
        x       = ifft2(X_final)

    `net` 의 출력 중 |D| > τ 성분은 **버려진다.** 그래서 네트워크는 X자 영역을
    메우는 데만 파라미터를 쓴다.
    """

    def __init__(
        self,
        model: str = "unet",
        features: int = 32,
        tau: float = 0.05,
        b0: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        self.net = build_model(model, features=features, num_of_layers=17)
        self.tau = tau
        self.b0 = b0

    def split(self, measure: Tensor, b0=None):
        """(관측으로 정해지는 부분 x0, null cone 마스크) 를 만든다."""
        D = _otf(tuple(measure.shape[-2:]), b0 or self.b0, measure.device)
        reliable = D.abs() > self.tau
        G = torch.fft.fft2(measure.float())
        X = torch.where(reliable, G / torch.where(reliable, D, torch.ones_like(D)),
                        torch.zeros_like(G))
        return torch.fft.ifft2(X).real, X, ~reliable

    def forward(self, measure: Tensor, b0=None) -> Tensor:
        x0, X_known, missing = self.split(measure, b0)
        z = self.net(x0.to(measure.dtype))
        Z = torch.fft.fft2(z.float())
        X = X_known + Z * missing
        return torch.fft.ifft2(X).real.to(measure.dtype)
