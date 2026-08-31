"""네트워크 — 제공된 DnCNN 과, 그것을 최소한으로 손본 개선판.

개선판(`DnCNNPlus`)이 바꾸는 것은 딱 하나, **입력 채널**이다.

제공 baseline 결과를 보면 salt & pepper 가 유독 어렵다 (입력 17.3 dB, 다른 노이즈는
24~30 dB). 임펄스 노이즈는 값이 0 이나 max 로 완전히 튀어 버려서, 3×3 합성곱이
주변에서 값을 복원하려면 여러 층을 거쳐야 한다. 반면 median 필터는 임펄스를
한 번에 지운다 — 제공 baseline 에서도 median 이 s&p 에서 28.5 dB 로 mean(23.2)을
5 dB 앞선다.

그래서 median 3×3 결과를 두 번째 입력 채널로 같이 넣어 준다. 네트워크가 임펄스를
스스로 배우는 대신, "이미 임펄스가 지워진 버전"을 참고해서 나머지 디테일만
복구하면 된다. 파라미터는 첫 층의 채널 하나만큼만 늘어난다.

출력은 두 경우 모두 **잔차 학습**이다: out = noisy + net(input).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from filters import median_filter

__all__ = ["DnCNN", "DnCNNPlus", "build_model"]


def _body(in_ch: int, out_ch: int, num_of_layers: int, kernel_size: int, padding: int, features: int) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, features, kernel_size=kernel_size, padding=padding, bias=False),
        nn.SiLU(inplace=True),
    ]
    for _ in range(num_of_layers - 1):
        layers += [
            nn.Conv2d(features, features, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(4, features),
            nn.SiLU(inplace=True),
        ]
    layers.append(nn.Conv2d(features, out_ch, kernel_size=kernel_size, padding=padding, bias=False))
    return nn.Sequential(*layers)


class DnCNN(nn.Module):
    """과제 제공 구조 그대로 (17층, 64채널, GroupNorm+SiLU, 전역 잔차)."""

    def __init__(
        self,
        channels: int = 1,
        num_of_layers: int = 17,
        kernel_size: int = 3,
        padding: int = 1,
        features: int = 64,
    ) -> None:
        super().__init__()
        self.dncnn = _body(channels, channels, num_of_layers, kernel_size, padding, features)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Input tensor must be 4D, but got {x.dim()}D tensor.")
        return x + self.dncnn(x)


class DnCNNPlus(nn.Module):
    """입력에 median 3×3 채널을 덧붙인 DnCNN. 출력은 여전히 noisy 기준 잔차."""

    def __init__(
        self,
        num_of_layers: int = 17,
        kernel_size: int = 3,
        padding: int = 1,
        features: int = 64,
        median_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.median_kernel = median_kernel
        self.dncnn = _body(2, 1, num_of_layers, kernel_size, padding, features)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Input tensor must be 4D, but got {x.dim()}D tensor.")
        med = median_filter(x, kernel_size=self.median_kernel)
        return x + self.dncnn(torch.cat([x, med], dim=1))


def build_model(name: str, **kwargs) -> nn.Module:
    if name == "dncnn":
        return DnCNN(**kwargs)
    if name == "dncnn_plus":
        kwargs.pop("channels", None)
        return DnCNNPlus(**kwargs)
    raise ValueError(f"unknown model: {name}")
