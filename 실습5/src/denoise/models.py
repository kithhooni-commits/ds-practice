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


class ResBlock(nn.Module):
    """conv-ReLU-conv + skip. BN 없음 — 정규화 계층은 노이즈 세기 정보를 지운다."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(x)


class DRUNet(nn.Module):
    """U-Net + residual block. DnCNN 을 대체할 더 강한 구조.

    DnCNN 의 한계는 **수용영역**이다. 17층 3×3 은 35픽셀밖에 못 본다. 256² 이미지에서
    넓은 평탄 영역의 노이즈를 지우거나 큰 구조를 복원하려면 그보다 넓게 봐야 한다.
    DRUNet 은 3번 다운샘플해서 같은 층수로 수용영역을 8배 넓힌다.

    이 구조를 고른 이유는 성능만이 아니다. DRUNet 은 DPIR(Plug-and-Play Image
    Restoration with Deep Denoiser Prior)의 디노이저 프라이어로 설계됐다.
    3일차의 deconvolution + denoising 결합에서 데이터 정합 단계와 사전지식 단계를
    번갈아 푸는 구조를 쓰려면, 사전지식 자리에 들어갈 디노이저가 바로 이것이다.
    `sigma_map=True` 로 두면 노이즈 세기를 채널로 받는데, 그 반복에서 매 단계
    "이번엔 이만큼만 지워라"라고 지시하는 데 쓴다.

    1일차에는 σ 를 모르므로 sigma_map=False (blind) 로 쓴다.

    참고: Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior",
    TPAMI 2021.
    """

    def __init__(
        self,
        in_nc: int = 1,
        out_nc: int = 1,
        nc: tuple[int, ...] = (64, 128, 256, 512),
        nb: int = 4,
        sigma_map: bool = False,
        global_residual: bool = True,
    ) -> None:
        super().__init__()
        self.sigma_map = sigma_map
        self.global_residual = global_residual
        cin = in_nc + (1 if sigma_map else 0)

        self.head = nn.Conv2d(cin, nc[0], 3, 1, 1, bias=False)

        def down(a, b):
            return nn.Sequential(*[ResBlock(a) for _ in range(nb)],
                                 nn.Conv2d(a, b, 2, 2, 0, bias=False))

        def up(a, b):
            return nn.Sequential(nn.ConvTranspose2d(a, b, 2, 2, 0, bias=False),
                                 *[ResBlock(b) for _ in range(nb)])

        self.down1, self.down2, self.down3 = down(nc[0], nc[1]), down(nc[1], nc[2]), down(nc[2], nc[3])
        self.body = nn.Sequential(*[ResBlock(nc[3]) for _ in range(nb)])
        self.up3, self.up2, self.up1 = up(nc[3], nc[2]), up(nc[2], nc[1]), up(nc[1], nc[0])
        self.tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, bias=False)

    def forward(self, x: Tensor, sigma: Tensor | None = None) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Input tensor must be 4D, but got {x.dim()}D tensor.")

        inp = x
        if self.sigma_map:
            if sigma is None:
                sigma = torch.zeros_like(x[:, :1])
            elif sigma.dim() == 1:
                sigma = sigma.view(-1, 1, 1, 1).expand(-1, 1, *x.shape[2:])
            x = torch.cat([x, sigma], dim=1)

        # 3번 다운샘플하므로 8의 배수로 패딩한다 (256, 128 은 그대로 통과)
        h, w = x.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")

        x1 = self.head(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        y = self.body(x4)
        y = self.up3(y + x4)
        y = self.up2(y + x3)
        y = self.up1(y + x2)
        y = self.tail(y + x1)

        if ph or pw:
            y = y[..., :h, :w]
        return inp + y if self.global_residual else y


def build_model(name: str, **kwargs) -> nn.Module:
    if name == "dncnn":
        return DnCNN(**kwargs)
    if name == "dncnn_plus":
        kwargs.pop("channels", None)
        return DnCNNPlus(**kwargs)
    if name == "drunet":
        # train.py 는 DnCNN 용 인자(num_of_layers/features)를 넘긴다. 여기서는
        # features 로 폭을, num_of_layers 로 레벨당 res block 수를 정한다.
        f = kwargs.pop("features", 64)
        nb = kwargs.pop("num_of_layers", None)
        kwargs.pop("channels", None)
        kwargs.pop("kernel_size", None)
        kwargs.pop("padding", None)
        return DRUNet(nc=(f, f * 2, f * 4, f * 8), nb=4 if nb in (None, 17) else nb, **kwargs)
    raise ValueError(f"unknown model: {name}")
