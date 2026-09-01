"""주파수 영역에서 학습하는 신경망 — 커널을 모른 채 역필터를 스스로 찾는다.

## 왜 CNN 이 디컨볼루션에 지는가

디컨볼루션의 정답은 **주파수마다 다른 상수를 곱하는 것**이다.

    f = ifft2( fft2(g) · W ),    W = 1/D

3×3 합성곱을 아무리 쌓아도 "|D|=0.001 인 주파수만 1000배, 옆 주파수는 1배" 같은
극단적으로 선택적인 증폭을 정확히 만들기 어렵다. 배포 U-Net 이 25.59 dB 에 그친
이유가 수용영역이 아니라 이것이다 — 4단 다운샘플이면 이미 전역을 본다.

## 그러면 그 곱셈 자체를 학습시키면 된다

`W` 를 256×256 실수 배열 하나로 두면 파라미터가 65,536개다. **정답이 가설 공간
안에 정확히 들어 있으므로** 학습이 찾아낼 수 있다.

중요한 것은 **커널 D 를 알려주지 않는다**는 점이다. `(측정치, 정답)` 쌍만 보고
역필터를 스스로 발견한다. 그래서 이건 "Wiener 를 코딩한 것"이 아니라 **데이터에서
Wiener 를 학습한 것**이다.

D 가 실수이고 짝함수(D(-k) = D(k))이므로 1/D 도 그렇다. W 를 실수로 두고 출력의
실수부를 취하면 그 성질이 자동으로 지켜진다.

## 주의 — 크기에 묶인다

W 가 256×256 이므로 **크롭 학습을 못 한다.** 128² 패치는 주파수 격자가 달라
같은 W 를 쓸 수 없다. 전체 이미지로 학습해야 한다. 파라미터가 작아 부담은 없다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "denoise"))
from models import build_model  # noqa: E402

__all__ = ["SpectralFilter", "SpectralNet"]


class SpectralFilter(nn.Module):
    """학습 가능한 주파수 영역 곱셈. 이것 하나가 곧 역필터다.

    **1(항등)에서 시작한다.** 처음엔 입력을 그대로 통과시키고, 학습이 각 주파수의
    이득을 키워 나간다. dipole 은 D 가 음수인 영역이 있어 역필터도 부호가 바뀌므로
    실수 파라미터로 두어 자유롭게 음수가 되게 한다.

    이전 판은 `tanh(sign)·exp(log_gain)` 으로 두고 sign 을 0 에서 시작했는데,
    그러면 W 가 통째로 0 이 되고 gain 의 gradient 도 0 이라 **아무것도 학습되지 않는다**.
    실제로 60 epoch 내내 loss 가 미동도 안 했다. 단순한 파라미터화가 옳았다.

    학습률이 결정적이다. 이득이 44,000 까지 가야 해서 나머지 층과 같은 lr 을 쓰면
    4.8 배에서 멈춘다 (`--lr-spectral` 참고).
    """

    def __init__(self, shape: tuple[int, int] = (256, 256), max_gain: float = 1e5) -> None:
        super().__init__()
        self.shape = shape
        self.max_gain = max_gain
        self.w = nn.Parameter(torch.ones(shape))

    def weight(self) -> Tensor:
        return self.w.clamp(-self.max_gain, self.max_gain)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-2:] != self.shape:
            raise ValueError(
                f"SpectralFilter 는 {self.shape} 전용이다. 크롭 없이 전체 이미지로 학습할 것 "
                f"(받은 것: {tuple(x.shape[-2:])})"
            )
        X = torch.fft.fft2(x.float())
        return torch.fft.ifft2(X * self.weight()).real.to(x.dtype)


class SpectralNet(nn.Module):
    """주파수 곱셈 + (선택) 공간 CNN 정리.

    곱셈 하나로 선형 역연산이 끝난다. 그 뒤에 CNN 을 붙이면 선형으로 못 잡는 것
    (경계 효과, 수치 잔차)을 국소적으로 다듬는다. `refine=None` 이면 순수 선형이라
    **학습으로 찾은 Wiener 필터**가 무엇인지 그대로 들여다볼 수 있다.
    """

    def __init__(
        self,
        shape: tuple[int, int] = (256, 256),
        refine: str | None = None,
        features: int = 32,
        max_gain: float = 1e4,
    ) -> None:
        super().__init__()
        self.spec = SpectralFilter(shape, max_gain)
        self.refine = build_model(refine, features=features, num_of_layers=17) if refine else None

    def forward(self, x: Tensor) -> Tensor:
        y = self.spec(x)
        return self.refine(y) if self.refine is not None else y
