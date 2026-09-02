"""1일차 답과 2일차 답을 한 모델로 이어 붙인다.

## 왜 이 구조인가

3일차는 `g = h*f + n` 이다. 여기서 **n 은 측정치 위에서 백색**이다 — 흐림 뒤에
붙었기 때문이다. 그러니 측정치 영역에서 보면 이것은 정확히 1일차 문제다.

    z = net(g)              측정치 영역 디노이징   <- 1일차 문제 그대로
    x = (D·Z) / (D² + λ)    역필터                <- 2일차 답 그대로

전달 곡선을 재보면 이 분해가 왜 옳은지가 숫자로 나온다 (val 30장).

    측정치 영역 정확도   ->   최종 PSNR / SSIM
             25 dB              20.24 / 0.5572
             35 dB              26.83 / 0.7809     <- 지금 우리 위치
             40 dB              29.86 / 0.8626
             45 dB              32.70 / 0.9066
        완벽 (오차 0)           71.47 / 0.9999     <- 2일차 답이 그대로 산다

마지막 줄이 핵심이다. **구조적 한계가 없다.** 최종 점수는 오직 측정치 영역을 얼마나
잘 지웠느냐로 결정된다. 그리고 그건 1일차에 이미 잘 하는 법을 배운 문제다.

## 전개형과 무엇이 다른가

`UnrolledNet` 은 네트워크를 **이미지 영역**에 둔다 — 먼저 Wiener 로 역산하고 그
결과를 다듬는다. 그러면 네트워크가 보는 잡음은 1/D 로 증폭돼 주파수마다 세기가 다르고
X자 방향으로 상관돼 있다. 1일차 디노이저가 한 번도 본 적 없는 종류다.

여기서는 네트워크를 **측정치 영역**에 둔다. 잡음이 백색이고, 1일차 가중치가 그대로
의미를 가지며, 역산의 병적인 부분은 전부 닫힌 해가 처리한다.

    이미지 영역 (전개형)   노이즈가 유색 · 역산도 네트워크가 감당
    측정치 영역 (여기)     노이즈가 백색 · 역산은 해석적으로 끝

## 손실은 최종 이미지에서

배포 방법 B 는 디노이저를 측정치 영역 정답(`h*f`)에 맞춰 따로 학습하고 나중에 Wiener 를
건다. 여기서는 **역필터까지 모델 안에 넣고 최종 이미지에서 손실을 잰다.** 그래야

  - 채점에 쓰는 SSIM 을 그대로 손실에 넣을 수 있고 (통과 기준이 SSIM 0.83 이다)
  - λ 를 같이 학습할 수 있다 — 위 표에서 최적 K 가 디노이징 품질에 따라
    1e-2 에서 1e-8 까지 네 자릿수를 움직인다. 고정값으로는 못 맞춘다

## λ

`log_lam` 으로 두어 항상 양수다. `lam_map=True` 면 **주파수마다** 배율을 따로 학습한다
(2일차 교훈 — 주파수별 계수는 학습률을 따로 크게 줘야 움직인다, `--lr-spectral`).
|D| 가 큰 곳은 측정치를 믿고 영널 원뿔 근처는 사전지식에 기대는 것이 최적인데,
스칼라 하나로는 그 둘을 동시에 못 맞춘다.

## refine

`refine_iters>0` 이면 역필터 뒤에 이미지 영역 다듬기를 몇 번 더 한다. 측정치 영역에서
못 지운 것이 역산으로 증폭돼 남으면 그것만 정리하는 자리다. 0 이면 순수 2단이다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "denoise"))
from models import build_model  # noqa: E402

from unrolled import data_consistency, estimate_sigma  # noqa: E402

__all__ = ["TwoStageNet"]


class TwoStageNet(nn.Module):
    """측정치 영역 디노이저 → 역필터 (→ 선택적 이미지 영역 다듬기)."""

    def __init__(
        self,
        model: str = "drunet",
        features: int = 64,
        sigma_map: bool = False,
        lam_map: bool = False,
        # **시작 시점**에 맞는 λ 를 준다. 원시 측정치의 val 최적 K 가 3.16e-2 다.
        # 수렴 후의 최적값(40 dB 일 때 1.8e-3)으로 시작하면 아직 아무것도 지우지
        # 않은 입력을 그 λ 로 역산하게 되어 노이즈를 크게 증폭한 채 출발한다
        # (K=3e-3 이면 원시 측정치에서 11 dB 언저리, K=3.16e-2 면 14.10 dB).
        # λ 는 학습되므로 디노이저가 좋아지는 만큼 알아서 내려간다.
        init_lam: float = 3.16e-2,
        refine_iters: int = 0,
        shape: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        self.sigma_map = sigma_map and model == "drunet"
        kw = {"sigma_map": True} if self.sigma_map else {}
        self.denoiser = build_model(model, features=features, num_of_layers=17, **kw)

        self.refine_iters = refine_iters
        self.refiners = nn.ModuleList(
            [build_model(model, features=features, num_of_layers=17, **kw)
             for _ in range(refine_iters)]
        )
        n_lam = 1 + refine_iters
        self.log_lam = nn.Parameter(
            torch.full((n_lam,), float(torch.log(torch.tensor(init_lam)))))
        self.log_lam_map = nn.Parameter(torch.zeros(n_lam, *shape)) if lam_map else None

    # 주파수별 배율이 벗어날 수 있는 범위. log 공간이라 ±3 이면 e^±3 = 20배다.
    # 제한이 없으면 어떤 주파수는 폭주하고 어떤 주파수는 0 이 되어 학습이 깨진다
    # (Adam 은 gradient 크기와 무관하게 스텝당 대략 lr 만큼 움직인다).
    LAM_MAP_CLAMP = 3.0

    def _lam(self, k: int, b: int, device):
        lam = self.log_lam[k].exp().expand(b)
        lmap = None
        if self.log_lam_map is not None:
            lmap = self.log_lam_map[k].clamp(-self.LAM_MAP_CLAMP, self.LAM_MAP_CLAMP).exp()
        return lam, lmap

    def forward(self, measure: Tensor, b0: tuple[float, float] = (0.0, 1.0)) -> Tensor:
        # σ 는 측정치에서 읽는다 — 정답도 메타데이터도 쓰지 않는다
        sigma = estimate_sigma(measure, b0=b0) if self.sigma_map else None

        # 1단: 측정치 영역에서 노이즈를 지운다. 여기가 1일차 문제다
        z = self.denoiser(measure, sigma) if self.sigma_map else self.denoiser(measure)

        # 2단: 역필터. 학습 파라미터는 λ 뿐이고 나머지는 닫힌 해다
        lam, lmap = self._lam(0, measure.shape[0], measure.device)
        x = data_consistency(torch.zeros_like(z), z, lam, b0, lmap)

        # 3단(선택): 남은 것만 이미지 영역에서 정리한다
        for k, net in enumerate(self.refiners, start=1):
            r = net(x, sigma) if self.sigma_map else net(x)
            lam, lmap = self._lam(k, measure.shape[0], measure.device)
            x = data_consistency(r, measure, lam, b0, lmap)
        return x
