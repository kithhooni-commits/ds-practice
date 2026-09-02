"""과평활을 되돌린다 — 학습 없이 SSIM 을 올리는 후처리.

## 무엇이 문제인가

    배포 baseline   25.01 dB / 0.8149
    우리 v1 ep41    27.92 dB / 0.8004    <- PSNR 은 2.9 dB 이기는데 SSIM 은 진다

L1·L2 계열 손실은 **뭉개는 쪽**으로 수렴한다. 평균을 맞추면 오차 제곱은 줄지만
국소 분산과 상관을 잃는다. SSIM 이 재는 것이 정확히 그 둘이다.

    SSIM = (2·μx·μy + C1)/(μx² + μy² + C1) · (2·σxy + C2)/(σx² + σy² + C2)
                    밝기                            대비 · 구조

밝기 항은 이미 맞다 (PSNR 이 높다). 깎이는 것은 두 번째 항이고, 원인은 σx 가 σy 보다
작다는 것 — 출력의 국소 대비가 정답보다 낮다.

## 되돌리는 법

언샤프 마스킹. 흐린 성분을 빼서 국소 대비를 키운다.

    out' = out + amount · (out − blur(out, σ))

`amount` 와 `σ` 는 **val 에서** 고른다. 세게 걸면 노이즈까지 살아나 PSNR 이 떨어지므로
공짜가 아니다. SSIM 을 최대로 하되 PSNR 이 기준선 아래로 내려가지 않는 지점을 찾는다.

## 결론: 듣지 않는다 (실측)

과평활은 실재한다. 그런데 언샤프로는 되돌릴 수 없다.

    복원의 국소 std 0.06590 · 정답 0.07814 · 비율 0.843   <- 확실히 뭉갰다

    후처리 전            21.71 dB / 0.6063
    amount 0.15 s 1.2    21.46 / 0.5914
    amount 0.30 s 1.2    21.16 / 0.5757
    amount 0.50 s 1.2    20.72 / 0.5545   <- 세게 걸수록 나빠진다

SSIM 의 구조 항은 2·σxy/(σx² + σy²) 다. 언샤프는 신호와 **잔차 노이즈를 같이** 키우는데,
잔차는 정답과 무상관이라 분자 σxy 는 늘지 않고 분모 σx² 만 커진다. 그래서 SSIM 이
떨어진다. PSNR 도 같이 떨어진다.

**잃어버린 국소 대비는 사후에 만들 수 없다.** 정답과 상관된 구조로 되살려야 하고,
그건 모델만 할 수 있다 — 학습 때부터 SSIM 을 손실에 넣는 것이 유일한 정공법이다
(`--loss charbonnier_ssim`). 이 모듈은 그 결론의 근거로 남긴다.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "denoise"))

__all__ = ["unsharp", "tune_sharpen"]

_K: dict = {}


def _gauss(sigma: float, device, dtype) -> Tensor:
    """분리형 가우시안 커널. 3σ 까지 자른다."""
    key = (round(sigma, 4), device, dtype)
    if key not in _K:
        r = max(1, int(math.ceil(3 * sigma)))
        x = torch.arange(-r, r + 1, device=device, dtype=dtype)
        k = torch.exp(-(x**2) / (2 * sigma * sigma))
        _K[key] = (k / k.sum()).view(1, 1, -1)
    return _K[key]


def unsharp(x: Tensor, amount: float, sigma: float) -> Tensor:
    """out + amount · (out − blur(out)). amount=0 이면 그대로 돌려준다."""
    if amount == 0.0:
        return x
    k = _gauss(sigma, x.device, x.dtype)
    r = k.shape[-1] // 2
    # 경계는 반사로 채운다 — 0 으로 채우면 테두리에 인공적인 대비가 생긴다
    b = F.pad(x, (r, r, 0, 0), mode="reflect")
    b = F.conv2d(b, k.view(1, 1, 1, -1))
    b = F.pad(b, (0, 0, r, r), mode="reflect")
    b = F.conv2d(b, k.view(1, 1, -1, 1))
    return x + amount * (x - b)


def tune_sharpen(val_items, base_fn, min_psnr: float | None = None,
                 amounts=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
                 sigmas=(0.8, 1.2, 1.8, 2.5)):
    """val 에서 (amount, sigma) 를 고른다. SSIM 최대, 단 PSNR 하한을 지킨다.

    `min_psnr` 를 주면 그 아래로 내려가는 설정은 후보에서 뺀다. 통과 기준이
    PSNR 26 · SSIM 0.83 이므로 PSNR 을 지키면서 SSIM 을 올리는 것이 목적이다.
    """
    from day3_common import score

    print(f"[언샤프 튜닝 — val {len(val_items)}장]")
    print(f"{'amount':>8}{'sigma':>8}{'PSNR':>10}{'SSIM':>10}")
    print("-" * 36)
    best = (0.0, 0.0, -1.0, -1.0)   # amount, sigma, ssim, psnr
    for a in amounts:
        for s in (sigmas if a else (sigmas[0],)):
            p, ss = score(val_items, lambda g, a=a, s=s: unsharp(base_fn(g), a, s))
            keep = min_psnr is None or p >= min_psnr
            mark = "" if keep else "   (PSNR 하한 미달)"
            print(f"{a:>8.2f}{s:>8.2f}{p:>10.2f}{ss:>10.4f}{mark}")
            if keep and ss > best[2]:
                best = (a, s, ss, p)
            if a == 0.0:
                break
    print(f"→ 선택: amount={best[0]:.2f} sigma={best[1]:.2f}   "
          f"val {best[3]:.2f} / {best[2]:.4f}\n")
    return best[0], best[1]
