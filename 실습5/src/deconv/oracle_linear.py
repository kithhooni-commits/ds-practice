"""선형 방법의 천장 — 정답을 알고 만든 최고의 위너 필터.

발표에서 "선형으로는 여기까지" 를 말하려면 그 천장이 얼마인지 알아야 한다. 보통
위너는 잡음/신호 비를 상수 `K` 하나로 때우지만, **주파수마다 참값** 을 넣으면 그보다
나을 수 없다. 그것이 선형 대각 연산자의 상한이다.

    W(k) = D(k) / (|D(k)|² + N(k)/F(k))

        F(k)  정답의 주파수별 파워   <- 정답을 봐야 안다
        N(k)  노이즈의 주파수별 파워 <- 노이즈를 봐야 안다

실제로는 둘 다 모르므로 이 필터는 **만들 수 없다.** 오직 상한을 재는 데만 쓴다.

## 어느 수준의 오라클인가

    (a) 여러 장 평균 스펙트럼        19.00 dB
    (b) 장마다 그 장의 참 스펙트럼   21.95 dB   <- 이것을 천장으로 쓴다

(a) 는 "이 데이터셋 전체에 맞춘 하나의 필터" 이고 (b) 는 "장마다 다시 맞춘 필터" 다.
상한으로는 (b) 가 맞다. 처음에 (a) 를 천장이라고 적었던 것은 과소평가였다.

## 재지 않는 것

주파수마다 정답에 맞춘 복소수를 곱하는 필터 `W = F·conj(G)/|G|²` 는 119 dB 를 낸다.
그러나 자유도가 이미지 화소 수와 같아 사실상 정답을 베끼는 것이라 상한이 아니다.
"선형" 이라는 말이 의미를 가지려면 필터가 데이터보다 단순해야 한다.

## 왜 이 숫자가 중요한가

    1일차 디노이저 → Wiener   21.06     학습 없이 조합만 한 것 중 최고
    오라클 선형 필터          21.95     정답을 알고 만든 최고의 선형 필터
    우리 모델                29.70

비학습 조합이 오라클에 **0.9 dB** 까지 닿아 있다. 선형으로 짜낼 것은 거의 다 짜냈고,
그 위 **7.75 dB 는 순전히 학습된 사전지식의 몫**이다. 3일차가 도구를 조립하는 문제가
아니라는 근거가 이 한 줄이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))
from metrics import calculate_psnr, calculate_ssim  # noqa: E402

from challenge import dipole_otf, forward  # noqa: E402
from day3_common import DEFAULT_DATA, load_val  # noqa: E402


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--n", type=int, default=30, help="val 장수")
    args = ap.parse_args()

    val = load_val(args.data, args.n)
    gt = np.stack([b.numpy().squeeze() for _, _, b in val]).astype(np.float64)
    g = np.stack([m.numpy().squeeze() for _, m, _ in val]).astype(np.float64)
    gc = np.stack([forward(x) for x in gt])
    noise = g - gc
    D = dipole_otf(gt.shape[-2:])

    F = np.abs(np.fft.fft2(gt)) ** 2
    N = np.abs(np.fft.fft2(noise)) ** 2

    def score(est):
        a = torch.from_numpy(est.astype(np.float32))[:, None]
        b = torch.from_numpy(gt.astype(np.float32))[:, None]
        return calculate_psnr(a, b).mean().item(), calculate_ssim(a, b).mean().item()

    def wiener(ratio):
        W = D / (D**2 + ratio)
        return np.real(np.fft.ifft2(np.fft.fft2(g) * W))

    print(f"선형 방법의 천장 — val {len(val)}장\n")
    print(f"{'필터':<40}{'PSNR':>9}{'SSIM':>9}")
    print("-" * 58)

    # 상수 K 하나 — 실제로 쓸 수 있는 위너
    best = (None, -1.0, 0.0)
    for K in np.logspace(-4, 0, 33):
        p, s = score(wiener(K))
        if p > best[1]:
            best = (K, p, s)
    print(f"{f'상수 K={best[0]:.3g} (val 에서 고름)':<40}{best[1]:>9.2f}{best[2]:>9.4f}")

    p, s = score(wiener(N.mean(0) / np.maximum(F.mean(0), 1e-20)))
    print(f"{'오라클 · 여러 장 평균 스펙트럼':<40}{p:>9.2f}{s:>9.4f}")

    p2, s2 = score(wiener(N / np.maximum(F, 1e-20)))
    print(f"{'오라클 · 장마다 참 스펙트럼  ← 천장':<40}{p2:>9.2f}{s2:>9.4f}")
    print("-" * 58)
    print(f"{'참고: 1일차 디노이저 → Wiener (test)':<40}{21.06:>9.2f}")
    print(f"{'참고: 우리 모델 (test)':<40}{29.70:>9.2f}{0.8828:>9.4f}")
    print()
    print(f"선형 오라클 {p2:.2f} dB 에 비학습 조합 21.06 이 {p2 - 21.06:.2f} dB 까지 닿아 있다.")
    print(f"그 위 {29.70 - p2:.2f} dB 는 학습된 사전지식의 몫이다.")


if __name__ == "__main__":
    main()
