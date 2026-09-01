"""커널을 모른 채 데이터에서 역필터를 학습한다.

## 무엇이 문제였나

디컨볼루션의 정답은 주파수마다 다른 상수를 곱하는 것이다. 배포 U-Net 은 25.59 dB
에 그쳤고, 그 곱셈을 직접 파라미터로 두고 **이미지 영역 MSE 로 경사하강**해도
26.25 dB 에서 멈춘다. 학습된 이득이 최대 4.8 인데 정답은 44074 다.

이유는 구조가 아니라 **조건수**다. |D| 가 작은 주파수는 이미지에 실린 에너지가
작아서 손실에 거의 기여하지 않는다. 경사하강이 "여기를 1000배 키워라"는 신호를
받지 못한다. U-Net 이 지는 것과 같은 원인이 손실 함수 쪽에도 있는 셈이다.

## 주파수마다 따로 풀면 된다

측정 모델이 `G(k) = D(k)·F(k)` 이므로 주파수끼리 섞이지 않는다. 각 k 에서 독립적인
1차원 최소제곱이고 닫힌 해가 있다.

    W(k) = Σᵢ Fᵢ(k)·conj(Gᵢ(k)) / Σᵢ |Gᵢ(k)|²

`D` 를 코드 어디에도 쓰지 않는다. `(측정치, 정답)` 쌍만 본다. 그런데도 학습된 W 가
1/D 와 상관계수 1.000000 으로 일치한다 — **데이터에서 커널의 역을 발견한 것**이다.

10장이면 109.8 dB, 200장이면 111.0 dB 로 해석적 답(111.04)에 붙는다.

## 노이즈가 있으면

이 추정은 자동으로 Wiener 해가 된다. G = D·F + N 이면

    E[F·conj(G)] / E[|G|²] = D·S / (D²·S + σ²) = (1/D)·D²/(D² + σ²/S)

즉 **K = σ²/S 인 Wiener 필터**가 데이터에서 그냥 나온다. K 를 손으로 고를 필요가
없다는 뜻이고, 3일차에서 그대로 쓸 수 있다.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))
from metrics import calculate_psnr, calculate_ssim  # noqa: E402

from challenge import forward  # noqa: E402

DEFAULT_DATA = Path(os.environ.get("DS_DATA", ROOT / "data" / "dataset"))


def learn_filter(pairs, device, chunk: int = 64) -> torch.Tensor:
    """(측정치, 정답) 쌍에서 주파수별 최소제곱 역필터를 구한다.

    pairs: (measure, label) numpy 배열 쌍의 이터러블. 전부 같은 크기여야 한다.
    """
    num = den = None
    buf_g, buf_f = [], []

    def flush():
        nonlocal num, den, buf_g, buf_f
        if not buf_g:
            return
        G = torch.fft.fft2(torch.from_numpy(np.stack(buf_g)).float().to(device))
        F = torch.fft.fft2(torch.from_numpy(np.stack(buf_f)).float().to(device))
        n = (F * G.conj()).sum(0)
        d = (G.abs() ** 2).sum(0)
        num = n if num is None else num + n
        den = d if den is None else den + d
        buf_g, buf_f = [], []

    for g, f in pairs:
        buf_g.append(g); buf_f.append(f)
        if len(buf_g) >= chunk:
            flush()
    flush()
    return (num / (den + 1e-20)).real


def apply_filter(measure: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(torch.fft.fft2(measure) * W).real


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--n-train", type=int, default=200, help="필터 학습에 쓸 train 장수")
    ap.add_argument("--noise", type=float, default=0.0, help="학습·평가 측정치에 얹을 노이즈 σ")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "learned_filter.json")
    ap.add_argument("--save-W", type=Path, default=None)
    args = ap.parse_args()

    ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if ok else "cpu")
    rng = np.random.default_rng(0)

    # ---- 학습: train 의 clean 으로 (측정치, 정답) 쌍을 만든다 ----
    tr = sorted(glob.glob(str(args.data / "train" / "*.npy")))[: args.n_train]

    def pairs():
        for f in tr:
            gt = np.load(f).astype(np.float64)
            g = forward(gt)
            if args.noise:
                g = g + rng.normal(0.0, args.noise, g.shape)
            yield g.astype(np.float32), gt.astype(np.float32)

    W = learn_filter(pairs(), device)
    print(f"필터 학습 완료 — train {len(tr)}장, 노이즈 σ={args.noise}")
    print(f"  |W| 범위 {W.abs().min():.3f} ~ {W.abs().max():.1f}")

    # ---- 평가: 배포된 test_deconv_only ----
    only = args.data / "test_deconv_only"
    meta = json.loads((only / "forward_meta.json").read_text(encoding="utf-8"))
    ps, ss = [], []
    for r in meta:
        g = np.load(only / r["file"]).astype(np.float32)
        gt = np.load(args.data / "test_label" / r["file"]).astype(np.float32)
        if args.noise:
            g = (g + rng.normal(0.0, args.noise, g.shape)).astype(np.float32)
        est = apply_filter(torch.from_numpy(g)[None, None].to(device), W)
        b = torch.from_numpy(gt)[None, None].to(device)
        ps.append(calculate_psnr(est, b).item())
        ss.append(calculate_ssim(est, b).item())

    print(f"\ntest_deconv_only {len(meta)}장")
    print(f"  PSNR {np.mean(ps):.2f}   SSIM {np.mean(ss):.4f}")

    # ---- 참고: 커널을 아는 해석적 답 ----
    from challenge import dipole_otf
    D = torch.from_numpy(dipole_otf((256, 256))).float().to(device)
    Wopt = 1.0 / torch.where(D.abs() < 1e-12, torch.full_like(D, 1e-12), D)
    corr = torch.corrcoef(torch.stack([W.flatten(), Wopt.flatten()]))[0, 1].item()
    print(f"  해석적 1/D 와 상관 {corr:.6f}  ← 커널을 안 보고 커널의 역을 찾았다")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"n_train": len(tr), "noise": args.noise, "psnr": float(np.mean(ps)),
         "ssim": float(np.mean(ss)), "corr_with_inverse_kernel": corr},
        ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_W:
        np.save(args.save_W, W.cpu().numpy())
    print(f"\n저장 → {args.out}")


if __name__ == "__main__":
    main()
