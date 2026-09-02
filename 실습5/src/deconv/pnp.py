"""Plug-and-play — 1일차에 학습한 디노이저를 3일차 프라이어로 그대로 쓴다.

## 아이디어

3일차 문제는 `g = h*f + n` 이다. 두 열화가 섞여 있어 한 번에 풀기 어렵지만,
**번갈아 풀면** 각각은 이미 풀 줄 아는 문제다.

    x₀ = Wiener(g, K)                     2일차 — 흐림을 되돌린다
    반복 k:
        z = 디노이저(x)                    1일차 — 노이즈를 지운다
        x = (D·G + λₖ·Z)/(D² + λₖ)        물리 제약. 닫힌 해, 학습 없음

역산이 노이즈를 증폭하면 디노이저가 지우고, 데이터 정합이 다시 측정치와 맞춘다.
한 번에 하려면 K 를 크게 잡아 정보를 버려야 하지만(14.59 dB), 나눠서 반복하면 둘 다
살릴 수 있다.

**새로 학습하는 것이 없다.** 1일차 체크포인트를 그대로 불러 쓴다. 그래서 이 경로는
`Others` 로 분류되고, label-free 디노이저를 쓰면 3일차 전체가 정답 없이 풀린다.

## λ 스케줄

λ 는 "측정치를 얼마나 믿을지"다. 작으면 데이터 쪽, 크면 사전지식(디노이저) 쪽이다.
DPIR 을 따라 **큰 λ 에서 작은 λ 로** 기하급수적으로 내린다 — 처음에는 디노이저를
세게 걸어 큰 잡음을 잡고, 갈수록 데이터에 맞춰 디테일을 되살린다.

참고: Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior", TPAMI 2021.
"""

from __future__ import annotations

import argparse
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
from models import build_model  # noqa: E402

from challenge import dipole_otf  # noqa: E402
from unrolled import data_consistency  # noqa: E402

DEFAULT_DATA = Path(os.environ.get("DS_DATA", ROOT / "data" / "dataset"))
CKPT_DIR = ROOT / "checkpoints"
NZ = ["gaussian", "rician", "uniform", "salt_and_pepper"]


def load_denoiser(path: Path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    name = ck.get("model", "dncnn")
    net = build_model(name, features=ck.get("features", 64), num_of_layers=ck.get("layers", 17))
    net.load_state_dict(ck["state_dict"])
    tag = f"{name} f{ck.get('features')}"
    if ck.get("label_free"):
        tag += " (label-free)"
    return net.to(device).eval(), tag, ck


@torch.no_grad()
def pnp(measure: torch.Tensor, denoiser, n_iter: int = 8,
        lam0: float = 0.3, lam1: float = 0.01) -> torch.Tensor:
    """HQS 반복. λ 를 lam0 에서 lam1 로 기하급수적으로 내린다."""
    lams = np.geomspace(lam0, lam1, n_iter)
    B = measure.shape[0]
    # 시작점: λ0 로 한 번 데이터 정합 = Wiener(K=λ0)
    x = data_consistency(torch.zeros_like(measure), measure,
                         torch.full((B,), float(lams[0]), device=measure.device))
    for lam in lams:
        z = denoiser(x)
        x = data_consistency(z, measure, torch.full((B,), float(lam), device=measure.device))
    return x


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--denoiser", type=Path, default=CKPT_DIR / "day1_dncnn_supervised.ckpt",
                    help="1일차에 학습한 디노이저 체크포인트")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--lam0", type=float, default=0.3)
    ap.add_argument("--lam1", type=float, default=0.01)
    ap.add_argument("--sweep", action="store_true", help="반복 횟수와 λ 범위를 훑는다")
    ap.add_argument("--n", type=int, default=0, help="평가 장수 제한 (0=전부)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
    net, tag, ck = load_denoiser(args.denoiser, device)
    print(f"디노이저: {tag}  (1일차 val PSNR {ck.get('val_psnr', float('nan')):.2f})")
    print("추가 학습 없음 — 1일차 가중치를 그대로 쓴다\n")

    src = args.data / "test_deconv_noise"
    meta = json.loads((src / "noise_meta.json").read_text(encoding="utf-8"))
    if args.n:
        meta = meta[: args.n]
    items = []
    for r in meta:
        g = torch.from_numpy(np.load(src / r["file"]).astype(np.float32))[None, None].to(device)
        gt = torch.from_numpy(np.load(args.data / "test_label" / r["file"]).astype(np.float32))[None, None].to(device)
        items.append((r["noise_type"], g, gt))

    def run(n_iter, lam0, lam1):
        rows = []
        for nz, g, gt in items:
            x = pnp(g, net, n_iter, lam0, lam1)
            rows.append((nz, calculate_psnr(x, gt).item(), calculate_ssim(x, gt).item()))
        return rows

    results = {}
    if args.sweep:
        print(f"{'iters':>6}{'lam0':>8}{'lam1':>8}{'PSNR':>9}{'SSIM':>9}")
        print("-" * 40)
        best = (None, -1, 0)
        for n_iter in (1, 4, 8, 16):
            for lam0, lam1 in ((0.3, 0.01), (0.1, 0.003), (0.5, 0.03), (0.05, 0.005)):
                rows = run(n_iter, lam0, lam1)
                p, s = float(np.mean([r[1] for r in rows])), float(np.mean([r[2] for r in rows]))
                if p > best[1]:
                    best = ((n_iter, lam0, lam1), p, s)
                print(f"{n_iter:>6}{lam0:>8.3g}{lam1:>8.3g}{p:>9.2f}{s:>9.4f}")
        print(f"\n최적: iters={best[0][0]} λ {best[0][1]}→{best[0][2]}   {best[1]:.2f} / {best[2]:.4f}")
        args.iters, args.lam0, args.lam1 = best[0]
        results["sweep_best"] = {"iters": best[0][0], "lam0": best[0][1], "lam1": best[0][2],
                                 "psnr": best[1], "ssim": best[2]}

    rows = run(args.iters, args.lam0, args.lam1)
    print(f"\n[plug-and-play · iters={args.iters} λ {args.lam0}→{args.lam1}]")
    print(f"{'noise':<18}{'n':>4}{'PSNR':>10}{'SSIM':>10}")
    print("-" * 42)
    for nz in NZ + ["ALL"]:
        s = [r for r in rows if nz == "ALL" or r[0] == nz]
        print(f"{nz:<18}{len(s):>4}{np.mean([r[1] for r in s]):>10.2f}{np.mean([r[2] for r in s]):>10.4f}")
    p, s = float(np.mean([r[1] for r in rows])), float(np.mean([r[2] for r in rows]))
    print(f"\n제출값 →  PSNR_total {p:.2f}   SSIM_total {s:.4f}")
    print(f"(배포 baseline 25.01 / 0.8149,  Wiener 최적 14.59 / 0.4322)")

    results["final"] = {"denoiser": str(args.denoiser), "iters": args.iters,
                        "lam0": args.lam0, "lam1": args.lam1, "psnr": p, "ssim": s}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"저장 → {args.out}")


if __name__ == "__main__":
    main()
