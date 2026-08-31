"""모델이 오히려 손해를 보는 이미지가 있다. 통과시키는 게 이득인지 따져본다.

노이즈가 거의 없는 입력(σ≈0)에서는 이미 40~60 dB 인데, 모델은 학습한 대로
무언가를 지우려 들다가 디테일을 깎는다. 문제는 **테스트 때 σ 를 모른다**는 것.
그래서 이미지에서 추정한 σ̂ 로만 판단하는 게이트가 실제로 이득인지 확인한다.

σ̂ 은 Haar diagonal detail 의 MAD — deconv 쪽에서 쓰던 것과 같은 추정기다.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from data import make_loader, resolve_test_noisy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = Path(os.environ.get("DS_DATA", ROOT / "data" / "dataset"))


def sigma_mad(x: np.ndarray) -> float:
    a = x[: x.shape[0] // 2 * 2, : x.shape[1] // 2 * 2]
    b = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2)
    dd = (b[:, 0, :, 0] - b[:, 0, :, 1] - b[:, 1, :, 0] + b[:, 1, :, 1]) / 2.0
    return float(np.median(np.abs(dd)) / 0.6745)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    metrics_path = ROOT / "runs" / "0831-1015_dncnn_plus_main" / "test_metrics.json"
    rows = {r["file"]: r for r in json.loads(metrics_path.read_text(encoding="utf-8"))["rows"]}

    noisy_dir = resolve_test_noisy(DATA)
    loader, _ = make_loader([DATA / "test_label"], training_mode=False, batch=1, num_workers=0, noisy_dir=noisy_dir)

    recs = []
    for _, noisy, names in loader:
        name = names[0]
        r = rows[name]
        recs.append({
            "file": name, "noise": r["noise_type"],
            "sig": sigma_mad(noisy.numpy().squeeze()),
            "in_p": r["psnr_noisy"], "in_q": r["ssim_noisy"],
            "md_p": r["psnr_model"], "md_q": r["ssim_model"],
        })

    hurt = [r for r in recs if r["in_p"] > r["md_p"]]
    hurt.sort(key=lambda r: r["md_p"] - r["in_p"])
    print(f"모델이 손해인 이미지: {len(hurt)} / {len(recs)}\n")
    print(f"{'file':<10}{'noise':<17}{'σ̂':>8}{'입력':>9}{'모델':>9}{'손실':>8}")
    print("-" * 61)
    for r in hurt:
        print(f"{r['file'][3:11]:<10}{r['noise']:<17}{r['sig']:>8.4f}{r['in_p']:>9.2f}{r['md_p']:>9.2f}{r['md_p']-r['in_p']:>8.2f}")

    base_p = float(np.mean([r["md_p"] for r in recs]))
    base_q = float(np.mean([r["md_q"] for r in recs]))
    print(f"\n게이트 없음        PSNR {base_p:.3f}  SSIM {base_q:.4f}")

    print(f"\nσ̂ < τ 이면 입력을 그대로 내보낸다 (blind — 정답도 노이즈 종류도 안 봄)")
    print(f"{'τ':>8}{'통과 장수':>10}{'PSNR':>10}{'ΔPSNR':>9}{'SSIM':>10}{'ΔSSIM':>10}")
    print("-" * 57)
    best = None
    for tau in [0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03]:
        p = float(np.mean([r["in_p"] if r["sig"] < tau else r["md_p"] for r in recs]))
        q = float(np.mean([r["in_q"] if r["sig"] < tau else r["md_q"] for r in recs]))
        n = sum(1 for r in recs if r["sig"] < tau)
        star = ""
        if best is None or p > best[1]:
            best = (tau, p, q, n); star = ""
        print(f"{tau:>8.3f}{n:>10}{p:>10.3f}{p-base_p:>+9.3f}{q:>10.4f}{q-base_q:>+10.4f}")

    print(f"\n최선 τ = {best[0]:.3f} → PSNR {best[1]:.3f} ({best[1]-base_p:+.3f}), SSIM {best[2]:.4f} ({best[2]-base_q:+.4f}), {best[3]}장 통과")

    orc_p = float(np.mean([max(r["in_p"], r["md_p"]) for r in recs]))
    print(f"오라클(정답을 보고 매번 유리한 쪽 선택) PSNR {orc_p:.3f} ({orc_p-base_p:+.3f}) — 게이트가 도달할 수 있는 상한")


if __name__ == "__main__":
    main()
