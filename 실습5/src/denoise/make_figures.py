"""발표용 그림 생성.

노이즈 종류마다 대표 이미지 한 장을 골라 4개 패널로 낸다.

    corrupted (입력) → restored (모델) → |error| (모델 − 정답) → ground truth

error map 을 같이 두는 이유: PSNR 숫자 하나로는 "어디서" 틀렸는지 안 보인다.
오차가 경계선에 몰려 있는지, 평탄한 곳에 흩어져 있는지가 다음 수를 결정한다.

대표 이미지는 그 종류의 σ 중앙값에 가장 가까운 것을 고른다. 제일 잘 나온 걸
고르면 발표가 거짓말이 된다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from data import make_loader, resolve_test_noisy
from evaluate import load_net, pick_device, predict
from metrics import calculate_psnr, calculate_ssim

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_DATA = Path(os.environ.get("DS_DATA", ROOT / "data" / "dataset"))
FIGDIR = ROOT / "figures"
NOISE_ORDER = ["gaussian", "rician", "uniform", "salt_and_pepper"]
NOISE_KO = {
    "gaussian": "Gaussian",
    "rician": "Rician",
    "uniform": "Uniform",
    "salt_and_pepper": "Salt & Pepper",
}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cand in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        try:
            matplotlib.font_manager.findfont(cand, fallback_to_default=False)
            plt.rcParams["font.family"] = cand
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--device", default=None)
    ap.add_argument("--self-ensemble", action="store_true", default=True)
    ap.add_argument("--no-self-ensemble", dest="self_ensemble", action="store_false")
    ap.add_argument("--out", type=Path, default=FIGDIR)
    args = ap.parse_args()

    plt = setup_mpl()
    args.out.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    net, _ = load_net(args.ckpt, device)

    noisy_dir = resolve_test_noisy(args.data)
    meta = {r["file"]: r for r in json.loads((noisy_dir / "noise_meta.json").read_text())}

    # 종류별로 σ 중앙값에 가장 가까운 파일 고르기
    picks: dict[str, str] = {}
    for nz in NOISE_ORDER:
        sub = [(f, m["sigma"]) for f, m in meta.items() if m["noise_type"] == nz]
        med = float(np.median([s for _, s in sub]))
        picks[nz] = min(sub, key=lambda t: abs(t[1] - med))[0]
    want = {v: k for k, v in picks.items()}

    loader, _ = make_loader(
        [args.data / "test_label"], training_mode=False, batch=1, num_workers=0, noisy_dir=noisy_dir
    )

    shots: dict[str, dict] = {}
    with torch.no_grad():
        for label, noisy, names in loader:
            name = names[0]
            if name not in want:
                continue
            label, noisy = label.to(device), noisy.to(device)
            out = predict(net, noisy, args.self_ensemble)
            shots[want[name]] = {
                "file": name,
                "sigma": meta[name]["sigma"],
                "gt": label.cpu().numpy().squeeze(),
                "in": noisy.cpu().numpy().squeeze(),
                "out": out.cpu().numpy().squeeze(),
                "psnr_in": calculate_psnr(noisy, label).item(),
                "ssim_in": calculate_ssim(noisy, label).item(),
                "psnr_out": calculate_psnr(out, label).item(),
                "ssim_out": calculate_ssim(out, label).item(),
            }

    # ---- 종류별 4패널 ----
    for nz, s in shots.items():
        gt = s["gt"]
        vmax = float(np.percentile(gt, 99.5))
        err = np.abs(s["out"] - gt)
        # 모델 오차 자체의 분포로 스케일을 잡는다. 입력 오차 기준으로 잡으면
        # (임펄스 때문에 최대가 1 근처) 화면이 통째로 까매져서 아무것도 안 보인다.
        # 입력과의 직접 비교는 error_maps.png 에서 같은 스케일로 따로 한다.
        emax = max(float(np.percentile(err, 99.5)), 1e-4)

        fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.3))
        panels = [
            ("corrupted (입력)", s["in"], "gray", 0, vmax, f"PSNR {s['psnr_in']:.2f} dB / SSIM {s['ssim_in']:.4f}"),
            ("restored (모델)", s["out"], "gray", 0, vmax, f"PSNR {s['psnr_out']:.2f} dB / SSIM {s['ssim_out']:.4f}"),
            ("|error| = |모델 - 정답|", err, "magma", 0, emax, f"평균 {err.mean():.4f} / 최대 {err.max():.3f}"),
            ("ground truth (정답)", gt, "gray", 0, vmax, "—"),
        ]
        for ax, (title, im, cmap, lo, hi, sub) in zip(axes, panels):
            h = ax.imshow(im, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f"{title}\n{sub}", fontsize=10.5, pad=8)
            ax.axis("off")
            if cmap == "magma":
                fig.colorbar(h, ax=ax, fraction=0.046, pad=0.02)

        fig.suptitle(
            f"{NOISE_KO[nz]}   σ = {s['sigma']:.4f}   ·   {s['file'][3:11]}"
            f"   ·   error map 은 모델 오차 자체 분포로 스케일 (99.5 percentile)",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        p = args.out / f"panel_{nz}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("저장:", p)

    # ---- error map 비교: 입력 오차 vs 모델 오차 (같은 스케일) ----
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.6))
    for c, nz in enumerate(NOISE_ORDER):
        s = shots[nz]
        gt = s["gt"]
        e_in, e_out = np.abs(s["in"] - gt), np.abs(s["out"] - gt)
        emax = float(np.percentile(e_in, 99.5))
        for r, (lab, e) in enumerate((("입력 오차", e_in), ("모델 오차", e_out))):
            ax = axes[r, c]
            h = ax.imshow(e, cmap="magma", vmin=0, vmax=emax)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"{NOISE_KO[nz]}\nσ={s['sigma']:.3f}", fontsize=11, pad=6)
            ax.text(0.02, 0.97, f"{lab}  평균 {e.mean():.4f}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=9, color="white",
                    bbox=dict(facecolor="black", alpha=0.55, pad=2.5, edgecolor="none"))
        fig.colorbar(h, ax=axes[:, c], fraction=0.028, pad=0.015)
    fig.suptitle("오차 지도 — 위: 아무것도 안 했을 때, 아래: 모델 통과 후 (열마다 같은 스케일)", fontsize=12.5)
    p = args.out / "error_maps.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("저장:", p)

    # ---- 노이즈 종류별 요약 막대 ----
    metrics = json.loads((args.ckpt.parent.parent / "test_metrics_se.json").read_text(encoding="utf-8"))["rows"]
    methods = [("psnr_noisy", "복원 안 함"), ("psnr_mean", "mean 3×3"), ("psnr_median", "median 3×3"),
               ("psnr_adaptive", "adaptive 5×5"), ("psnr_model", "제안 모델")]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    xs = np.arange(len(NOISE_ORDER) + 1)
    w = 0.16
    colors = ["#B0846F", "#B9C2BD", "#9FAFA7", "#8598A0", "#0B6E5E"]
    for i, (key, lab) in enumerate(methods):
        vals = []
        for nz in NOISE_ORDER:
            sub = [r for r in metrics if r["noise_type"] == nz]
            vals.append(np.mean([r[key] for r in sub]))
        vals.append(np.mean([r[key] for r in metrics]))
        bars = ax.bar(xs + (i - 2) * w, vals, w, label=lab, color=colors[i])
        if i == len(methods) - 1:
            ax.bar_label(bars, fmt="%.1f", fontsize=8.5, padding=1.5)
    ax.axhline(30.510, ls="--", lw=1.2, color="#8C4A6B")
    ax.text(len(xs) - 0.45, 30.9, "배포 기준선 30.51", fontsize=9, color="#8C4A6B", ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([NOISE_KO[n] for n in NOISE_ORDER] + ["전체"])
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(10, 40)
    ax.legend(ncol=5, fontsize=9, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    p = args.out / "summary_bars.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("저장:", p)


if __name__ == "__main__":
    main()
