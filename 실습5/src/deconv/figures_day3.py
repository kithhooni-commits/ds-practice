"""3일차 발표용 그림 — 배포 안내가 요구한 것을 그대로 담는다.

  "train 을 위해서 사용한 synthetic image, 결과 test 이미지는 꼭 visualize 해 보세요.
   difference map, detail 한 부분은 zoom-in 을 해 주세요."
  "구축한 프레임워크가 어떤 노이즈에 취약한지, 어떤 type 의 이미지에 취약한지 분석해 보세요."

만드는 것

  1. forward_chain    clean → dipole blur → +noise. 학습 쌍이 어떻게 만들어지는가
  2. methods_grid     노이즈 종류별 × 방법별 복원 결과 + difference map
  3. zoom             한 장을 골라 원본/복원/차이를 확대 (세부 손실이 어디서 나는가)
  4. weakness         노이즈 종류별·이미지별 성능 분포. 어디에 취약한가

대표 이미지는 그 종류의 σ **중앙값**에 가장 가까운 것을 고른다. 제일 잘 나온 걸
고르면 발표가 거짓말이 된다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))
from filters import median_filter  # noqa: E402
from metrics import calculate_psnr, calculate_ssim  # noqa: E402

from challenge import forward  # noqa: E402
from day3_common import DEFAULT_DATA, NZ  # noqa: E402
from pnp import load_denoiser  # noqa: E402
from unrolled import data_consistency  # noqa: E402

FIG = ROOT / "figures"
CKPT = ROOT / "checkpoints"
NOISE_KO = {"gaussian": "Gaussian", "rician": "Rician",
            "uniform": "Uniform", "salt_and_pepper": "Salt & Pepper"}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for c in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        try:
            matplotlib.font_manager.findfont(c, fallback_to_default=False)
            plt.rcParams["font.family"] = c
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def wien(x, K):
    return data_consistency(torch.zeros_like(x), x,
                            torch.full((x.shape[0],), float(K), device=x.device))


def sc(e, gt):
    return calculate_psnr(e, gt).item(), calculate_ssim(e, gt).item()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--ckpt", type=Path, default=None, help="학습한 모델 (있으면 비교에 넣는다)")
    ap.add_argument("--post-wiener", type=float, default=None, help="--target measure 모델용")
    ap.add_argument("--wiener-K", type=float, default=0.03, help="val 에서 고른 K")
    ap.add_argument("--out", type=Path, default=FIG)
    args = ap.parse_args()

    plt = setup_mpl()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")

    src = args.data / "test_deconv_noise"
    meta = json.loads((src / "noise_meta.json").read_text(encoding="utf-8"))

    # 종류별 σ 중앙값에 가장 가까운 파일
    picks = {}
    for nz in NZ:
        sub = [(r["file"], r["sigma"]) for r in meta if r["noise_type"] == nz]
        med = float(np.median([s for _, s in sub]))
        picks[nz] = min(sub, key=lambda t: abs(t[1] - med))

    # ---------------------------------------------------------- 방법들
    dens = {}
    for p in sorted(CKPT.glob("day1_*.ckpt")):
        net, _, ck = load_denoiser(p, device)
        dens["label-free" if ck.get("label_free") else "supervised"] = net

    methods = {
        "측정치 (입력)": lambda g: g,
        f"Wiener 단독\nK={args.wiener_K:.3g}": lambda g: wien(g, args.wiener_K),
        f"median → Wiener\nK={args.wiener_K:.3g}": lambda g: wien(median_filter(g, 3), args.wiener_K),
    }
    if "supervised" in dens:
        methods["1일차 디노이저 → Wiener"] = lambda g: wien(dens["supervised"](g), args.wiener_K)
    if args.ckpt and args.ckpt.exists():
        from eval_day3 import load_net
        net, label = load_net(args.ckpt, device)
        K = args.post_wiener
        methods[f"학습 모델\n{label.split()[0]}"] = (
            lambda g: wien(net(g), K)) if K else (lambda g: net(g))

    # ---------------------------------------------------------- 1. forward chain
    nz0, (f0, s0) = "gaussian", picks["gaussian"]
    gt = np.load(args.data / "test_label" / f0).astype(np.float64)
    blur = forward(gt)
    noisy = np.load(src / f0).astype(np.float64)

    fig, ax = plt.subplots(1, 4, figsize=(15.5, 4.3))
    panels = [("clean (정답)", gt, 0, 1),
              ("dipole blur\ng = h * f", blur, blur.min(), blur.max()),
              (f"+ noise ({NOISE_KO[nz0]}, σ={s0:.3f})\ng = h * f + n", noisy, blur.min(), blur.max()),
              ("noise 만\n(g_noisy - g_blur)", noisy - blur, -3 * s0, 3 * s0)]
    for a, (t, im, lo, hi) in zip(ax, panels):
        h = a.imshow(im, cmap="gray" if "noise 만" not in t else "coolwarm", vmin=lo, vmax=hi)
        a.set_title(t, fontsize=10.5); a.axis("off")
        if "noise 만" in t:
            fig.colorbar(h, ax=a, fraction=0.046)
    fig.suptitle("3일차 열화 사슬 — 노이즈가 흐림 **뒤에** 붙는다 "
                 "(그래서 측정치 위에서는 백색이고, 거기서 지워야 한다)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(args.out / "day3_forward_chain.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("저장:", args.out / "day3_forward_chain.png")

    # ---------------------------------------------------------- 2. 방법 × 노이즈 격자
    ncol = 1 + len(methods)
    fig, axes = plt.subplots(len(NZ), ncol, figsize=(2.9 * ncol, 3.15 * len(NZ)))
    for r, nz in enumerate(NZ):
        f, s = picks[nz]
        gtn = np.load(args.data / "test_label" / f).astype(np.float32)
        gn = np.load(src / f).astype(np.float32)
        gt_t = torch.from_numpy(gtn)[None, None].to(device)
        g_t = torch.from_numpy(gn)[None, None].to(device)
        vmax = float(np.percentile(gtn, 99.5))
        axes[r, 0].imshow(gtn, cmap="gray", vmin=0, vmax=vmax)
        axes[r, 0].set_ylabel(f"{NOISE_KO[nz]}\nσ={s:.3f}", fontsize=10)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("clean (정답)", fontsize=10)
        with torch.no_grad():
            for c, (name, fn) in enumerate(methods.items(), start=1):
                e = fn(g_t)
                p, ss = sc(e, gt_t)
                axes[r, c].imshow(e.cpu().numpy().squeeze(), cmap="gray", vmin=0, vmax=vmax)
                if r == 0:
                    axes[r, c].set_title(name, fontsize=10)
                # 점수는 칸 안에 적는다 — 행 사이에 두면 어느 행 것인지 헷갈린다
                axes[r, c].text(0.03, 0.03, f"{p:.2f} dB / {ss:.3f}", fontsize=9,
                                transform=axes[r, c].transAxes, color="w", va="bottom",
                                bbox=dict(fc="k", alpha=0.55, pad=1.8, lw=0))
                axes[r, c].axis("off")
    fig.suptitle("노이즈 종류별 복원 결과 — σ 중앙값에 가까운 대표 이미지", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out / "day3_methods_grid.png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("저장:", args.out / "day3_methods_grid.png")

    # ---------------------------------------------------------- 3. difference map + zoom
    f, s = picks["gaussian"]
    gtn = np.load(args.data / "test_label" / f).astype(np.float32)
    gn = np.load(src / f).astype(np.float32)
    gt_t = torch.from_numpy(gtn)[None, None].to(device)
    g_t = torch.from_numpy(gn)[None, None].to(device)
    zy, zx, zs = 96, 96, 72   # 확대할 자리

    # 맨 왼쪽은 정답. 확대를 정답과 나란히 놔야 무엇을 잃었는지 보인다
    names = ["정답 (GT)"] + list(methods)[1:]
    fig, axes = plt.subplots(3, len(names), figsize=(3.4 * len(names), 10.6))
    axes = np.asarray(axes).reshape(3, len(names))
    with torch.no_grad():
        outs = {"정답 (GT)": gtn,
                **{n: methods[n](g_t).cpu().numpy().squeeze() for n in list(methods)[1:]}}
    emax = float(np.percentile(np.abs(np.stack([outs[n] for n in names[1:]]) - gtn), 99.5))
    vmax = float(np.percentile(gtn, 99.5))
    for c, n in enumerate(names):
        e = outs[n]
        axes[0, c].imshow(e, cmap="gray", vmin=0, vmax=vmax)
        if c == 0:
            axes[0, c].set_title(n, fontsize=10)
            axes[1, c].text(0.5, 0.5, "기준\n(차이 = 0)", ha="center", va="center", fontsize=11)
        else:
            p, ss = sc(torch.from_numpy(e)[None, None].to(device), gt_t)
            axes[0, c].set_title(f"{n}\n{p:.2f} dB / {ss:.3f}", fontsize=9.5)
            h = axes[1, c].imshow(np.abs(e - gtn), cmap="magma", vmin=0, vmax=emax)
            axes[1, c].set_title(f"|difference|  평균 {np.abs(e - gtn).mean():.4f}", fontsize=9.5)
        axes[2, c].imshow(e[zy:zy + zs, zx:zx + zs], cmap="gray", vmin=0, vmax=vmax)
        axes[2, c].set_title(f"zoom {zs}×{zs}", fontsize=9.5)
        for r in range(3):
            axes[r, c].axis("off")
    fig.suptitle(f"difference map 과 zoom — {NOISE_KO['gaussian']} σ={s:.3f} · "
                 f"위: 복원, 가운데: |복원-정답| (0-{emax:.2f} 공통 스케일), 아래: 확대",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(args.out / "day3_diff_zoom.png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("저장:", args.out / "day3_diff_zoom.png")

    # ---------------------------------------------------------- 4. 취약점 분석
    rows = []
    with torch.no_grad():
        for r in meta:
            gtn = torch.from_numpy(np.load(args.data / "test_label" / r["file"]).astype(np.float32))[None, None].to(device)
            gn = torch.from_numpy(np.load(src / r["file"]).astype(np.float32))[None, None].to(device)
            row = {"noise": r["noise_type"], "sigma": r["sigma"]}
            for n, fn in methods.items():
                row[n] = sc(fn(gn), gtn)[0]
            rows.append(row)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    names_all = list(methods)
    x = np.arange(len(NZ))
    w = 0.8 / len(names_all)
    for i, n in enumerate(names_all):
        vals = [np.mean([r[n] for r in rows if r["noise"] == nz]) for nz in NZ]
        ax[0].bar(x + (i - len(names_all) / 2) * w, vals, w, label=n.replace("\n", " "))
    ax[0].set_xticks(x); ax[0].set_xticklabels([NOISE_KO[n] for n in NZ])
    ax[0].set_ylabel("PSNR (dB)")
    ax[0].set_ylim(0, max(b.get_height() for b in ax[0].patches) * 1.42)  # 범례 자리
    ax[0].legend(fontsize=8, ncol=2, frameon=False, loc="upper center")
    ax[0].set_title("어떤 노이즈에 취약한가", fontsize=11)
    ax[0].spines[["top", "right"]].set_visible(False); ax[0].grid(axis="y", alpha=0.25)

    best_name = max(names_all[1:], key=lambda n: np.mean([r[n] for r in rows]))
    for nz in NZ:
        sub = [r for r in rows if r["noise"] == nz]
        ax[1].scatter([r["sigma"] for r in sub], [r[best_name] for r in sub],
                      s=22, alpha=0.8, label=NOISE_KO[nz])
    ax[1].set_xlabel("노이즈 세기 σ"); ax[1].set_ylabel("PSNR (dB)")
    ax[1].set_title(f"σ 가 커질수록 — {best_name.replace(chr(10), ' ')}", fontsize=11)
    ax[1].legend(fontsize=9, frameon=False)
    ax[1].spines[["top", "right"]].set_visible(False); ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "day3_weakness.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("저장:", args.out / "day3_weakness.png")

    (args.out / "day3_per_image.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", args.out / "day3_per_image.json")


if __name__ == "__main__":
    main()
