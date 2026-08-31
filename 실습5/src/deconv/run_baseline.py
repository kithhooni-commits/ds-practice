"""A안 기준선 실행 — 합성 데이터로 바닥 성능과 하이퍼파라미터를 확정한다.

실제 학습셋(7,368장)과 test code 를 아직 받지 못했으므로, 여기서는 skimage 표본
이미지를 clean 으로 두고 forward 모델로 직접 corrupted 를 만든다. 데이터가 들어오면
`--images` 로 디렉터리만 갈아끼우면 된다.

주의 — DC 성분은 복원 불가능하다. D(0)=0 이라 평균값이 forward 에서 완전히 사라진다.
따라서 모든 지표는 복원 결과의 평균을 정답 평균에 맞춘 뒤 계산한다. 실제 evaluator 가
이 보정을 해 주지 않는다면 평균을 따로 추정해야 하고, 그건 별도 문제다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from baselines import cosmos, direct_inverse, tikhonov, tkd, wiener
from dipole import forward, make_orientations
from metrics import scores
from noise import estimate_noise

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"


def match_dc(est: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """복원 불가능한 DC 한 자유도만 정답에 맞춘다."""
    return est - est.mean() + ref.mean()


def load_clean(images: Path | None, size: int = 256) -> list[tuple[str, np.ndarray]]:
    from skimage import img_as_float, io
    from skimage.color import rgb2gray
    from skimage.transform import resize

    def prep(a: np.ndarray) -> np.ndarray:
        a = img_as_float(a)
        if a.ndim == 3:
            a = rgb2gray(a)
        return resize(a, (size, size), anti_aliasing=True)

    if images is None:
        from skimage import data

        return [(n, prep(getattr(data, n)())) for n in ("camera", "coins", "moon")]

    out = []
    for p in sorted(images.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            out.append((p.stem, prep(io.imread(p))))
    return out


def sweep_lambda(f: np.ndarray, g: np.ndarray, theta: float, lams: np.ndarray) -> tuple[float, float]:
    """검증용 λ 스윕. 정답을 쓰므로 학습셋에서만 할 수 있다."""
    best = (-np.inf, lams[0])
    for lam in lams:
        p = scores(f, match_dc(wiener(g, theta, lam), f))["psnr"]
        if p > best[0]:
            best = (p, lam)
    return best[1], best[0]


def main() -> None:
    # Windows 콘솔 기본 코드페이지에서 한글/기호가 깨지지 않도록
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=None, help="clean 이미지 디렉터리 (없으면 skimage 표본)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=0.01, help="합성 잡음 표준편차")
    ap.add_argument("--n-orient", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figures", action="store_true", help="before/after 그림 저장")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    thetas = make_orientations(args.n_orient)
    clean = load_clean(args.images, args.size)

    rows: list[dict] = []
    noise_report: list[dict] = []
    first = None

    for name, f in clean:
        gs = [forward(f, th, args.sigma, rng) for th in thetas]
        g0, th0 = gs[0], thetas[0]

        est = estimate_noise(g0, th0)
        est["image"] = name
        est["true_sigma"] = args.sigma
        noise_report.append(est)

        lam_star, _ = sweep_lambda(f, g0, th0, np.logspace(-4, 0, 25))

        recon = {
            "direct": direct_inverse(g0, th0),
            "tkd(t=0.15)": tkd(g0, th0, 0.15),
            f"wiener(λ={lam_star:.3g})": wiener(g0, th0, lam_star),
            "tikhonov(λ=1e-3)": tikhonov(g0, th0, 1e-3),
            f"cosmos({args.n_orient} orient)": cosmos(gs, thetas),
            f"cosmos({args.n_orient} orient, λ=1e-3)": cosmos(gs, thetas, 1e-3),
        }
        for m, r in recon.items():
            rows.append({"image": name, "method": m, **scores(f, match_dc(r, f))})

        if first is None:
            first = (name, f, g0, recon)

    # ---- 표 ----
    methods = [r["method"] for r in rows if r["image"] == rows[0]["image"]]
    print(f"\n합성 실험 — clean {len(clean)}장, {args.size}×{args.size}, σ={args.sigma}, orientation {args.n_orient}개\n")
    print(f"{'method':28s} {'PSNR(dB)':>10s} {'SSIM':>8s}")
    print("-" * 48)
    summary = []
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        p = float(np.mean([r["psnr"] for r in sel]))
        s = float(np.mean([r["ssim"] for r in sel]))
        summary.append({"method": m, "psnr": p, "ssim": s})
        print(f"{m:28s} {p:10.2f} {s:8.4f}")

    print("\n잡음 추정 (참값 σ = {:.4f})".format(args.sigma))
    print(f"{'image':10s} {'mad':>9s} {'flat':>9s} {'null':>9s} {'consensus':>10s}")
    print("-" * 50)
    for e in noise_report:
        print(f"{e['image']:10s} {e['mad']:9.4f} {e['flat']:9.4f} {e.get('null', float('nan')):9.4f} {e['consensus']:10.4f}")

    out = HERE.parent / "figures" / "baseline_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_image": rows, "noise": noise_report,
                               "config": vars(args) | {"images": str(args.images)}},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장 → {out}")

    if args.figures and first is not None:
        save_figures(*first, args)


def save_figures(name, f, g, recon, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Windows 한글 폰트 (없으면 기본값으로 둔다 — 라벨은 대부분 ASCII)
    for cand in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        try:
            matplotlib.font_manager.findfont(cand, fallback_to_default=False)
            plt.rcParams["font.family"] = cand
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    from dipole import dipole_kernel

    FIGDIR.mkdir(exist_ok=True)

    panels = [("clean", f), ("corrupted", g)] + [(k, match_dc(v, f)) for k, v in recon.items()]
    n = len(panels)
    cols = 4
    rows_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(3.2 * cols, 3.4 * rows_))
    for ax, (title, im) in zip(axes.ravel(), panels):
        vmin, vmax = (im.min(), im.max()) if title == "corrupted" else (0, 1)
        ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
        t = title if title in ("clean", "corrupted") else f"{title}\nPSNR {scores(f, im)['psnr']:.2f} dB"
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{name} — σ={args.sigma}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
    p1 = FIGDIR / "baseline_panels.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # dipole 커널과 0 영역
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    D0 = np.fft.fftshift(dipole_kernel(f.shape, 0.0))
    axes[0].imshow(D0, cmap="coolwarm", vmin=-2 / 3, vmax=1 / 3)
    axes[0].set_title("D(k), θ=0°", fontsize=9)
    axes[1].imshow(np.abs(D0) < 0.05, cmap="gray")
    axes[1].set_title("|D| < 0.05 — 정보가 사라지는 자리", fontsize=9)
    den = sum(np.fft.fftshift(dipole_kernel(f.shape, t)) ** 2 for t in make_orientations(args.n_orient))
    axes[2].imshow(den < 0.05 ** 2, cmap="gray")
    axes[2].set_title(f"orientation {args.n_orient}개 합산 후 남은 빈칸", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    p2 = FIGDIR / "dipole_nullspace.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    print(f"그림 저장 → {p1}\n           {p2}")


if __name__ == "__main__":
    main()
