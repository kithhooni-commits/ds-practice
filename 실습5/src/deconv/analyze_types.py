"""어떤 **type 의 이미지**에 취약한가 — 배포 tips 가 요구한 분석.

노이즈 종류별 분석은 `noise_meta.json` 이 라벨을 주니 쉽다. 이미지 종류는 라벨이
없으므로 이미지 자체에서 뽑은 특징으로 가른다. test 100장의 변동폭:

    대비(std)      0.037 ~ 0.391   10.6배
    에지 밀도      0.013 ~ 0.146   11.7배
    무늬 조밀도    45.5  ~ 90.7     2.0배
    동적범위       0.140 ~ 0.967    6.9배

## 왜 이 특징들인가

**무늬 조밀도** = 주파수 에너지의 무게중심 반경. 조밀할수록 정보가 고주파에 몰리는데,
dipole 은 고주파에서 `|D|` 가 작아 역산이 노이즈를 크게 증폭한다. 즉 조밀한 무늬일수록
물리적으로 불리하다 — 이것이 이 문제에 특화된 예측 변수다.

**대비**와 **동적범위** = 신호의 세기. 노이즈 σ 는 이미지와 무관하게 뽑히므로, 대비가
낮은 이미지는 같은 σ 에서 실효 SNR 이 낮다.

**에지 밀도** = 구조의 복잡도. 평탄한 영역이 많으면 사전지식이 강하게 작동한다.

## 쓰는 법

    python analyze_types.py --per-image ../../figures/day3_per_image.json

`figures_day3.py` 가 남긴 장별 점수를 읽어 특징과 엮는다. 세 구간으로 나눈 표와
상관계수를 내고, 산점도를 그린다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))

from day3_common import DEFAULT_DATA, NZ  # noqa: E402

FEATURES = {
    "freq": "무늬 조밀도",
    "std": "대비",
    "edge": "에지 밀도",
    "dyn": "동적범위",
}


def image_features(gt: np.ndarray) -> dict:
    h, w = gt.shape
    F = np.abs(np.fft.fftshift(np.fft.fft2(gt)))
    F[h // 2, w // 2] = 0                       # DC 는 무늬가 아니다
    yy, xx = np.mgrid[-h // 2:h // 2, -w // 2:w // 2]
    rad = np.sqrt(yy**2 + xx**2)
    gy, gx = np.gradient(gt)
    return {"freq": float((F * rad).sum() / (F.sum() + 1e-12)),
            "std": float(gt.std()),
            "edge": float(np.sqrt(gy**2 + gx**2).mean()),
            "dyn": float(np.percentile(gt, 99) - np.percentile(gt, 1))}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--per-image", type=Path,
                    default=ROOT / "figures" / "day3_per_image.json",
                    help="figures_day3.py 가 남긴 장별 점수")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()

    if not args.per_image.exists():
        raise SystemExit(f"{args.per_image} 가 없다. figures_day3.py 를 먼저 돌릴 것")
    scores = json.loads(args.per_image.read_text(encoding="utf-8"))
    meta = json.loads((args.data / "test_deconv_noise" / "noise_meta.json")
                      .read_text(encoding="utf-8"))
    methods = [k for k in scores[0] if k not in ("noise", "sigma")]

    rows = []
    for r, sc in zip(meta, scores):
        gt = np.load(args.data / "test_label" / r["file"]).astype(np.float64)
        rows.append({**image_features(gt), "noise": r["noise_type"],
                     "sigma": r["sigma"], **{m: sc[m] for m in methods}})

    best = max(methods, key=lambda m: np.mean([r[m] for r in rows]))
    lab = best.replace(chr(10), " ")
    print(f"평가 대상: {lab}   전체 {np.mean([r[best] for r in rows]):.2f} dB\n")

    # ---- 특징별 3구간 ----
    print("=" * 70)
    print("  어떤 type 의 이미지에 취약한가 (특징별 3등분)")
    print("=" * 70)
    for key, name in FEATURES.items():
        v = np.array([r[key] for r in rows])
        q = np.quantile(v, [1 / 3, 2 / 3])
        g = np.digitize(v, q)
        cells = [np.mean([r[best] for r, gg in zip(rows, g) if gg == i]) for i in range(3)]
        span = cells[2] - cells[0]
        print(f"\n{name}  (낮음 → 높음)")
        for i, tag in enumerate(("하위 1/3", "중간", "상위 1/3")):
            n = int((g == i).sum())
            print(f"    {tag:<10}{v[g == i].mean():>9.4f}   {cells[i]:>7.2f} dB   ({n}장)")
        print(f"    차이 {span:+.2f} dB   상관계수 {np.corrcoef(v, [r[best] for r in rows])[0,1]:+.3f}")

    # ---- 가장 나쁜 열 장 ----
    print("\n" + "=" * 70)
    print("  가장 못 살린 10장 — 무엇이 공통인가")
    print("=" * 70)
    worst = sorted(rows, key=lambda r: r[best])[:10]
    print(f"{'noise':<18}{'σ':>7}{'PSNR':>8}{'조밀도':>9}{'대비':>8}{'에지':>8}")
    print("-" * 60)
    for r in worst:
        print(f"{r['noise']:<18}{r['sigma']:>7.3f}{r[best]:>8.2f}"
              f"{r['freq']:>9.1f}{r['std']:>8.4f}{r['edge']:>8.4f}")
    allm = {k: np.mean([r[k] for r in rows]) for k in ("sigma", "freq", "std", "edge")}
    wm = {k: np.mean([r[k] for r in worst]) for k in allm}
    print("-" * 60)
    print(f"{'하위 10장 평균':<18}{wm['sigma']:>7.3f}{'':>8}{wm['freq']:>9.1f}"
          f"{wm['std']:>8.4f}{wm['edge']:>8.4f}")
    print(f"{'전체 평균':<18}{allm['sigma']:>7.3f}{'':>8}{allm['freq']:>9.1f}"
          f"{allm['std']:>8.4f}{allm['edge']:>8.4f}")
    cnt = {n: sum(1 for r in worst if r["noise"] == n) for n in NZ}
    print(f"\n하위 10장의 노이즈 구성: " + "  ".join(f"{k} {v}" for k, v in cnt.items()))

    # ---- 그림 ----
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

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    COL = {"gaussian": "#4C78A8", "rician": "#F58518",
           "uniform": "#54A24B", "salt_and_pepper": "#E45756"}
    for a_, (key, name) in zip(ax, FEATURES.items()):
        for nz in NZ:
            s = [r for r in rows if r["noise"] == nz]
            a_.scatter([r[key] for r in s], [r[best] for r in s], s=20,
                       alpha=0.8, c=COL[nz], label=nz)
        v = np.array([r[key] for r in rows]); y = np.array([r[best] for r in rows])
        k, b = np.polyfit(v, y, 1)
        xs = np.linspace(v.min(), v.max(), 20)
        a_.plot(xs, k * xs + b, "k--", lw=1.2, alpha=0.7)
        a_.set_xlabel(name); a_.set_title(f"{name}   r = {np.corrcoef(v, y)[0,1]:+.3f}",
                                          fontsize=11)
        a_.spines[["top", "right"]].set_visible(False); a_.grid(alpha=0.25)
    ax[0].set_ylabel("PSNR (dB)")
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"어떤 type 의 이미지에 취약한가 — {lab}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "day3_image_types.png", dpi=140, bbox_inches="tight")
    print(f"\n저장 → {args.out / 'day3_image_types.png'}")


if __name__ == "__main__":
    main()
