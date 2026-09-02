"""3일차 전체 실행 — val 에서 튜닝하고 test 로 채점한다.

**test 는 채점에만 쓴다.** K·λ 같은 하이퍼파라미터는 전부 val 에서 고른다.
학습이 아니더라도 하이퍼파라미터를 test 로 고르면 test 를 쓴 것이다.

## 재는 것

    (a) 입력 그대로
    (b) Wiener 단독                        2일차 답. K 는 val 에서
    (c) 전처리 → Wiener                     배포 방법 B 의 순서. mean/median/adaptive
                                            그리고 1일차 디노이저(supervised / label-free)
    (d) Wiener → 1일차 디노이저             순서를 뒤집은 대조군
    (e) plug-and-play                       역산 ↔ 디노이저 반복

(c) 가 맞는 순서다. 노이즈가 흐림 **뒤에** 붙었으므로 측정치 위에서는 백색이고,
거기서 지워야 디노이저가 잘한다. Wiener 를 먼저 걸면 잡음이 주파수마다 다르게
증폭되고 X자 방향으로 상관돼 디노이저가 본 적 없는 종류가 된다.
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
from filters import adaptive_filter, mean_filter, median_filter  # noqa: E402

from day3_common import DEFAULT_DATA, load_test, load_val, report, score, tune  # noqa: E402
from pnp import load_denoiser, pnp  # noqa: E402
from unrolled import data_consistency  # noqa: E402

CKPT = ROOT / "checkpoints"
K_GRID = [float(k) for k in np.logspace(-4, 0.5, 19)]   # 배포 노트북과 같은 격자


def wien(x, K):
    return data_consistency(torch.zeros_like(x), x,
                            torch.full((x.shape[0],), float(K), device=x.device))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "day3_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
    val = load_val(args.data, args.n_val, device)
    test = load_test(args.data, args.n_test, device)
    print(f"val {len(val)}장 (튜닝 전용)  ·  test {len(test)}장 (채점 전용)\n")

    dens = {}
    for p in sorted(CKPT.glob("day1_*.ckpt")):
        net, tag, ck = load_denoiser(p, device)
        dens["label-free" if ck.get("label_free") else "supervised"] = net
    print("1일차 디노이저:", ", ".join(dens), "\n")

    res, summary = {}, []

    def add(label, fn, category):
        p, s, rows = report(test, fn, label)
        summary.append((label, p, s, category))
        res[label] = {"psnr": p, "ssim": s, "category": category,
                      "per_noise": {nz: float(np.mean([r[1] for r in rows if r[0] == nz]))
                                    for nz in {r[0] for r in rows}}}
        print()

    # (a) 입력
    add("입력 (blur + noise)", lambda g: g, "—")

    # (b) Wiener 단독 — K 는 val 에서
    K = tune(val, [(f"K={k:.3g}", k) for k in K_GRID], lambda k: (lambda g: wien(g, k)),
             "Wiener K")
    add(f"Wiener 단독 (K={K:.3g})", lambda g: wien(g, K), "Others")
    res["wiener_K"] = K

    # (c) 전처리 → Wiener
    PRE = {"mean 3×3": lambda x: mean_filter(x, 3), "median 3×3": lambda x: median_filter(x, 3),
           "adaptive 5×5": lambda x: adaptive_filter(x, 5),
           **{f"1일차 {k}": v for k, v in dens.items()}}
    for name, pre in PRE.items():
        Kp = tune(val, [(f"K={k:.3g}", k) for k in K_GRID],
                  lambda k, pre=pre: (lambda g: wien(pre(g), k)), f"{name} → Wiener 의 K")
        cat = "Supervised" if name == "1일차 supervised" else (
            "Self-supervised" if name == "1일차 label-free" else "Others")
        add(f"{name} → Wiener (K={Kp:.3g})", lambda g, pre=pre, k=Kp: wien(pre(g), k), cat)

    # (d) 순서를 뒤집은 대조군
    for key, net in dens.items():
        Kr = tune(val, [(f"K={k:.3g}", k) for k in K_GRID],
                  lambda k, net=net: (lambda g: net(wien(g, k))), f"Wiener → 1일차 {key} 의 K")
        add(f"Wiener → 1일차 {key} (K={Kr:.3g})", lambda g, net=net, k=Kr: net(wien(g, k)),
            "Supervised" if key == "supervised" else "Self-supervised")

    # (e) plug-and-play
    cfgs = [((n, a, b), (n, a, b)) for n in (2, 4, 8)
            for a, b in ((0.5, 0.03), (0.3, 0.01), (0.1, 0.01))]
    for key, net in dens.items():
        cfg = tune(val, cfgs, lambda c, net=net: (lambda g: pnp(g, net, *c)),
                   f"plug-and-play [{key}]")
        add(f"plug-and-play · 1일차 {key} (iters={cfg[0]}, λ {cfg[1]}→{cfg[2]})",
            lambda g, net=net, c=cfg: pnp(g, net, *c),
            "Supervised" if key == "supervised" else "Self-supervised")

    # ---- 요약 ----
    print("=" * 76)
    print(f"{'방법':<46}{'PSNR':>9}{'SSIM':>9}{'분류':>12}")
    print("-" * 76)
    for label, p, s, cat in sorted(summary, key=lambda x: -x[1]):
        print(f"{label:<46}{p:>9.2f}{s:>9.4f}{cat:>12}")
    print("-" * 76)
    print(f"{'배포 baseline (End2End U-Net, 학습됨)':<46}{25.01:>9.2f}{0.8149:>9.4f}{'Supervised':>12}")
    print("\n모든 K·λ 는 val 에서 골랐다. test 는 채점에만 썼다.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 → {args.out}")


if __name__ == "__main__":
    main()
