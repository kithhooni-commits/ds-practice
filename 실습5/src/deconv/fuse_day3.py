"""여러 모델의 출력을 섞는다 — 학습 없이 얻는 마지막 점수.

## 왜 섞이는가

서로 다른 구조는 **서로 다른 방식으로 틀린다.** 전개형은 물리 제약에 묶여 있어
데이터에 충실한 쪽으로 틀리고, 방법 B(측정치 영역 디노이저 + Wiener)는 노이즈를
지우는 쪽으로 틀린다. 두 오차가 상관이 낮으면 평균이 둘 다보다 좋다.

    E[(w·a + (1-w)·b − f)²] = w²·Va + (1-w)²·Vb + 2w(1-w)·Cov

`Cov` 가 작으면 w=0.5 근처에서 분산이 반으로 준다. 상관이 1이면 아무 이득이 없다 —
그래서 **비슷한 모델끼리 섞으면 소용없고, 다른 구조여야 한다.**

## 무게는 val 에서 고른다

test 로 고르면 학습이 아니어도 test 를 쓴 것이다. `day3_common.tune` 을 그대로 쓴다.

SSIM 도 같이 본다. 평균은 국소 대비를 낮추는 방향이라 PSNR 은 오르는데 SSIM 이
떨어질 수 있다 — 그럴 땐 무게를 한쪽으로 기울이거나 융합을 쓰지 않는 게 맞다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))

from day3_common import DEFAULT_DATA, load_test, load_val, report, score, tune  # noqa: E402
from eval_day3 import load_net  # noqa: E402
from unrolled import data_consistency, self_ensemble  # noqa: E402


def wien(x, K):
    return data_consistency(torch.zeros_like(x), x,
                            torch.full((x.shape[0],), float(K), device=x.device))


SHIFT = False


def make_infer(ckpt: Path, device, se: bool):
    """체크포인트 하나를 '측정치 -> 복원' 함수로 만든다.

    --target measure 로 학습한 모델은 노이즈 없는 blur 를 내놓으므로 뒤에 Wiener 를
    한 번 더 걸어야 한다. K 는 그 모델마다 val 에서 따로 고른다.
    """
    net, label = load_net(ckpt, device)
    cfg_p = ckpt.parent.parent / "config.json"
    cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else {}
    base = (lambda g: self_ensemble(net, g, shifts=SHIFT)) if se else net
    return base, label, cfg.get("target", "label") == "measure"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", type=Path, required=True,
                    help="섞을 체크포인트들. 구조가 다를수록 이득이 크다")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--self-ensemble", action="store_true")
    ap.add_argument("--shift-ensemble", action="store_true",
                    help="self-ensemble 에 순환 이동을 더해 16x 로")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "day3_fuse.json")
    args = ap.parse_args()

    global SHIFT
    SHIFT = args.shift_ensemble
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
    val = load_val(args.data, args.n_val, device)
    test = load_test(args.data, 0, device)
    print(f"val {len(val)}장 (튜닝) · test {len(test)}장 (채점)\n")

    K_GRID = [float(k) for k in np.logspace(-4, 0.5, 19)]
    fns, labels = [], []
    for ck in args.ckpts:
        base, label, needs_wiener = make_infer(ck, device, args.self_ensemble)
        if needs_wiener:
            K = tune(val, [(f"K={k:.3g}", k) for k in K_GRID],
                     lambda k, b=base: (lambda g: wien(b(g), k)), f"{label} 뒤 Wiener K")
            fn = lambda g, b=base, k=K: wien(b(g), k)
            label += f" + Wiener K={K:.3g}"
        else:
            fn = base
        p, s = score(val, fn)
        print(f"  단독  {label:<44} val {p:6.2f} / {s:.4f}")
        fns.append(fn); labels.append(label)
    print()

    if len(fns) < 2:
        print("체크포인트가 하나뿐이라 섞을 것이 없다"); return

    # 무게 격자. 2개면 0~1 을 0.1 씩, 3개 이상이면 균등 + 한쪽 치우침
    if len(fns) == 2:
        grid = [(f"{w:.1f} : {1-w:.1f}", (w, 1 - w)) for w in np.arange(0, 1.01, 0.1)]
    else:
        grid = [("균등", tuple([1 / len(fns)] * len(fns)))]
        for i in range(len(fns)):
            w = [0.15] * len(fns); w[i] = 1 - 0.15 * (len(fns) - 1)
            grid.append((f"{labels[i][:18]} 우세", tuple(w)))

    def mix(ws):
        return lambda g: sum(w * f(g) for w, f in zip(ws, fns))

    best = tune(val, grid, mix, "융합 무게")
    p, s, rows = report(test, mix(best), f"융합 {tuple(round(w, 2) for w in best)}")

    print("\n" + "=" * 62)
    print(f"{'':<44}{'PSNR':>9}{'SSIM':>9}")
    print("-" * 62)
    for fn, label in zip(fns, labels):
        tp, ts = score(test, fn)
        print(f"{label[:43]:<44}{tp:>9.2f}{ts:>9.4f}")
    print(f"{'융합 (무게는 val 에서)':<44}{p:>9.2f}{s:>9.4f}")
    print("-" * 62)
    print("융합이 단독보다 나쁘면 두 모델이 같은 방식으로 틀리고 있다는 뜻이다.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"weights": list(best), "labels": labels, "psnr": p, "ssim": s},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 → {args.out}")


if __name__ == "__main__":
    main()
