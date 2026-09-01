"""배포된 test_deconv_only / test_deconv_multi 평가.

## 두 세트의 차이

    test_deconv_only    100장, 전부 B0 = (0, 1) — 방향이 하나로 고정
    test_deconv_multi   100장, 25장씩 4방향 (0° / 45° / 90° / 135°)
                        **폴더끼리 이미지가 겹치지 않는다** — COSMOS 가 아니다.
                        같은 장면을 여러 방향으로 찍은 게 아니라, 이미지마다
                        커널 방향이 다른 것이다.

## 그래서 무엇이 문제인가

방향을 틀리면 복원이 입력보다 나빠진다 (0° 커널을 전부에 적용하면 −11 dB).
`forward_meta.json` 에 방향이 적혀 있지만, 채점 세트에 없을 수도 있으므로
**측정치만 보고 추정**하는 경로를 같이 만든다 (`estimate_b0`).

## K 를 얼마나 낮출 것인가

노이즈가 없으니 K→0 이 최선처럼 보이지만, 그 답은 **방향 오차에 극도로 취약**하다.
0.01° 만 틀려도 107 dB 가 54 dB 로 떨어진다. 게다가 45°/135° 는 방향이 정확해도
K=1e-12 에서 25 dB 밖에 안 나온다 — 격자와 0 영역이 맞물려 1/D 가 수치적으로
폭발하기 때문이다. 적당한 K 가 두 문제를 동시에 눌러 준다.
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

from challenge import estimate_b0, wiener  # noqa: E402

DEFAULT_DATA = Path(os.environ.get("DS_DATA", ROOT / "data" / "dataset"))
GROUPS = ["0_deg", "45_deg", "90_deg", "135_deg"]


def sc(est: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    a = torch.from_numpy(est[None, None]).float()
    b = torch.from_numpy(gt[None, None]).float()
    return calculate_psnr(a, b).item(), calculate_ssim(a, b).item()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--n", type=int, default=0, help="방향별 장수 제한 (0=전부)")
    ap.add_argument("--estimate", action="store_true", help="메타 대신 측정치에서 방향을 추정한다")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "deconv_multi_results.json")
    args = ap.parse_args()

    Ks = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-12]
    results: dict = {}

    # ------------------------------------------------------ test_deconv_only
    only = args.data / "test_deconv_only"
    if only.exists():
        meta = json.loads((only / "forward_meta.json").read_text(encoding="utf-8"))
        if args.n:
            meta = meta[: args.n * 4]
        print(f"test_deconv_only — {len(meta)}장, B0 고정 (0, 1)\n")
        print(f"{'K':>10}{'PSNR':>10}{'SSIM':>9}")
        print("-" * 29)
        results["only"] = {}
        for K in Ks:
            ps, ss = [], []
            for r in meta:
                g = np.load(only / r["file"]).astype(np.float64)
                gt = np.load(args.data / "test_label" / r["file"]).astype(np.float64)
                p, s = sc(wiener(g, K, tuple(r["B0_dir"])), gt)
                ps.append(p); ss.append(s)
            results["only"][f"{K:.0e}"] = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))}
            print(f"{K:>10.0e}{np.mean(ps):>10.2f}{np.mean(ss):>9.4f}")

    # ------------------------------------------------------ test_deconv_multi
    multi = args.data / "test_deconv_multi"
    if not multi.exists():
        return
    meta = json.loads((multi / "forward_meta.json").read_text(encoding="utf-8"))
    if args.n:
        meta = [r for g in GROUPS for r in [x for x in meta if x["B0_name"] == g][: args.n]]

    # 방향은 이미지마다 한 번만 추정한다 (K 스윕과 무관)
    items = []
    for r in meta:
        g = np.load(multi / r["file"]).astype(np.float64)
        gt = np.load(args.data / "test_label" / r["file"].split("/")[-1]).astype(np.float64)
        b_meta = tuple(r["B0_dir"])
        b_est = estimate_b0(g)[1] if args.estimate else None
        items.append((r["B0_name"], g, gt, b_meta, b_est))
    if args.estimate:
        errs = []
        for name, _, _, _, b_est in items:
            deg = np.degrees(np.arctan2(b_est[0], b_est[1])) % 180.0
            true = float(name.split("_")[0])
            errs.append(min(abs(deg - true), 180 - abs(deg - true)))
        print(f"\n방향 추정 오차: 평균 {np.mean(errs):.3f}°  최대 {max(errs):.3f}°")

    print(f"\ntest_deconv_multi — {len(items)}장, 4방향 (겹치는 이미지 없음)\n")
    srcs = [("메타 방향", 3)] + ([("추정 방향", 4)] if args.estimate else [])
    results["multi"] = {}
    for label, idx in srcs:
        print(f"[{label}]")
        print(f"{'K':>10}" + "".join(f"{g:>11}" for g in GROUPS) + f"{'ALL':>11}{'SSIM':>9}")
        print("-" * (10 + 11 * 5 + 9))
        results["multi"][label] = {}
        for K in Ks:
            per = {}
            allp, alls = [], []
            for name, g, gt, b_meta, b_est in items:
                p, s = sc(wiener(g, K, b_meta if idx == 3 else b_est), gt)
                per.setdefault(name, []).append(p)
                allp.append(p); alls.append(s)
            results["multi"][label][f"{K:.0e}"] = {
                "per_group": {k: float(np.mean(v)) for k, v in per.items()},
                "psnr": float(np.mean(allp)), "ssim": float(np.mean(alls))}
            print(f"{K:>10.0e}" + "".join(f"{np.mean(per[g]):>11.2f}" for g in GROUPS)
                  + f"{np.mean(allp):>11.2f}{np.mean(alls):>9.4f}")
        print()

    # 0° 커널을 전부에 적용했을 때 (방향을 무시하면 어떻게 되는가)
    print("[방향 무시 — 전부 0° 커널]")
    print(f"{'K':>10}" + "".join(f"{g:>11}" for g in GROUPS) + f"{'ALL':>11}")
    print("-" * (10 + 11 * 5))
    results["multi"]["0° 고정"] = {}
    for K in (1e-4, 1e-6, 1e-12):
        per, allp = {}, []
        for name, g, gt, _, _ in items:
            p, _ = sc(wiener(g, K, (0.0, 1.0)), gt)
            per.setdefault(name, []).append(p); allp.append(p)
        results["multi"]["0° 고정"][f"{K:.0e}"] = {"psnr": float(np.mean(allp))}
        print(f"{K:>10.0e}" + "".join(f"{np.mean(per[g]):>11.2f}" for g in GROUPS) + f"{np.mean(allp):>11.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 → {args.out}")


if __name__ == "__main__":
    main()
