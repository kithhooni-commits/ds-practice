"""test 100장을 이미지 하나씩 뜯어보는 표.

이미지마다 (1) 어떤 노이즈가 얼마나 실렸는지, (2) 각 방법이 그걸 얼마나 지웠는지를
한 줄로 낸다. 전체 평균만 보면 "어떤 방법이 어떤 노이즈에서 무너지는가"가 안 보인다.

출력: figures/per_image.csv (전체) + 콘솔 요약
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from data import make_loader, resolve_test_noisy
from filters import adaptive_filter, mean_filter, median_filter
from metrics import calculate_psnr, calculate_ssim

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_DATA = Path(os.environ.get("DS_DATA", ROOT / "data" / "dataset"))
NOISE_ORDER = ["gaussian", "rician", "uniform", "salt_and_pepper"]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--ckpt", type=Path, default=None, help="있으면 model 열을 추가한다")
    ap.add_argument("--device", default="cpu", help="학습 중이면 cpu 로 두는 게 안전하다")
    ap.add_argument("--self-ensemble", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "per_image.csv")
    ap.add_argument("--show", type=int, default=0, help="콘솔에 찍을 이미지 줄 수 (0 = 요약만)")
    args = ap.parse_args()

    device = torch.device(args.device)
    noisy_dir = resolve_test_noisy(args.data)
    meta = {r["file"]: r for r in json.loads((noisy_dir / "noise_meta.json").read_text())}

    net = None
    methods = ["noisy", "mean", "median", "adaptive"]
    if args.ckpt is not None:
        from evaluate import load_net

        net, ck = load_net(args.ckpt, device)
        methods.insert(1, "model")
        print(f"model: {args.ckpt.name}  (epoch {ck.get('epoch')}, val PSNR {ck.get('val_psnr', float('nan')):.3f})")

    loader, _ = make_loader(
        [args.data / "test_label"], training_mode=False, batch=1, num_workers=0, noisy_dir=noisy_dir
    )

    rows: list[dict] = []
    with torch.no_grad():
        for label, noisy, names in loader:
            label, noisy = label.to(device), noisy.to(device)
            outs = {
                "noisy": noisy,
                "mean": mean_filter(noisy, 3),
                "median": median_filter(noisy, 3),
                "adaptive": adaptive_filter(noisy, 5),
            }
            if net is not None:
                from evaluate import predict

                outs["model"] = predict(net, noisy, args.self_ensemble)

            name = names[0]
            m = meta.get(name, {})
            row = {"file": name, "noise_type": m.get("noise_type", "unknown"), "sigma": round(m.get("sigma", float("nan")), 4)}
            for k in methods:
                row[f"psnr_{k}"] = round(calculate_psnr(outs[k], label).item(), 3)
                row[f"ssim_{k}"] = round(calculate_ssim(outs[k], label).item(), 4)
            # 그 이미지에서 가장 잘한 방법 (noisy 는 제외)
            cand = [k for k in methods if k != "noisy"]
            row["best_psnr"] = max(cand, key=lambda k: row[f"psnr_{k}"])
            row["best_ssim"] = max(cand, key=lambda k: row[f"ssim_{k}"])
            rows.append(row)

    rows.sort(key=lambda r: (NOISE_ORDER.index(r["noise_type"]) if r["noise_type"] in NOISE_ORDER else 9, -r["sigma"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- 노이즈 종류별 요약 ----
    for metric in ("psnr", "ssim"):
        fmt = ".3f" if metric == "psnr" else ".4f"
        w_ = 11
        print(f"\n[{metric.upper()}] 노이즈 종류별 평균")
        print(f"{'noise':<18}{'n':>4}{'σ 평균':>9}" + "".join(f"{k:>{w_}}" for k in methods))
        print("-" * (31 + w_ * len(methods)))
        for nz in NOISE_ORDER + ["ALL"]:
            sub = [r for r in rows if nz == "ALL" or r["noise_type"] == nz]
            if not sub:
                continue
            sig = np.mean([r["sigma"] for r in sub])
            cells = "".join(f"{np.mean([r[f'{metric}_{k}'] for r in sub]):>{w_}{fmt}}" for k in methods)
            print(f"{nz:<18}{len(sub):>4}{sig:>9.3f}{cells}")

    # ---- 이미지별로 어떤 방법이 이겼는지 ----
    print("\n이미지별 최고 성적 방법 (PSNR 기준)")
    for nz in NOISE_ORDER:
        sub = [r for r in rows if r["noise_type"] == nz]
        tally = {k: sum(1 for r in sub if r["best_psnr"] == k) for k in methods if k != "noisy"}
        tally = {k: v for k, v in tally.items() if v}
        print(f"  {nz:<18} " + ", ".join(f"{k} {v}장" for k, v in sorted(tally.items(), key=lambda x: -x[1])))

    if args.show:
        print(f"\n이미지별 상세 (앞 {args.show}장, σ 큰 순)")
        hdr = f"{'file':<14}{'noise':<17}{'σ':>7}" + "".join(f"{k:>10}" for k in methods)
        print(hdr); print("-" * len(hdr))
        for r in rows[: args.show]:
            cells = "".join(f"{r[f'psnr_{k}']:>10.2f}" for k in methods)
            print(f"{r['file'][3:11]:<14}{r['noise_type']:<17}{r['sigma']:>7.3f}{cells}")

    print(f"\n전체 100장 → {args.out}")


if __name__ == "__main__":
    main()
