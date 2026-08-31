"""우리 데이터 경로·지표 구현이 과제 제공 결과와 일치하는지 확인한다.

제공된 예시 로그(`log_denoising_example/00012_train/baseline_metrics.json`)에는
mean / median / adaptive 필터의 test 성적이 들어 있다. 같은 데이터에 같은 필터를
돌렸으니 숫자가 맞아야 한다. 어긋나면 로더나 지표 어딘가가 틀린 것이고, 그러면
학습 결과도 믿을 수 없다. 모델을 돌리기 전에 여기부터 맞춘다.
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
from filters import adaptive_filter, mean_filter, median_filter
from metrics import calculate_psnr, calculate_ssim

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_DATA = Path(os.environ.get("DS_DATA", ROOT / "data" / "dataset"))
REF = ROOT / "data" / "log_denoising_example" / "00012_train" / "baseline_metrics.json"
NOISE_ORDER = ["gaussian", "rician", "uniform", "salt_and_pepper"]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset 폴더")
    args = ap.parse_args()
    DATA = args.data
    print(f"data root: {DATA}")

    noisy_dir = resolve_test_noisy(DATA)
    meta = json.loads((noisy_dir / "noise_meta.json").read_text())
    lookup = {r["file"]: r["noise_type"] for r in meta}

    loader, _ = make_loader(
        [DATA / "test_label"], training_mode=False, batch=1, num_workers=0,
        noisy_dir=noisy_dir,
    )

    methods = ["noisy", "mean", "median", "adaptive"]
    rows = []
    with torch.no_grad():
        for label, noisy, names in loader:
            outs = {
                "noisy": noisy,
                "mean": mean_filter(noisy, 3),
                "median": median_filter(noisy, 3),
                "adaptive": adaptive_filter(noisy, 5),
            }
            row = {"file": names[0], "noise_type": lookup.get(names[0], "unknown")}
            for m in methods:
                row[f"psnr_{m}"] = calculate_psnr(outs[m], label).item()
                row[f"ssim_{m}"] = calculate_ssim(outs[m], label).item()
            rows.append(row)

    ref = json.loads(REF.read_text()) if REF.exists() else None

    for metric in ("psnr", "ssim"):
        fmt = ".3f" if metric == "psnr" else ".4f"
        print(f"\n[{metric.upper()}]  ours / 제공 예시 로그")
        print(f"{'noise':<18}{'n':>4}" + "".join(f"{m:>22}" for m in methods))
        print("-" * (22 + 22 * len(methods)))
        for nz in NOISE_ORDER + ["ALL"]:
            sub = [r for r in rows if nz == "ALL" or r["noise_type"] == nz]
            cells = ""
            for m in methods:
                ours = np.mean([r[f"{metric}_{m}"] for r in sub])
                if ref is None:
                    cells += f"{ours:>22{fmt}}"
                else:
                    rsub = [r for r in ref if nz == "ALL" or r["noise_type"] == nz]
                    theirs = np.mean([r[f"{metric}_{m}"] for r in rsub])
                    cells += f"{f'{ours:{fmt}} / {theirs:{fmt}}':>22}"
            print(f"{nz:<18}{len(sub):>4}{cells}")

    if ref is not None:
        diffs = [
            abs(np.mean([r[f"psnr_{m}"] for r in rows]) - np.mean([r[f"psnr_{m}"] for r in ref]))
            for m in methods
        ]
        ok = max(diffs) < 0.01
        print(f"\nPSNR 최대 차이: {max(diffs):.4f} dB — {'일치' if ok else '어긋남, 확인 필요'}")


if __name__ == "__main__":
    main()
