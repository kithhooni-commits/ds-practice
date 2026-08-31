"""test 세트 평가 — 제출용 PSNR_total / SSIM_total 을 낸다.

입력은 `dataset/test_noise_only` 의 손상 이미지, 정답은 `dataset/test_label`.
노이즈 종류는 `test_noise_only/noise_meta.json` 에서 읽어 표를 종류별로 쪼갠다.
(종류 정보는 표를 나누는 데만 쓴다. 복원 자체는 종류를 모르는 채로 한다 —
채점 세트에 메타가 없을 수도 있고, 있어도 그건 다른 문제다.)

conventional 비교군(mean/median/adaptive)도 같은 데이터로 함께 잰다.
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
from models import build_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_DATA = Path(os.environ.get("DS_DATA", ROOT / "data" / "dataset"))
NOISE_ORDER = ["gaussian", "rician", "uniform", "salt_and_pepper"]


def load_net(ckpt_path: Path, device) -> torch.nn.Module:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = build_model(ck.get("model", "dncnn"), num_of_layers=ck.get("layers", 17), features=ck.get("features", 64))
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval(), ck


@torch.no_grad()
def predict(net, x: torch.Tensor, self_ensemble: bool = False) -> torch.Tensor:
    """self-ensemble: 8가지 dihedral 변환으로 추론하고 되돌려 평균낸다.

    학습 때 flip/rot90 증강을 했으므로 네트워크는 이 8개를 같게 다뤄야 맞다.
    실제로는 조금씩 다르게 나오고, 평균이 그 편차를 지운다. 학습 비용 0, 추론 8배.
    """
    if not self_ensemble:
        return net(x)
    acc = torch.zeros_like(x)
    for k in range(4):
        for flip in (False, True):
            t = torch.rot90(x, k, dims=[2, 3])
            if flip:
                t = torch.flip(t, dims=[3])
            y = net(t)
            if flip:
                y = torch.flip(y, dims=[3])
            acc += torch.rot90(y, -k, dims=[2, 3])
    return acc / 8.0


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path, help="checkpoint_best.ckpt 경로")
    ap.add_argument("--self-ensemble", action="store_true")
    ap.add_argument("--clip", action="store_true", help="출력을 [0, label_max] 로 자른다")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=None, help="결과 저장 폴더 (기본: ckpt 의 run 폴더)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, ck = load_net(args.ckpt, device)
    out_dir = args.out or args.ckpt.parent.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    DATA = args.data
    noisy_dir = resolve_test_noisy(DATA)
    meta_path = noisy_dir / "noise_meta.json"
    noise_lookup: dict[str, str] = {}
    if meta_path.exists():
        noise_lookup = {r["file"]: r["noise_type"] for r in json.loads(meta_path.read_text())}

    loader, ds = make_loader(
        [DATA / "test_label"], training_mode=False, batch=1, num_workers=0,
        noisy_dir=noisy_dir,
    )

    methods = ["noisy", "model", "mean", "median", "adaptive"]
    rows: list[dict] = []
    samples: dict[str, dict] = {}

    with torch.no_grad():
        for label, noisy, names in loader:
            label, noisy = label.to(device), noisy.to(device)
            pred = predict(net, noisy, args.self_ensemble)
            if args.clip:
                pred = pred.clamp(0.0, float(label.max()))

            outs = {
                "noisy": noisy,
                "model": pred,
                "mean": mean_filter(noisy, 3),
                "median": median_filter(noisy, 3),
                "adaptive": adaptive_filter(noisy, 5),
            }
            name = names[0]
            row = {"file": name, "noise_type": noise_lookup.get(name, "unknown")}
            for m in methods:
                row[f"psnr_{m}"] = calculate_psnr(outs[m], label).item()
                row[f"ssim_{m}"] = calculate_ssim(outs[m], label).item()
            rows.append(row)

            if row["noise_type"] not in samples:
                samples[row["noise_type"]] = {
                    "metrics": row,
                    "label": label.cpu().numpy().squeeze(),
                    **{m: outs[m].cpu().numpy().squeeze() for m in methods},
                }

    order = [n for n in NOISE_ORDER if any(r["noise_type"] == n for r in rows)]
    order += sorted({r["noise_type"] for r in rows} - set(order))

    def table(metric: str) -> None:
        fmt = ".3f" if metric == "psnr" else ".4f"
        w = 12
        print(f"\n[{metric.upper()}]  ckpt: {args.ckpt.name}  self-ensemble: {args.self_ensemble}")
        print(f"{'noise':<18}{'n':>4}" + "".join(f"{m:>{w}}" for m in methods))
        print("-" * (22 + w * len(methods)))
        for nz in order + ["ALL"]:
            sub = [r for r in rows if nz == "ALL" or r["noise_type"] == nz]
            cells = "".join(f"{np.mean([r[f'{metric}_{m}'] for r in sub]):>{w}{fmt}}" for m in methods)
            print(f"{nz:<18}{len(sub):>4}{cells}")

    table("psnr")
    table("ssim")

    psnr_total = float(np.mean([r["psnr_model"] for r in rows]))
    ssim_total = float(np.mean([r["ssim_model"] for r in rows]))
    print(f"\n제출값 →  PSNR_total {psnr_total:.2f}   SSIM_total {ssim_total:.4f}")
    print(f"(학습 시 best val PSNR {ck.get('val_psnr', float('nan')):.3f} @ epoch {ck.get('epoch')})")

    tag = "_se" if args.self_ensemble else ""
    res = out_dir / f"test_metrics{tag}.json"
    res.write_text(
        json.dumps(
            {"psnr_total": psnr_total, "ssim_total": ssim_total,
             "ckpt": str(args.ckpt), "self_ensemble": args.self_ensemble, "rows": rows},
            ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"저장 → {res}")

    if args.figures:
        save_grid(samples, order, methods, out_dir / f"test_grid{tag}.png")


def save_grid(samples: dict, order: list[str], methods: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["noisy", "model", "median", "adaptive"]
    fig, axes = plt.subplots(len(order), len(cols) + 1, figsize=(3.1 * (len(cols) + 1), 3.4 * len(order)))
    axes = np.asarray(axes).reshape(len(order), len(cols) + 1)

    for r, nz in enumerate(order):
        s = samples[nz]
        lab = s["label"]
        vmax = float(np.percentile(lab, 98) * 1.2)
        axes[r, 0].imshow(lab, cmap="gray", vmin=0, vmax=vmax)
        axes[r, 0].set_ylabel(nz, fontsize=10)
        axes[r, 0].set_title("label", fontsize=9)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for c, m in enumerate(cols, start=1):
            ax = axes[r, c]
            ax.imshow(s[m], cmap="gray", vmin=0, vmax=vmax)
            ax.set_title(f"{m}\n{s['metrics'][f'psnr_{m}']:.2f} dB / {s['metrics'][f'ssim_{m}']:.4f}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"그림 저장 → {path}")


if __name__ == "__main__":
    main()
