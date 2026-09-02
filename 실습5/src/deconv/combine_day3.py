"""3일차 — 학습 없이 조합할 수 있는 모든 경로를 한 번에 잰다.

1일차 디노이저와 2일차 역산을 어떻게 붙이느냐로 결과가 갈린다. 세 가지를 비교한다.

    (a) Wiener 단독              2일차 답. K 를 스윕한다
    (b) Wiener → 디노이저 1회    2단 연결. 역산하고 남은 잡음을 지운다
    (c) plug-and-play 반복       역산 ↔ 디노이저를 번갈아 N번

그리고 디노이저를 바꿔 가며 본다.

    supervised   1일차 지도학습 DnCNN (day1 test 34.13)
    label-free   1일차 Noise2Void DnCNN (day1 test 30.95). clean 을 한 장도 안 썼다

**학습이 전혀 없다.** 전부 1일차 가중치를 그대로 불러 쓴다. 그래서 이 표는
`Others`(디노이저 없음) 와 `Supervised`/`Self-supervised`(디노이저 재사용) 칸을
동시에 채운다.

## 왜 K=1e-12 를 같이 재는가

2일차 최고가 K→0 이었으니 당연히 시도해 볼 조합이다. 다만 3일차에서 그 값은 단독으로
−24 dB 다 — 역산이 노이즈를 32 dB 증폭하기 때문이다. 디노이저를 뒤에 붙이면 그걸
복구할 수 있는지가 질문이고, 표가 답한다.
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

from pnp import load_denoiser, pnp  # noqa: E402
from unrolled import data_consistency  # noqa: E402

DEFAULT_DATA = Path(os.environ.get("DS_DATA", ROOT / "data" / "dataset"))
CKPT = ROOT / "checkpoints"
NZ = ["gaussian", "rician", "uniform", "salt_and_pepper"]


def wiener_t(measure: torch.Tensor, K: float) -> torch.Tensor:
    B = measure.shape[0]
    return data_consistency(torch.zeros_like(measure), measure,
                            torch.full((B,), float(K), device=measure.device))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--n", type=int, default=0, help="평가 장수 제한 (0=전부)")
    ap.add_argument("--denoisers", nargs="*", default=None,
                    help="쓸 디노이저 체크포인트. 기본은 checkpoints/ 의 day1_* 전부")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "day3_combine.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")

    src = args.data / "test_deconv_noise"
    meta = json.loads((src / "noise_meta.json").read_text(encoding="utf-8"))
    if args.n:
        meta = meta[: args.n]
    items = []
    for r in meta:
        g = torch.from_numpy(np.load(src / r["file"]).astype(np.float32))[None, None].to(device)
        gt = torch.from_numpy(np.load(args.data / "test_label" / r["file"]).astype(np.float32))[None, None].to(device)
        items.append((r["noise_type"], g, gt))
    print(f"test_deconv_noise {len(items)}장 · g = dipole(f) + n\n")

    paths = [Path(p) for p in args.denoisers] if args.denoisers else sorted(CKPT.glob("day1_*.ckpt"))
    dens = {}
    for p in paths:
        net, tag, ck = load_denoiser(p, device)
        key = "label-free" if ck.get("label_free") else "supervised"
        dens[key] = (net, tag)
        print(f"  {key:<12} {p.name:<32} {tag}")
    print()

    Ks = [1e-12, 1e-6, 1e-3, 1e-2, 3e-2, 1e-1]
    res, rows_out = {}, []

    def score(fn):
        rs = [(nz, calculate_psnr(fn(g), gt).item(), calculate_ssim(fn(g), gt).item())
              for nz, g, gt in items]
        return float(np.mean([r[1] for r in rs])), float(np.mean([r[2] for r in rs]))

    with torch.no_grad():
        # (a) Wiener 단독  +  (b) Wiener → 디노이저 1회
        cols = ["Wiener 단독"] + [f"→ {k}" for k in dens]
        print("(a) Wiener 단독 · (b) Wiener 뒤에 1일차 디노이저를 한 번\n")
        print(f"{'K':>10}" + "".join(f"{c:>18}" for c in cols))
        print("-" * (10 + 18 * len(cols)))
        for K in Ks:
            line, entry = f"{K:>10.0e}", {}
            p, s = score(lambda g, K=K: wiener_t(g, K))
            entry["wiener"] = {"psnr": p, "ssim": s}
            line += f"{f'{p:.2f} / {s:.4f}':>18}"
            for key, (net, _) in dens.items():
                p, s = score(lambda g, K=K, net=net: net(wiener_t(g, K)))
                entry[key] = {"psnr": p, "ssim": s}
                line += f"{f'{p:.2f} / {s:.4f}':>18}"
            res[f"K={K:.0e}"] = entry
            print(line)

        # (c) plug-and-play
        print(f"\n(c) plug-and-play — 역산 ↔ 디노이저를 번갈아\n")
        print(f"{'iters':>7}{'λ0→λ1':>14}" + "".join(f"{k:>18}" for k in dens))
        print("-" * (21 + 18 * len(dens)))
        best = {k: (None, -1, 0) for k in dens}
        for n_iter in (1, 2, 4, 8):
            for lam0, lam1 in ((0.5, 0.03), (0.3, 0.01), (0.1, 0.01), (0.05, 0.01)):
                line = f"{n_iter:>7}{f'{lam0}→{lam1}':>14}"
                for key, (net, _) in dens.items():
                    p, s = score(lambda g, n=n_iter, a=lam0, b=lam1, net=net: pnp(g, net, n, a, b))
                    if p > best[key][1]:
                        best[key] = ((n_iter, lam0, lam1), p, s)
                    line += f"{f'{p:.2f} / {s:.4f}':>18}"
                    rows_out.append({"denoiser": key, "iters": n_iter, "lam0": lam0,
                                     "lam1": lam1, "psnr": p, "ssim": s})
                print(line)

    print()
    for key, (cfg, p, s) in best.items():
        print(f"plug-and-play 최적 [{key}]  iters={cfg[0]} λ {cfg[1]}→{cfg[2]}   {p:.2f} / {s:.4f}")
    res["pnp"] = rows_out
    res["pnp_best"] = {k: {"iters": v[0][0], "lam0": v[0][1], "lam1": v[0][2],
                           "psnr": v[1], "ssim": v[2]} for k, v in best.items()}

    print(f"\n비교 기준:  배포 baseline 25.01 / 0.8149   ·   입력 8.02 / −0.0187")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 → {args.out}")


if __name__ == "__main__":
    main()
