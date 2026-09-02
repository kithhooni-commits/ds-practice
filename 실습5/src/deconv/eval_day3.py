"""3일차 평가 — test_deconv_noise 100장.

`g = dipole(f) + n` 이고, 노이즈는 1일차와 **파일별로 동일**하다 (종류·σ 100/100 일치).
그래서 1일차 디노이저와 2일차 역산이 그대로 합쳐진다.

체크포인트 없이 돌리면 고전 기법(Wiener K 스윕)만 재고, `--ckpt` 를 주면 그 모델도 잰다.
배포 베이스라인(End2End U-Net, 13.43M)은 `--baseline` 으로 바로 불러올 수 있다.
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
from models import build_model  # noqa: E402

from challenge import wiener  # noqa: E402

DEFAULT_DATA = Path(os.environ.get("DS_DATA", ROOT / "data" / "dataset"))
NZ = ["gaussian", "rician", "uniform", "salt_and_pepper"]


def sc(est, gt):
    a = est if torch.is_tensor(est) else torch.from_numpy(est[None, None]).float()
    b = gt if torch.is_tensor(gt) else torch.from_numpy(gt[None, None]).float()
    return calculate_psnr(a, b).item(), calculate_ssim(a, b).item()


def table(name, rows, key_p=1, key_s=2):
    print(f"\n[{name}]")
    print(f"{'noise':<18}{'n':>4}{'PSNR':>10}{'SSIM':>10}")
    print("-" * 42)
    for nz in NZ + ["ALL"]:
        s = [r for r in rows if nz == "ALL" or r[0] == nz]
        print(f"{nz:<18}{len(s):>4}{np.mean([r[key_p] for r in s]):>10.2f}"
              f"{np.mean([r[key_s] for r in s]):>10.4f}")
    return float(np.mean([r[key_p] for r in rows])), float(np.mean([r[key_s] for r in rows]))


def load_net(path: Path, device):
    """우리 체크포인트와 배포 베이스라인을 모두 받는다."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ck:  # 배포 베이스라인
        cfg = ck.get("model_config", {})
        net = build_model("unet", features=cfg.get("chans", 64), num_of_layers=17)
        net.load_state_dict(ck["model_state_dict"])
        return net.to(device).eval(), "배포 베이스라인 (End2End U-Net)"

    name = ck.get("model", "unet")
    if name == "twostage":
        from twostage import TwoStageNet
        net = TwoStageNet(model=ck.get("refine", "drunet"), features=ck.get("features", 64),
                          sigma_map=ck.get("sigma_map", False),
                          lam_map=ck.get("lam_map", False),
                          refine_iters=ck.get("refine_iters", 0))
    elif name == "unrolled":
        from unrolled import UnrolledNet
        net = UnrolledNet(n_iter=ck.get("unroll_iters", 5), model=ck.get("refine", "unet"),
                          features=ck.get("features", 32),
                          share_weights=ck.get("share_weights", True),
                          sigma_map=ck.get("sigma_map", False))
    elif name == "dcnet":
        from dcnet import DCNet
        net = DCNet(model=ck.get("refine", "unet"), features=ck.get("features", 32),
                    tau=ck.get("tau", 0.05))
    elif name.startswith("spectral"):
        from spectral import SpectralNet
        net = SpectralNet(refine={"spectral": None, "spectral_dncnn": "dncnn",
                                  "spectral_unet": "unet"}[name], features=ck.get("features", 16))
    else:
        net = build_model(name, features=ck.get("features", 32), num_of_layers=17)
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval(), f"{name} (epoch {ck.get('epoch')})"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--baseline", action="store_true", help="배포 베이스라인 체크포인트를 쓴다")
    ap.add_argument("--wiener", action="store_true", help="고전 Wiener K 스윕도 잰다")
    ap.add_argument("--post-wiener", type=float, default=None,
                    help="모델 출력에 Wiener 를 한 번 더 건다. --target measure 로 학습한 모델용")
    ap.add_argument("--sweep-K", action="store_true",
                    help="--post-wiener 의 K 를 스윕한다 — **val 에서** 고른다")
    ap.add_argument("--n-val", type=int, default=100, help="스윕에 쓸 val 장수")
    ap.add_argument("--sigma-ablation", action="store_true",
                    help="σ 조건화 모델에 일부러 틀린 σ 를 먹여 얼마나 쓰고 있는지 잰다. "
                         "학습이 필요 없는 ablation 이다")
    ap.add_argument("--self-ensemble", action="store_true",
                    help="4x self-ensemble. dipole 이 견디는 대칭만 쓴다 (좌우/상하/180도). "
                         "90도 회전은 B0 방향을 돌려버려 못 쓴다")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.baseline and args.ckpt is None:
        args.ckpt = ROOT / "data" / "code_denoising+deconv" / "checkpoint_baseline_best.ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
    src = args.data / "test_deconv_noise"
    meta = json.loads((src / "noise_meta.json").read_text(encoding="utf-8"))
    items = [(r["noise_type"],
              np.load(src / r["file"]).astype(np.float64),
              np.load(args.data / "test_label" / r["file"]).astype(np.float64)) for r in meta]
    print(f"test_deconv_noise {len(items)}장 · g = dipole(f) + n")

    # K 스윕은 **val 에서** 한다. test 로 고르면 학습이 아니어도 test 를 쓴 것이다
    def val_items():
        from day3_common import load_val

        return [(nz, g.numpy()[0, 0].astype(np.float64), gt.numpy()[0, 0].astype(np.float64))
                for nz, g, gt in load_val(args.data, args.n_val)]

    res = {}
    rows = [(nz, *sc(g.astype(np.float32), gt.astype(np.float32))) for nz, g, gt in items]
    res["input"] = table("입력 (blur + noise)", rows)

    if args.wiener:
        best = (None, -1, 0)
        vitems = val_items()
        print(f"\n[Wiener K 스윕 — val {len(vitems)}장. test 는 건드리지 않는다]")
        print(f"{'K':>10}{'PSNR':>10}{'SSIM':>10}")
        print("-" * 30)
        for K in (1e-3, 1e-2, 3e-2, 1e-1, 3e-1):
            r = [(nz, *sc(wiener(g, K).astype(np.float32), gt.astype(np.float32))) for nz, g, gt in vitems]
            p, s = float(np.mean([x[1] for x in r])), float(np.mean([x[2] for x in r]))
            if p > best[1]:
                best = (K, p, s)
            print(f"{K:>10.0e}{p:>10.2f}{s:>10.4f}")
        print(f"val 최적 K={best[0]:.0e}  {best[1]:.2f} / {best[2]:.4f}")
        res["wiener_best_on_val"] = {"K": best[0], "val_psnr": best[1], "val_ssim": best[2]}
        # 고른 K 로 test 채점 — test 는 여기서 한 번만 본다
        r = [(nz, *sc(wiener(g, best[0]).astype(np.float32), gt.astype(np.float32)))
             for nz, g, gt in items]
        res["wiener_test"] = table(f"Wiener K={best[0]:.0e} (val 에서 고름)", r)

    if args.ckpt:
        net, label = load_net(args.ckpt, device)
        from unrolled import data_consistency

        def post(x, K):
            if K is None:
                return x
            return data_consistency(torch.zeros_like(x), x,
                                    torch.full((x.shape[0],), float(K), device=x.device))

        from unrolled import self_ensemble
        infer = (lambda a: self_ensemble(net, a)) if args.self_ensemble else net

        def run(K, its=None):
            rs = []
            with torch.no_grad():
                for nz, g, gt in (items if its is None else its):
                    a = torch.from_numpy(g.astype(np.float32))[None, None].to(device)
                    b = torch.from_numpy(gt.astype(np.float32))[None, None].to(device)
                    rs.append((nz, *sc(post(infer(a), K), b)))
            return rs

        if args.sweep_K:
            print()
            vit = val_items()
            print(f"[출력에 Wiener 를 한 번 더 — K 스윕, val {len(vit)}장]")
            print(f"{'K':>10}{'PSNR':>10}{'SSIM':>10}")
            print("-" * 30)
            bk = (None, -1, 0)
            for K in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
                r = run(K, vit)
                p_, s_ = float(np.mean([x[1] for x in r])), float(np.mean([x[2] for x in r]))
                if p_ > bk[1]:
                    bk = (K, p_, s_)
                print(f"{K:>10.0e}{p_:>10.2f}{s_:>10.4f}")
            print(f"val 최적 K={bk[0]:.0e}")
            args.post_wiener = bk[0]
            res["post_wiener_best_on_val"] = {"K": bk[0], "val_psnr": bk[1], "val_ssim": bk[2]}

        if args.sigma_ablation:
            # σ 를 어떻게 주느냐만 바꾼다. 가중치는 그대로 — 학습 비용 0
            import unrolled as _u

            real = _u.estimate_sigma
            variants = {
                "추정 σ (정상)": real,
                "σ = 0 (없다고 알려줌)": lambda m, **kw: torch.zeros(m.shape[0], device=m.device),
                "σ 뒤섞음 (다른 장의 값)": lambda m, **kw: real(m, **kw).flip(0),
                "σ 2배 (과대평가)": lambda m, **kw: real(m, **kw) * 2,
            }
            print()
            print("[σ ablation — 가중치는 그대로, σ 입력만 바꾼다]")
            print(f"{'σ 를 어떻게 주는가':<26}{'PSNR':>10}{'SSIM':>10}")
            print("-" * 46)
            for lab, fn in variants.items():
                _u.estimate_sigma = fn
                r = run(args.post_wiener)
                print(f"{lab:<26}{np.mean([x[1] for x in r]):>10.2f}"
                      f"{np.mean([x[2] for x in r]):>10.4f}")
            _u.estimate_sigma = real
            res["sigma_ablation"] = True

        rows = run(args.post_wiener)
        suffix = f" + Wiener K={args.post_wiener:.0e}" if args.post_wiener else ""
        res["model"] = table(label + suffix, rows)
        print(f"\n제출값 →  PSNR_total {res['model'][0]:.2f}   SSIM_total {res['model'][1]:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"저장 → {args.out}")


if __name__ == "__main__":
    main()
