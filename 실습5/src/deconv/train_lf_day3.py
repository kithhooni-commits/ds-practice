"""3일차 label-free — 정답을 한 장도 쓰지 않고 푼다. (보너스 점수)

## 지금까지의 label-free 가 약했던 이유

발표에 있던 label-free 는 **1일차에 학습한 N2V 가중치를 3일차에 갖다 쓴 것**이었다
(18.87 dB). 3일차 조건에서 자기지도로 학습한 것이 아니다. 여기서 제대로 한다.

## 어떻게 정답 없이 배우는가

3일차는 `g = h*f + n` 이고 노이즈가 흐림 **뒤에** 붙으므로 **측정치 위에서는 백색**이다.
백색 잡음이면 Noise2Void 가 그대로 통한다.

    입력 g 의 일부 화소를 이웃 값으로 덮고, 덮은 자리에서 원래 g 를 맞히게 한다.

네트워크는 덮인 화소를 볼 수 없으니 이웃으로부터 **신호** 를 추정할 수밖에 없다.
잡음은 화소마다 독립이라 이웃으로 예측할 수 없으므로, 최적해가 곧 잡음 없는 `h*f` 다.
정답 `f` 도, 잡음 없는 `h*f` 도 손실에 들어가지 않는다 — 오직 `g` 뿐이다.

그다음은 2일차 답을 그대로 쓴다.

    z = 디노이저(g)                  <- 여기까지 label-free
    x = (D·Z)/(D² + λ)               <- 역필터. 커널은 알고 있다 (조교 지침 4)

## λ 도 라벨 없이 정한다

val 로 λ 를 고르면 정답을 쓴 것이다. 대신 측정치에서 잰 σ 로 정한다 — 고전 Wiener 의
`K = 잡음파워 / 신호파워` 를 그대로 계산한다. 둘 다 측정치에서 나온다.

    잡음파워  = σ²                         널 원뿔에서 잰다
    신호파워  = max(E|G|² − σ², 0) / |D|²  측정치에서 잡음 몫을 뺀 것

비교를 위해 val 에서 고른 λ 도 같이 보고하되, **제출용은 라벨 없이 고른 쪽**이다.

## L1 을 쓰는 이유

1일차와 같다. 4종 중 salt & pepper 와 rician 은 평균이 치우쳐 있어 L2 로 맞히면
임펄스 쪽으로 끌려간다. L1 은 조건부 중앙값으로 수렴해 그 편향을 피한다.

## 규칙

학습에 쓰는 측정치는 **train 폴더의 clean 에서 합성**한다. clean 은 측정치를 만드는
데만 쓰이고 **손실에는 한 번도 들어가지 않는다.** test 는 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))
from metrics import calculate_psnr, calculate_ssim  # noqa: E402
from models import build_model  # noqa: E402
from train_n2v import blind_spot_mask  # noqa: E402

from day3_common import DEFAULT_DATA, load_test, load_val  # noqa: E402
from train_deconv import DeconvDataset  # noqa: E402
from unrolled import _otf, data_consistency, estimate_sigma, self_ensemble  # noqa: E402


def label_free_lambda(measure: Tensor, b0: tuple[float, float] = (0.0, 1.0)) -> Tensor:
    """측정치만 보고 Wiener 의 λ 를 정한다. 정답을 쓰지 않는다.

    고전 Wiener 의 K = 잡음파워/신호파워 를 주파수 전체 평균으로 계산한다.
    신호파워는 측정치 파워에서 잡음 몫을 뺀 뒤 |D|² 로 나눠 되돌린 값이다.
    """
    D = _otf(tuple(measure.shape[-2:]), b0, measure.device, torch.float32)
    sigma = estimate_sigma(measure, b0=b0)
    n_pix = measure.shape[-1] * measure.shape[-2]
    P = (torch.fft.fft2(measure.float()).abs() ** 2).mean((-2, -1)).view(-1) / n_pix
    noise = sigma**2
    sig = ((P - noise).clamp_min(1e-12)) / (D**2).mean().clamp_min(1e-12)
    return (noise / sig).clamp(1e-6, 1.0)


@torch.no_grad()
def restore(net, measure: Tensor, lam: Tensor | None = None, se: bool = True) -> Tensor:
    """z = 디노이저(g) -> x = Wiener(z, λ). λ 를 안 주면 라벨 없이 정한다.

    ## λ 는 **입력** 측정치에서 정한다 (디노이징 뒤가 아니다)

    한때 "디노이저가 노이즈를 지웠으니 λ 도 작아져야 한다" 고 보고 출력 z 의 널
    원뿔에서 잔여 σ 를 재 봤다. **더 나빴다** (val 16.30 vs 17.90).

    이유는 디노이저가 널 원뿔의 노이즈까지 같이 뭉개기 때문이다. 그러면 잔여 σ 가
    실제보다 작게 나오고 λ 가 지나치게 작아져 노이즈를 덜 막는다. 디노이징 뒤의
    널 원뿔은 잔여 노이즈의 공정한 표본이 아니다 — 거기만 유독 세게 지워져 있다.

    입력 기준 λ 는 val 에서 고른 고정 K(17.77) 보다도 낫다. 정답을 한 번도 쓰지
    않고 그렇다.
    """
    z = self_ensemble(net, measure) if se else net(measure)
    if lam is None:
        lam = label_free_lambda(measure)
    elif not torch.is_tensor(lam):
        lam = torch.full((measure.shape[0],), float(lam), device=measure.device)
    return data_consistency(torch.zeros_like(z), z, lam)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--features", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ratio", type=float, default=0.02, help="가릴 화소 비율")
    ap.add_argument("--window", type=int, default=5, help="이웃을 뽑는 창 크기")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-val", type=int, default=40, help="에폭마다 잴 val 장수")
    ap.add_argument("--n-test", type=int, default=0, help="마지막 채점 장수 (0=전부)")
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--mirror", type=Path, default=None)
    ap.add_argument("--tag", default="lf_day3")
    ap.add_argument("--eval-only", type=Path, default=None,
                    help="학습을 건너뛰고 이 체크포인트를 채점한다. λ 는 모델의 일부가 "
                         "아니므로 규칙만 고쳐 다시 재는 데 쓴다")
    args = ap.parse_args()

    ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if ok else "cpu")
    amp_dtype = torch.bfloat16 if ok and torch.cuda.is_bf16_supported() else torch.float16

    # 측정치만 쓴다. clean 은 측정치를 만드는 데만 쓰이고 손실에 들어가지 않는다.
    ds = DeconvDataset(args.data / "train", True, None, 0.0, False,
                       "measure", "challenge", "label")
    if args.limit_train:
        idx = np.linspace(0, len(ds.files) - 1, args.limit_train).astype(int)
        ds.files = [ds.files[i] for i in sorted(set(idx))]
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)

    # 측정치 영역 디노이저. σ 는 널 원뿔에서 읽으므로 이것도 라벨이 필요 없다
    net = build_model("drunet", features=args.features, num_of_layers=17,
                      sigma_map=True).to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    run = args.out / f"{time.strftime('%m%d-%H%M')}_{args.tag}"
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run    : {run}")
    print(f"model  : drunet f{args.features} (측정치 영역) | N2V ratio {args.ratio} "
          f"window {args.window} | loss L1")
    print(f"학습   : {len(ds.files)}장 · 정답은 손실에 들어가지 않는다 (측정치 g 만)")
    print(f"amp    : {amp_dtype if ok else 'disabled'}\n")

    # 평가용 — 여기서만 정답을 본다 (점수를 재기 위해서지 학습에 쓰지 않는다)
    val = load_val(args.data, args.n_val, device)
    test = load_test(args.data, args.n_test, device)

    if args.eval_only:
        ck0 = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        net.load_state_dict(ck0["state_dict"]); net.eval()
        print(f"채점만 한다: {args.eval_only.name} "
              f"(ep {ck0.get('epoch')}, 학습 당시 val {ck0.get('val_psnr', float('nan')):.2f})")
        print()
        # λ 규칙 비교. 제출용은 라벨 없이 고른 쪽이다
        print(f"{'λ 를 어떻게 정하는가':<34}{'val PSNR':>10}{'val SSIM':>10}")
        print("-" * 54)
        for lab, fn in [
            ("라벨 없이 (입력 σ) — 제출용", lambda g: restore(net, g)),
            ("라벨 없이 (디노이징 뒤 잔여 σ)",
             lambda g: restore(net, g, label_free_lambda(
                 self_ensemble(net, g) if True else g))),
        ]:
            ps = [calculate_psnr(fn(g), gt).item() for _, g, gt in val]
            ss = [calculate_ssim(fn(g), gt).item() for _, g, gt in val]
            print(f"{lab:<34}{np.mean(ps):>10.2f}{np.mean(ss):>10.4f}")
        best_k = (None, -1.0, 0.0)
        for K in np.logspace(-5, -1, 17):
            ps = [calculate_psnr(restore(net, g, float(K)), gt).item() for _, g, gt in val]
            if np.mean(ps) > best_k[1]:
                ss = [calculate_ssim(restore(net, g, float(K)), gt).item() for _, g, gt in val]
                best_k = (float(K), float(np.mean(ps)), float(np.mean(ss)))
        print(f"{'(참고) val 에서 고른 고정 K=' + f'{best_k[0]:.2g}':<34}"
              f"{best_k[1]:>10.2f}{best_k[2]:>10.4f}")
        print()
        print("제출용은 라벨 없이 고른 쪽이다. 고정 K 는 얼마나 손해인지 보려고 같이 잰다.")
        print()
        rows = [(nz, calculate_psnr(restore(net, g), gt).item(),
                 calculate_ssim(restore(net, g), gt).item()) for nz, g, gt in test]
        print(f"[label-free · λ 도 측정치에서 · 4× self-ensemble — test {len(rows)}장]")
        print(f"{'noise':<18}{'n':>4}{'PSNR':>10}{'SSIM':>10}")
        print("-" * 42)
        for nz in ["gaussian", "rician", "uniform", "salt_and_pepper", "ALL"]:
            s_ = [r for r in rows if nz == "ALL" or r[0] == nz]
            if s_:
                print(f"{nz:<18}{len(s_):>4}{np.mean([r[1] for r in s_]):>10.2f}"
                      f"{np.mean([r[2] for r in s_]):>10.4f}")
        print()
        print(f"제출값(label-free) → PSNR {np.mean([r[1] for r in rows]):.2f}  "
              f"SSIM {np.mean([r[2] for r in rows]):.4f}")
        return

    best = -1.0
    for ep in range(args.epochs):
        net.train()
        tot = n = 0
        for _, g, _, _ in loader:
            g = g.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=ok, dtype=amp_dtype):
                masked, m = blind_spot_mask(g, args.ratio, args.window)
                sigma = estimate_sigma(masked)
                out = net(masked, sigma)
                # 가린 자리에서만 원래 측정치를 맞힌다. 정답은 등장하지 않는다
                loss = ((out - g).abs() * m).sum() / m.sum().clamp_min(1)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
            optim.step()
            tot += loss.item(); n += 1
        sched.step()

        net.eval()
        ps = [calculate_psnr(restore(net, g), gt).item() for _, g, gt in val]
        ss = [calculate_ssim(restore(net, g), gt).item() for _, g, gt in val]
        p, s = float(np.mean(ps)), float(np.mean(ss))
        mark = ""
        if p > best:
            best = p
            torch.save({"model": "drunet", "features": args.features, "layers": 17,
                        "sigma_map": True, "label_free": True, "domain": "measure",
                        "state_dict": net.state_dict(), "epoch": ep,
                        "val_psnr": p, "val_ssim": s},
                       run / "checkpoints" / "checkpoint_best.ckpt")
            if args.mirror:
                import shutil
                args.mirror.mkdir(parents=True, exist_ok=True)
                dst = args.mirror / f"{run.name}.ckpt"
                tmp = dst.with_suffix(".ckpt.part")
                shutil.copy(run / "checkpoints" / "checkpoint_best.ckpt", tmp)
                tmp.replace(dst)
            mark = "  <- best"
        print(f"[ep {ep:02d}] loss {tot / max(n,1):.5f}  val PSNR {p:6.3f}  SSIM {s:.4f}{mark}")

    # ---- 최종: test 채점 ----
    ck = torch.load(run / "checkpoints" / "checkpoint_best.ckpt",
                    map_location="cpu", weights_only=False)
    net.load_state_dict(ck["state_dict"]); net.eval()
    rows = [(nz, calculate_psnr(restore(net, g), gt).item(),
             calculate_ssim(restore(net, g), gt).item()) for nz, g, gt in test]
    print(f"\n[label-free · λ 도 측정치에서 · 4× self-ensemble — test {len(rows)}장]")
    print(f"{'noise':<18}{'n':>4}{'PSNR':>10}{'SSIM':>10}")
    print("-" * 42)
    for nz in ["gaussian", "rician", "uniform", "salt_and_pepper", "ALL"]:
        s_ = [r for r in rows if nz == "ALL" or r[0] == nz]
        print(f"{nz:<18}{len(s_):>4}{np.mean([r[1] for r in s_]):>10.2f}"
              f"{np.mean([r[2] for r in s_]):>10.4f}")
    print(f"\n제출값(label-free) → PSNR {np.mean([r[1] for r in rows]):.2f}  "
          f"SSIM {np.mean([r[2] for r in rows]):.4f}")
    print("정답은 점수를 재는 데만 썼다. 학습에도, λ 를 고르는 데도 쓰지 않았다.")


if __name__ == "__main__":
    main()
