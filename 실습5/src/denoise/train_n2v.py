"""label-free 학습 — clean 이미지를 loss 에 한 번도 쓰지 않는다 (Noise2Void).

## 원리

노이즈가 픽셀마다 독립이라는 성질 하나에 기댄다. 어떤 픽셀의 값을 **주변 값으로
바꿔치기해서 입력에서 지운 다음**, 네트워크에게 그 자리를 맞혀 보라고 한다.
정답으로는 **원래의 noisy 값**을 준다.

    입력   : noisy 이미지 (일부 픽셀을 이웃 값으로 덮어씀)
    정답   : 덮어쓰기 전 그 픽셀의 noisy 값
    loss   : 덮어쓴 자리에서만 계산

네트워크는 그 픽셀의 노이즈를 알 방법이 없다 — 입력에서 지워졌고, 주변에는
독립이라 단서가 없다. 맞힐 수 있는 건 구조뿐이다. 그래서 노이즈를 못 배우고
구조만 배운다. clean 은 어디에도 등장하지 않는다.

## 우리 노이즈 4종은 이 전제를 만족하는가

gaussian · uniform · salt&pepper · rician 전부 픽셀별로 독립이다. 만족한다.

다만 loss 를 L2 로 두면 조건부 **평균**을 학습하는데, 그게 clean 과 같으려면
노이즈 평균이 0 이어야 한다. gaussian·uniform 은 0 이지만 salt&pepper 는 0/max 로
덮으니 아니고, rician 은 절댓값 때문에 위로 들린다. 그래서 기본을 L1 으로 둔다 —
L1 은 조건부 **중앙값**을 학습하고, 중앙값은 임펄스에 끌려가지 않는다.

## 두 가지 모드

  --source test   test_noise_only 100장만 쓴다. clean 을 한 장도 건드리지 않는다.
                  가장 순수한 label-free 이고, test-time adaptation 이기도 하다.
  --source train  train/ clean 으로 noisy 를 합성한 뒤 그 noisy 만 학습에 쓴다.
                  loss 에 clean 이 안 들어가는 건 같지만, clean 이 존재해야 가능하다.
                  데이터가 7,268장이라 훨씬 유리하다.

둘 다 돌려서 정직하게 같이 보고한다.

## 모델 선택도 label-free 로

best 체크포인트를 val PSNR(= clean 필요)로 고르면 파이프라인이 label-free 가 아니게 된다.
그래서 선택 기준은 val noisy 에 대한 마스킹 loss 다. clean 기반 PSNR 도 같이 찍지만
그건 진단용일 뿐 선택에 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

from data import make_loader, resolve_test_noisy
from metrics import calculate_psnr, calculate_ssim
from models import build_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_DATA = Path(os.environ.get("DS_DATA", ROOT / "data" / "dataset"))


def blind_spot_mask(
    x: Tensor,
    ratio: float,
    window: int,
    n_shift: int = 8,
) -> tuple[Tensor, Tensor]:
    """일부 픽셀을 근처 이웃 값으로 덮어쓴다.

    이웃을 픽셀마다 하나씩 뽑는 대신 n_shift 개의 무작위 평행이동본을 만들어 두고
    픽셀마다 그중 하나를 고른다. 결과는 같고 훨씬 빠르다. (0,0) 이동은 제외한다 —
    자기 자신으로 덮으면 지운 게 아니다.

    반환: (덮어쓴 입력, 덮어쓴 자리 mask)
    """
    offsets: list[tuple[int, int]] = []
    while len(offsets) < n_shift:
        dy = int(torch.randint(-window, window + 1, (1,)).item())
        dx = int(torch.randint(-window, window + 1, (1,)).item())
        if (dy, dx) != (0, 0):
            offsets.append((dy, dx))

    shifted = torch.stack([torch.roll(x, (dy, dx), dims=(2, 3)) for dy, dx in offsets])
    pick = torch.randint(0, n_shift, x.shape, device=x.device)
    repl = shifted.gather(0, pick.unsqueeze(0)).squeeze(0)

    mask = torch.rand(x.shape, device=x.device) < ratio
    return torch.where(mask, repl, x), mask


def masked_loss(pred: Tensor, target: Tensor, mask: Tensor, kind: str = "l1") -> Tensor:
    """덮어쓴 자리에서만 loss. target 은 clean 이 아니라 원래 noisy 다."""
    d = pred - target
    per = d.abs() if kind == "l1" else d.pow(2)
    return (per * mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def evaluate_lf(net, loader, device, ratio, window, kind) -> float:
    """label-free 검증 — noisy 만 쓰는 마스킹 loss. 체크포인트 선택은 이 값으로 한다."""
    net.eval()
    torch.manual_seed(1234)  # 매번 같은 마스크가 나오도록
    tot, n = 0.0, 0
    for _, noisy, _ in loader:
        noisy = noisy.to(device)
        xin, mask = blind_spot_mask(noisy, ratio, window)
        tot += masked_loss(net(xin), noisy, mask, kind).item() * noisy.shape[0]
        n += noisy.shape[0]
    net.train()
    return tot / max(n, 1)


@torch.no_grad()
def diagnose_clean(net, loader, device) -> tuple[float, float]:
    """진단용 clean 기반 PSNR/SSIM. 체크포인트 선택에는 쓰지 않는다."""
    net.eval()
    ps, ss = [], []
    for label, noisy, _ in loader:
        label, noisy = label.to(device), noisy.to(device)
        out = net(noisy)
        ps.append(calculate_psnr(out, label).mean().item())
        ss.append(calculate_ssim(out, label).mean().item())
    net.train()
    return sum(ps) / len(ps), sum(ss) / len(ss)



def setup_amp(args, device):
    """(autocast dtype, GradScaler, 설명 문자열) 을 돌려준다.

    bf16 은 지수부가 fp32 와 같아 스케일링이 필요 없다 — GradScaler 를 끈다.
    """
    if not (getattr(args, "amp", True) and device.type == "cuda"):
        return torch.float32, torch.amp.GradScaler("cuda", enabled=False), "off"

    want = args.amp_dtype
    if want == "auto":
        want = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if want == "bf16" and not torch.cuda.is_bf16_supported():
        print("경고: 이 GPU 는 bf16 을 지원하지 않는다. fp16 으로 내려간다")
        want = "fp16"

    if want == "bf16":
        return torch.bfloat16, torch.amp.GradScaler("cuda", enabled=False), "bf16"
    return torch.float16, torch.amp.GradScaler("cuda", enabled=True), "fp16"

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="test", choices=["test", "train"],
                    help="test: test_noise_only 100장만 (clean 무접촉) / train: train clean 으로 noisy 합성")
    ap.add_argument("--model", default="dncnn", choices=["dncnn", "dncnn_plus", "drunet"])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--loss", default="l1", choices=["l1", "l2"])
    ap.add_argument("--mask-ratio", type=float, default=0.02, help="한 번에 덮어쓰는 픽셀 비율")
    ap.add_argument("--mask-window", type=int, default=2, help="이웃을 고르는 반경")
    ap.add_argument("--layers", type=int, default=17)
    ap.add_argument("--features", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp-dtype", default="auto", choices=["auto", "bf16", "fp16"],
                    help="auto: 지원하면 bf16. DRUNet 처럼 정규화 계층이 없는 큰 모델은 "
                         "fp16 에서 오버플로로 NaN 이 난다")
    ap.add_argument("--clip-grad", type=float, default=1.0,
                    help="gradient norm 상한. 0 이면 자르지 않는다")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.out is None:
        args.out = ROOT / "runs"
    torch.manual_seed(args.seed)
    ok_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if ok_cuda else "cpu")

    noisy_dir = resolve_test_noisy(args.data)

    if args.source == "test":
        train_loader, train_ds = make_loader(
            [noisy_dir], training_mode=True, batch=args.batch, num_workers=args.workers,
            patch=args.patch, already_noisy=True,
        )
        src_desc = f"test_noise_only {len(train_ds)}장 — clean 무접촉"
    else:
        train_loader, train_ds = make_loader(
            [args.data / "train"], training_mode=True, batch=args.batch,
            num_workers=args.workers, patch=args.patch,
        )
        src_desc = f"train {len(train_ds)}장에서 합성한 noisy — loss 에 clean 미사용"

    valid_loader, _ = make_loader([args.data / "val"], training_mode=False, batch=4, num_workers=0)

    net = build_model(args.model, num_of_layers=args.layers, features=args.features).to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.steps, eta_min=args.lr * 0.02)
    amp_dtype, scaler, amp_name = setup_amp(args, device)

    run_dir = args.out / f"{time.strftime('%m%d-%H%M')}_n2v-{args.source}{'_' + args.tag if args.tag else ''}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"run     : {run_dir}")
    print(f"source  : {src_desc}")
    print(f"model   : {args.model}  loss: masked {args.loss}  mask {args.mask_ratio:.1%} / 반경 {args.mask_window}")
    print(f"steps   : {args.steps}  batch {args.batch}  patch {args.patch}  lr {args.lr}"
          f"  amp {amp_name}  clip {args.clip_grad}")
    print("선택 기준: val noisy 마스킹 loss (label-free). clean PSNR 은 진단용이며 선택에 쓰지 않는다.\n")

    history: list[dict] = []
    best = {"lf": float("inf"), "step": -1, "psnr": float("nan"), "ssim": float("nan")}
    step, t0 = 0, time.time()
    skipped = 0  # loss 가 유한하지 않아 건너뛴 스텝
    it = iter(train_loader)

    while step < args.steps:
        try:
            _, noisy, _ = next(it)
        except StopIteration:
            it = iter(train_loader)
            continue

        noisy = noisy.to(device, non_blocking=True)
        xin, mask = blind_spot_mask(noisy, args.mask_ratio, args.mask_window)

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_name != "off", dtype=amp_dtype):
            loss = masked_loss(net(xin), noisy, mask, args.loss)

        if not torch.isfinite(loss):
            skipped += 1
            if skipped > 50:
                raise SystemExit(
                    f"loss 가 {skipped}번 발산했다 (step {step}). "
                    "--amp-dtype bf16 이나 --lr 을 낮춰서 다시 시도할 것."
                )
            continue
        skipped = 0

        scaler.scale(loss).backward()
        if args.clip_grad > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
        scaler.step(optim)
        scaler.update()
        sched.step()
        step += 1

        if step % args.eval_every == 0 or step == args.steps:
            lf = evaluate_lf(net, valid_loader, device, args.mask_ratio, args.mask_window, args.loss)
            psnr, ssim = diagnose_clean(net, valid_loader, device)
            history.append({"step": step, "train_loss": loss.item(), "val_lf_loss": lf,
                            "diag_psnr": psnr, "diag_ssim": ssim, "sec": time.time() - t0})
            mark = ""
            if lf < best["lf"]:
                best = {"lf": lf, "step": step, "psnr": psnr, "ssim": ssim}
                torch.save({"model": args.model, "layers": args.layers, "features": args.features,
                            "state_dict": net.state_dict(), "epoch": step,
                            "val_psnr": psnr, "val_ssim": ssim, "val_lf_loss": lf,
                            "label_free": True, "source": args.source},
                           run_dir / "checkpoints" / "checkpoint_best.ckpt")
                mark = "  <- best (label-free 기준)"
            print(f"[{step:6d}/{args.steps}] masked-loss {lf:.5f}   진단 PSNR {psnr:.3f} SSIM {ssim:.4f}"
                  f"   {time.time() - t0:.0f}s{mark}", flush=True)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\n총 {time.time() - t0:.0f}s. best step {best['step']} — masked-loss {best['lf']:.5f}")
    print(f"  (그 시점 진단 PSNR {best['psnr']:.3f} SSIM {best['ssim']:.4f})")
    print(f"checkpoint: {run_dir / 'checkpoints' / 'checkpoint_best.ckpt'}")


if __name__ == "__main__":
    main()
