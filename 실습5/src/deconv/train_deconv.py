"""2일차 deconvolution 학습.

## 입력을 무엇으로 줄 것인가

이 문제의 핵심 설계 결정이다. 세 가지를 비교한다.

    --input measure   흐린 측정치를 그대로 넣는다. 배포 노트북의 방식.
                      네트워크가 역연산 전체를 배워야 한다.
    --input wiener    Wiener 로 먼저 역산하고 그 결과를 넣는다.
                      역연산은 해석적으로 끝내고 네트워크는 남은 줄무늬만 지운다.
    --input both      둘 다 채널로 준다. 네트워크가 알아서 섞는다.

노이즈가 없으면 Wiener 만으로 118 dB 라 학습할 것이 없다. 그래서 `--noise` 로
측정치에 노이즈를 얹어 문제를 실제로 어렵게 만든 조건도 같이 본다 — 3일차가 정확히
그 조건이다.

## loss

    l2 / charbonnier   출력과 정답을 직접 비교
    model_loss         (1-w)·L2(출력, 정답) + w·L1(dipole(출력), 측정치)
                       배포 노트북의 방식. 복원 결과를 다시 흐리게 만들면 측정치와
                       같아야 한다는 물리 제약을 손실에 넣는다.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))
from metrics import calculate_psnr, calculate_ssim  # noqa: E402
from models import build_model  # noqa: E402

from challenge import dipole_otf  # noqa: E402
from run_challenge import adaptive_K  # noqa: E402

DEFAULT_DATA = Path(os.environ.get("DS_DATA", ROOT / "data" / "dataset"))


# ------------------------------------------------------------------ 연산자


def dipole_torch(shape: tuple[int, int], device) -> Tensor:
    return torch.from_numpy(dipole_otf(shape)).float().to(device)


def forward_t(img: Tensor) -> Tensor:
    """g = ifft2(fft2(f) · D). 배포 ForwardSimulator 와 동일 (노이즈 없음)."""
    D = dipole_torch(tuple(img.shape[-2:]), img.device)
    return torch.fft.ifft2(torch.fft.fft2(img) * D).real


def wiener_t(measure: Tensor, K: float | Tensor) -> Tensor:
    D = dipole_torch(tuple(measure.shape[-2:]), measure.device)
    h2 = D**2
    safe = torch.where(D.abs() < 1e-12, torch.full_like(D, 1e-12), D)
    if isinstance(K, Tensor):
        K = K.view(-1, 1, 1, 1)
    W = (1.0 / safe) * (h2 / (h2 + K))
    return torch.fft.ifft2(torch.fft.fft2(measure) * W).real


# ------------------------------------------------------------------ 데이터


class DeconvDataset(Dataset):
    """clean 을 읽어 dipole 로 흐리게 만든다. 노이즈는 선택."""

    def __init__(self, root: Path, training: bool, patch: int | None = None,
                 noise: float = 0.0, noise_random: bool = False,
                 input_mode: str = "measure") -> None:
        self.files = sorted(glob.glob(str(root / "*.npy")))
        if not self.files:
            raise FileNotFoundError(root)
        self.training = training
        self.patch = patch
        self.noise = noise
        self.noise_random = noise_random
        # Wiener 역산은 **크롭 전 전체 이미지**에서 해야 한다. dipole 은 전역 연산이라
        # 128² 조각의 FFT 는 256² 원본과 커널 자체가 다르다.
        self.input_mode = input_mode

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        gt = torch.from_numpy(np.load(self.files[i])).float().unsqueeze(0)

        if self.training:
            if random.random() < 0.5:
                gt = torch.flip(gt, dims=[1])
            if random.random() < 0.5:
                gt = torch.flip(gt, dims=[2])
            k = random.randint(0, 3)
            if k:
                gt = torch.rot90(gt, k, dims=[1, 2])

        # 크롭은 흐리게 만든 **뒤에** 한다. dipole 은 전역 연산이라 자르고 흐리게 하면
        # 경계에서 실제와 다른 측정치가 만들어진다.
        g = forward_t(gt.unsqueeze(0)).squeeze(0)

        sigma = self.noise
        if self.noise_random and self.noise > 0:
            sigma = random.uniform(0.0, self.noise)
        if sigma > 0:
            g = g + torch.randn_like(g) * sigma

        # 네트워크 입력을 전체 해상도에서 만든다
        if self.input_mode == "measure":
            net_in = g
        else:
            K = adaptive_K(g.squeeze(0).double().numpy())
            w = wiener_t(g.unsqueeze(0), K).squeeze(0).float()
            net_in = w if self.input_mode == "wiener" else torch.cat([g, w], dim=0)

        if self.patch and gt.shape[-1] > self.patch:
            p = self.patch
            y = random.randint(0, gt.shape[-2] - p)
            x = random.randint(0, gt.shape[-1] - p)
            gt = gt[:, y:y + p, x:x + p]
            g = g[:, y:y + p, x:x + p]
            net_in = net_in[:, y:y + p, x:x + p]

        return gt, g, net_in, Path(self.files[i]).name


class ModelLoss(nn.Module):
    """배포 노트북과 동일: (1-w)·L2(출력,정답) + w·L1(dipole(출력), 측정치)."""

    def __init__(self, weight: float = 0.8) -> None:
        super().__init__()
        self.w = weight

    def forward(self, out: Tensor, target: Tensor, measure: Tensor) -> Tensor:
        md = (forward_t(out) - measure).abs().mean()
        l2 = ((out - target) ** 2).mean()
        return (1 - self.w) * l2 + self.w * md


class Charbonnier(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target, measure=None):
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


# ------------------------------------------------------------------ 학습


@torch.no_grad()
def validate(net, loader, device, mode) -> tuple[float, float]:
    net.eval()
    ps, ss = [], []
    for gt, g, net_in, _ in loader:
        gt, net_in = gt.to(device), net_in.to(device)
        out = net(net_in)
        ps.append(calculate_psnr(out, gt).mean().item())
        ss.append(calculate_ssim(out, gt).mean().item())
    net.train()
    return sum(ps) / len(ps), sum(ss) / len(ss)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="drunet", choices=["dncnn", "drunet"])
    ap.add_argument("--input", default="measure", choices=["measure", "wiener", "both"])
    ap.add_argument("--loss", default="charbonnier", choices=["l2", "charbonnier", "model_loss"])
    ap.add_argument("--model-loss-weight", type=float, default=0.8)
    ap.add_argument("--noise", type=float, default=0.0, help="측정치에 얹을 노이즈 σ (0 이면 배포 조건)")
    ap.add_argument("--noise-random", action="store_true", help="[0, σ] 에서 매번 다시 뽑는다")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--features", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if ok else "cpu")
    amp_dtype = torch.bfloat16 if ok and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=ok and amp_dtype is torch.float16)

    in_ch = 2 if args.input == "both" else 1
    net = build_model(args.model, features=args.features, num_of_layers=17).to(device)
    if in_ch == 2:  # 첫 층만 2채널로 갈아끼운다
        first = net.dncnn[0] if args.model == "dncnn" else net.head
        new = nn.Conv2d(2, first.out_channels, first.kernel_size, first.stride,
                        first.padding, bias=False).to(device)
        with torch.no_grad():
            new.weight[:, :1] = first.weight
            new.weight[:, 1:] = first.weight
        if args.model == "dncnn":
            net.dncnn[0] = new
        else:
            net.head = new

    train_ds = DeconvDataset(args.data / "train", True, args.patch, args.noise,
                             args.noise_random, args.input)
    valid_ds = DeconvDataset(args.data / "val", False, None, args.noise, False, args.input)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    valid_loader = DataLoader(valid_ds, batch_size=1, num_workers=0)

    crit = {"l2": nn.MSELoss(), "charbonnier": Charbonnier(),
            "model_loss": ModelLoss(args.model_loss_weight)}[args.loss]
    optim = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * len(train_loader), eta_min=args.lr * 0.02)

    run = args.out / f"{time.strftime('%m%d-%H%M')}_deconv-{args.input}{'_' + args.tag if args.tag else ''}"
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()},
                                                ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"run    : {run}")
    print(f"model  : {args.model} f{args.features} | 입력 {args.input} ({in_ch}ch) | loss {args.loss}")
    print(f"노이즈 : σ={args.noise}{' (매번 [0,σ] 에서 랜덤)' if args.noise_random else ''}")
    print(f"학습   : {len(train_ds)}장 patch {args.patch} batch {args.batch} -> {len(train_loader)} iter/ep, {args.epochs} ep")
    print(f"amp    : {amp_dtype} | clip {args.clip_grad}\n")

    best, hist, skipped = {"psnr": -1.0, "epoch": -1}, [], 0
    t0 = time.time()
    for ep in range(args.epochs):
        run_loss, n, te = 0.0, 0, time.time()
        for it, (gt, g, net_in, _) in enumerate(train_loader):
            gt = gt.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)
            net_in = net_in.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=ok, dtype=amp_dtype):
                out = net(net_in)
                loss = crit(out, gt, g) if args.loss == "model_loss" else crit(out, gt)
            if not torch.isfinite(loss):
                skipped += 1
                if skipped > 50:
                    raise SystemExit(f"loss 가 {skipped}번 발산했다 (ep {ep}, it {it}). lr 을 낮출 것.")
                continue
            skipped = 0
            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
            scaler.step(optim)
            scaler.update()
            sched.step()
            run_loss += loss.item() * gt.shape[0]
            n += gt.shape[0]
            if it % 100 == 0:
                print(f"  ep {ep:02d} it {it:4d}/{len(train_loader)} loss {run_loss / max(n, 1):.5f}", flush=True)

        psnr, ssim = validate(net, valid_loader, device, args.input)
        hist.append({"epoch": ep, "loss": run_loss / max(n, 1), "val_psnr": psnr,
                     "val_ssim": ssim, "lr": sched.get_last_lr()[0], "sec": time.time() - te})
        mark = ""
        if psnr > best["psnr"]:
            best = {"psnr": psnr, "ssim": ssim, "epoch": ep}
            torch.save({"model": args.model, "features": args.features, "input": args.input,
                        "in_ch": in_ch, "state_dict": net.state_dict(), "epoch": ep,
                        "val_psnr": psnr, "val_ssim": ssim},
                       run / "checkpoints" / "checkpoint_best.ckpt")
            mark = "  <- best"
        print(f"[ep {ep:02d}] loss {hist[-1]['loss']:.5f}  val PSNR {psnr:.3f}  SSIM {ssim:.4f}"
              f"  {hist[-1]['sec']:.0f}s{mark}", flush=True)
        (run / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")

    print(f"\n총 {time.time() - t0:.0f}s. best epoch {best['epoch']} — val PSNR {best['psnr']:.3f} SSIM {best['ssim']:.4f}")
    print(f"checkpoint: {run / 'checkpoints' / 'checkpoint_best.ckpt'}")


if __name__ == "__main__":
    main()
