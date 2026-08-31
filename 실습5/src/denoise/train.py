"""DnCNN 계열 디노이저 학습.

제공된 Colab 노트북과 같은 데이터·같은 노이즈 합성·같은 지표를 쓰되, 학습 쪽만
손봤다. 바꾼 것과 이유:

  patch 128 랜덤 크롭  256² 배치 16 은 6GB GPU 에 안 올라간다. DnCNN 은 전부
                       합성곱이라 패치로 배우고 256² 로 추론해도 문제가 없다.
  Charbonnier loss     L2 는 salt&pepper 처럼 극단값이 섞이면 그 몇 픽셀에 끌려간다.
                       Charbonnier(=smooth L1)는 큰 오차의 영향을 선형으로 제한한다.
  cosine LR            제공 설정은 plateau 마다 0.88배. 10 epoch 밖에 안 돌리면
                       거의 감쇠가 안 걸린다. 길게 돌릴 거라면 cosine 이 낫다.
  rot90 증강           flip 만으로는 방향 다양성이 부족하다.

검증은 val 100장에 파일명 기반 고정 seed 노이즈를 얹어서 잰다 (제공 코드와 동일).
best 는 validation PSNR 기준.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

from data import make_loader
from metrics import calculate_psnr, calculate_ssim
from models import build_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # 실습5/
DEFAULT_DATA = Path(os.environ.get("DS_DATA", ROOT / "data" / "dataset"))


class Charbonnier(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


LOSSES = {"l1": nn.L1Loss, "l2": nn.MSELoss, "charbonnier": Charbonnier}


@torch.no_grad()
def validate(net: nn.Module, loader, device) -> tuple[float, float]:
    net.eval()
    psnrs, ssims = [], []
    for label, noisy, _ in loader:
        label, noisy = label.to(device), noisy.to(device)
        out = net(noisy)
        psnrs.append(calculate_psnr(out, label).mean().item())
        ssims.append(calculate_ssim(out, label).mean().item())
    net.train()
    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dncnn", choices=["dncnn", "dncnn_plus", "drunet"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--loss", default="charbonnier", choices=list(LOSSES))
    ap.add_argument("--layers", type=int, default=17)
    ap.add_argument("--features", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--tag", default="")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset 폴더 (train/ val/ 를 담고 있는)")
    ap.add_argument("--out", type=Path, default=None, help="run 저장 위치 (기본: <data>/../runs 가 아니라 저장소 runs/)")
    args = ap.parse_args()

    if args.out is None:
        args.out = ROOT / "runs"
    if not (args.data / "train").exists():
        raise SystemExit(f"--data 아래에 train/ 이 없다: {args.data}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = args.out / f"{time.strftime('%m%d-%H%M')}_{args.model}{'_' + args.tag if args.tag else ''}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_loader, train_ds = make_loader(
        [args.data / "train"], training_mode=True, batch=args.batch, num_workers=args.workers, patch=args.patch
    )
    valid_loader, valid_ds = make_loader([args.data / "val"], training_mode=False, batch=1, num_workers=0)

    net = build_model(args.model, num_of_layers=args.layers, features=args.features).to(device)
    n_param = sum(p.numel() for p in net.parameters())

    optim = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * len(train_loader), eta_min=args.lr * 0.02)
    crit = LOSSES[args.loss]()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    print(f"run     : {run_dir}")
    print(f"device  : {device}  model: {args.model}  params: {n_param:,}")
    print(f"train   : {len(train_ds)} images, patch {args.patch}, batch {args.batch} -> {len(train_loader)} iters/epoch")
    print(f"valid   : {len(valid_ds)} images (파일명 고정 seed 노이즈)")
    print(f"loss    : {args.loss}   lr: {args.lr} (cosine)   amp: {scaler.is_enabled()}\n")

    history: list[dict] = []
    best = {"psnr": -1.0, "epoch": -1}
    t0 = time.time()

    for epoch in range(args.epochs):
        running, n = 0.0, 0
        te = time.time()
        for it, (label, noisy, _) in enumerate(train_loader):
            label, noisy = label.to(device, non_blocking=True), noisy.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = crit(net(noisy), label)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += loss.item() * label.shape[0]
            n += label.shape[0]
            if it % 100 == 0:
                print(f"  ep {epoch:02d} it {it:4d}/{len(train_loader)} loss {running / max(n, 1):.5f}", flush=True)

        psnr, ssim = validate(net, valid_loader, device)
        rec = {
            "epoch": epoch,
            "loss": running / max(n, 1),
            "val_psnr": psnr,
            "val_ssim": ssim,
            "lr": sched.get_last_lr()[0],
            "sec": time.time() - te,
        }
        history.append(rec)

        mark = ""
        if psnr > best["psnr"]:
            best = {"psnr": psnr, "ssim": ssim, "epoch": epoch}
            torch.save(
                {"model": args.model, "layers": args.layers, "features": args.features,
                 "state_dict": net.state_dict(), "epoch": epoch, "val_psnr": psnr, "val_ssim": ssim},
                run_dir / "checkpoints" / "checkpoint_best.ckpt",
            )
            mark = "  <- best"
        print(
            f"[ep {epoch:02d}] loss {rec['loss']:.5f}  val PSNR {psnr:.3f}  SSIM {ssim:.4f}"
            f"  lr {rec['lr']:.2e}  {rec['sec']:.0f}s{mark}",
            flush=True,
        )
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\n총 {time.time() - t0:.0f}s. best epoch {best['epoch']} — val PSNR {best['psnr']:.3f} SSIM {best['ssim']:.4f}")
    print(f"checkpoint: {run_dir / 'checkpoints' / 'checkpoint_best.ckpt'}")


if __name__ == "__main__":
    main()
