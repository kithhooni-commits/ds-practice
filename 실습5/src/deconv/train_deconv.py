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
import zlib
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))
from data import RandomNoiseSimulator  # noqa: E402
from metrics import calculate_psnr, calculate_ssim  # noqa: E402
from models import build_model  # noqa: E402

from challenge import dipole_otf  # noqa: E402
from dcnet import DCNet  # noqa: E402
from spectral import SpectralNet  # noqa: E402
from run_challenge import adaptive_K  # noqa: E402
from twostage import TwoStageNet  # noqa: E402
from unrolled import UnrolledNet  # noqa: E402

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
                 input_mode: str = "measure", noise_model: str = "gaussian",
                 target: str = "label") -> None:
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
        # challenge: 1일차와 같은 4종(gaussian/rician/uniform/salt&pepper)을 흐림 뒤에 얹는다.
        # 3일차 test_deconv_noise 가 정확히 그렇게 만들어졌다 (파일별 종류·σ 가 1일차와 동일).
        self.noise_model = noise_model
        self.sim = RandomNoiseSimulator() if noise_model == "challenge" else None
        # target="measure": 노이즈 **없는** dipole blur 를 맞히게 한다. 배포 노트북의
        # 방법 B — 디컨볼루션은 학습이 아니라 Wiener 가 맡고, 네트워크는 측정치 영역에서
        # 노이즈만 지운다. 노이즈가 흐림 뒤에 붙었으므로 측정치 위에서는 백색이고,
        # 그게 디노이저가 가장 잘하는 조건이다.
        self.target = target

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
        g_clean = g.clone()   # 노이즈 얹기 전. target="measure" 의 정답

        if self.noise_model == "challenge":
            # 검증은 파일명 seed 로 고정해 epoch 마다 값이 흔들리지 않게 한다 (1일차와 같은 방식)
            seed = None if self.training else zlib.crc32(Path(self.files[i]).name.encode())
            g = self.sim(g, seed=seed)
            sigma = 0.0
        else:
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

        tgt = gt if self.target == "label" else g_clean
        return tgt, g, net_in, Path(self.files[i]).name


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


class CharbonnierSSIM(nn.Module):
    """Charbonnier + (1 − SSIM). **채점에 쓰는 그 SSIM 을 그대로 미분한다.**

    `metrics.calculate_ssim` 은 conv2d 와 사칙연산뿐이라 전부 미분 가능하다. 일반
    MS-SSIM 라이브러리를 쓰면 창 모양(11×11 box)과 data_range 가 달라 채점과 어긋나므로
    배포된 구현을 그대로 최적화한다.

    ## 왜 필요한가

    Charbonnier 만 쓰면 PSNR 은 오르는데 SSIM 이 안 따라온다. 실제로

        우리 전개형   26.85 dB / 0.7705
        배포 baseline 25.01 dB / 0.8149   <- PSNR 은 낮은데 SSIM 이 높다

    L1/L2 계열 손실은 **뭉개는 쪽**으로 수렴한다. 평균을 맞추면 오차 제곱은 줄지만
    국소 대비(variance)와 상관(covariance)은 잃는다. SSIM 이 재는 게 정확히 그것이다.
    이 데이터는 주기적인 격자 무늬라 국소 대비를 잃으면 SSIM 이 크게 깎인다.

    참고: Zhao et al., "Loss Functions for Image Restoration with Neural Networks",
    IEEE TCI 2017 — L1 과 SSIM 계열을 섞으면 둘 다 좋아진다.
    """

    def __init__(self, alpha: float = 0.5, eps: float = 1e-3) -> None:
        super().__init__()
        self.alpha = alpha
        self.eps2 = eps * eps

    def forward(self, pred, target, measure=None):
        l1 = torch.sqrt((pred - target) ** 2 + self.eps2).mean()
        ssim = calculate_ssim(pred, target).mean()
        return (1 - self.alpha) * l1 + self.alpha * (1 - ssim)


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
    ap.add_argument("--model", default="drunet",
                    choices=["dncnn", "unet", "drunet", "spectral", "spectral_dncnn",
                             "spectral_unet", "dcnet", "unrolled", "twostage"])
    ap.add_argument("--input", default="measure", choices=["measure", "wiener", "both"])
    ap.add_argument("--loss", default="charbonnier",
                    choices=["l2", "charbonnier", "model_loss", "charbonnier_ssim"])
    ap.add_argument("--ssim-weight", type=float, default=0.5,
                    help="charbonnier_ssim 에서 SSIM 항의 비중. 채점 SSIM 을 그대로 미분한다")
    ap.add_argument("--model-loss-weight", type=float, default=0.8)
    ap.add_argument("--noise", type=float, default=0.0, help="측정치에 얹을 노이즈 σ (0 이면 배포 조건)")
    ap.add_argument("--noise-random", action="store_true", help="[0, σ] 에서 매번 다시 뽑는다")
    ap.add_argument("--target", default="label", choices=["label", "measure"],
                    help="measure: 노이즈 없는 dipole blur 를 맞힌다. 디컨볼루션은 Wiener 가 맡는다 "
                         "(배포 노트북 방법 B). 평가 때 --post-wiener 로 K 를 준다")
    ap.add_argument("--noise-model", default="gaussian", choices=["gaussian", "challenge"],
                    help="challenge: 1일차 4종 노이즈를 흐림 뒤에 얹는다 (3일차 조건)")
    ap.add_argument("--unroll-iters", type=int, default=5)
    ap.add_argument("--lam-map", action="store_true",
                    help="λ 를 주파수마다 따로 학습한다. log 공간 파라미터라 --lr-spectral 은 "
                         "1e-3 ~ 3e-3 이 적당하다 (2일차 스펙트럼 층은 직접 배율이라 컸다)")
    ap.add_argument("--init-lam", type=float, default=3.16e-2,
                    help="λ 초기값. 기본은 원시 측정치의 val 최적 K")
    ap.add_argument("--refine-iters", type=int, default=0,
                    help="twostage: 역필터 뒤 이미지 영역 다듬기 횟수")
    ap.add_argument("--noise-stats", action="store_true",
                    help="σ 하나 대신 (σ, 왜도, 첨도) 를 조건으로 준다. 널 원뿔의 모양이 노이즈 "
                         "종류를 알려준다 — rician 첨도 10.39 vs 나머지 2.9~3.8. --sigma-map 필요")
    ap.add_argument("--sigma-map", action="store_true",
                    help="측정치에서 σ 를 읽어 디노이저에 조건으로 준다 (--refine drunet 전용). "
                         "3일차 σ 는 이미지마다 200배 차이가 난다")
    ap.add_argument("--init-refine", type=Path, default=None,
                    help="사전지식 자리를 1일차 디노이저 체크포인트로 초기화한다")
    ap.add_argument("--init-model", type=Path, default=None,
                    help="같은 구조의 체크포인트에서 이어서 학습한다. 손실을 바꿔 미세조정할 때 쓴다 "
                         "— Charbonnier 로 PSNR 을 벌어 두고 SSIM 을 얹는 식")
    ap.add_argument("--share-weights", action="store_true", default=True)
    ap.add_argument("--no-share-weights", dest="share_weights", action="store_false")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=0,
                    help="0 이면 크롭하지 않는다 (기본). deconvolution 은 전역 연산이라 "
                         "잘린 조각만 보면 복원에 필요한 정보가 조각 밖에 있다")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--features", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.05,
                    help="dcnet: |D| 가 이보다 크면 관측으로 정확히 복원하고 네트워크가 못 건드린다")
    ap.add_argument("--refine", default="unet", choices=["unet", "dncnn", "drunet"],
                    help="dcnet 이 null cone 을 채울 때 쓰는 신경망")
    ap.add_argument("--lr-spectral", type=float, default=0.5,
                    help="주파수 층 전용 학습률. 이득이 44,000 까지 가야 해서 훨씬 커야 한다")
    ap.add_argument("--warmup", type=int, default=200, help="lr 을 0 에서 올리는 step 수")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.patch == 0:
        args.patch = None

    torch.manual_seed(0)
    ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if ok else "cpu")
    amp_dtype = torch.bfloat16 if ok and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=ok and amp_dtype is torch.float16)

    in_ch = 2 if args.input == "both" else 1
    if args.model == "twostage":
        if args.patch:
            print("[주의] twostage 는 전역 연산이라 크롭을 못 한다. --patch 를 무시한다")
            args.patch = None
        net = TwoStageNet(model=args.refine, features=args.features,
                          sigma_map=args.sigma_map, lam_map=args.lam_map,
                          init_lam=args.init_lam, refine_iters=args.refine_iters).to(device)
        print(f"λ 초기값 {args.init_lam:.3g}  (원시 측정치의 val 최적 K. 디노이저가 "
              f"좋아지는 만큼 학습으로 내려간다)")
        if args.lam_map and args.lr_spectral > 5e-3:
            # 경고만 하면 오래된 셀을 다시 돌렸을 때 그대로 지나가 버린다. log 공간에서
            # lr 0.05 는 한 에폭에 λ 를 e^37 배 옮기고, clamp 에 부딪힌 주파수는 gradient 가
            # 0 이 되어 거기 얼어붙는다. 몇 시간을 버리느니 여기서 낮춘다.
            print(f"[자동 조정] --lam-map 은 log 공간이라 --lr-spectral {args.lr_spectral} 은 "
                  f"너무 크다 (한 에폭에 λ 가 e^{args.lr_spectral * 908:.0f} 배). 2e-3 으로 낮춘다. "
                  f"의도한 값이면 --lr-spectral 을 5e-3 이하로 직접 줄 것")
            args.lr_spectral = 2e-3
        if net.sigma_map:
            print("sigma-map: 측정치의 널 원뿔에서 σ 를 읽어 디노이저에 준다")
        if args.init_refine:
            ck = torch.load(args.init_refine, map_location="cpu", weights_only=False)
            # 1일차 디노이저인지 확인한다. 2·3일차 deconv 체크포인트는 "input" 키를
            # 갖는다 (train_deconv 가 저장). 그걸 사전지식 자리에 넣으면 이미지 영역
            # 역산을 배운 가중치를 측정치 영역 디노이저로 쓰는 셈이라 도움이 안 된다.
            # 받아도 되는 것은 두 가지다.
            #   (a) 1일차 디노이저 — "input" 키가 없다 (train.py 가 저장)
            #   (b) --target measure 로 학습한 측정치 영역 디노이저 — 이것이야말로
            #       이 자리가 할 일을 그대로 배운 가중치다 (배포 방법 B 의 1단계)
            # 막아야 하는 것은 --target label 로 학습한 **이미지 영역** 모델이다.
            # 그것은 역산까지 배운 가중치라 측정치 영역 디노이저로 쓰면 맞지 않는다.
            bad = ("unroll_iters" in ck and ck.get("model") in ("unrolled", "dcnet")) or (
                "input" in ck and ck.get("target", "label") == "label")
            if bad:
                raise SystemExit(
                    f"[중단] {Path(args.init_refine).name} 은 이미지 영역 모델이다 "
                    f"(model={ck.get('model')}, target={ck.get('target')}, "
                    f"val {ck.get('val_psnr', float('nan')):.2f}). 여기는 측정치 영역 디노이저 "
                    f"자리다 — 1일차 체크포인트나 --target measure 로 학습한 것을 줄 것")
            if ck.get("target") == "measure":
                print("측정치 영역 디노이저를 이어받는다 (배포 방법 B 의 1단계)")
            sd = ck.get("state_dict", ck)
            subs = [net.denoiser] + list(net.refiners)
            n_ok = n_skip = 0
            for sub in subs:
                tgt = sub.state_dict(); fit = {}
                for k, v in sd.items():
                    if k not in tgt:
                        continue
                    if tgt[k].shape == v.shape:
                        fit[k] = v
                    elif tgt[k].dim() == 4 and tgt[k].shape[1] == v.shape[1] + 1                             and tgt[k].shape[0] == v.shape[0]:
                        w = tgt[k].clone().zero_(); w[:, : v.shape[1]] = v; fit[k] = w
                    else:
                        n_skip += 1
                sub.load_state_dict(fit, strict=False); n_ok += len(fit)
            print(f"디노이저 초기화: {Path(args.init_refine).name} "
                  f"(1일차 val PSNR {ck.get('val_psnr', float('nan')):.2f}, "
                  f"{n_ok}/{len(sd) * len(subs)} 텐서 적재)")
    elif args.model == "unrolled":
        if args.patch:
            print("[주의] unrolled 는 전역 연산이라 크롭을 못 한다. --patch 를 무시한다")
            args.patch = None
        net = UnrolledNet(n_iter=args.unroll_iters, model=args.refine, features=args.features,
                          share_weights=args.share_weights, sigma_map=args.sigma_map,
                          lam_map=args.lam_map, noise_stats=args.noise_stats).to(device)
        if net.noise_stats:
            print("noise-stats: 널 원뿔에서 (σ, 왜도, 첨도) 를 읽어 준다 — rician 을 구분하기 위해서")
        if args.sigma_map and not net.sigma_map:
            print("[주의] --sigma-map 은 --refine drunet 에서만 쓴다. 무시한다")
        elif net.sigma_map:
            print("sigma-map: 측정치의 널 원뿔에서 σ 를 읽어 매 단계 디노이저에 준다")
        if args.init_refine:
            # 1일차 디노이저를 사전지식 자리의 출발점으로 쓴다. 무작위 초기화보다
            # 훨씬 나은 곳에서 시작한다 — 어차피 그 자리가 할 일이 '노이즈 지우기'다
            ck = torch.load(args.init_refine, map_location="cpu", weights_only=False)
            sd = ck.get("state_dict", ck)
            n_ok = n_skip = 0
            for sub in net.nets:
                tgt = sub.state_dict()
                fit = {}
                for k, v in sd.items():
                    if k not in tgt:
                        continue
                    if tgt[k].shape == v.shape:
                        fit[k] = v
                    elif tgt[k].dim() == 4 and tgt[k].shape[1] == v.shape[1] + 1 \
                            and tgt[k].shape[0] == v.shape[0]:
                        # σ 조건화로 입력 채널이 하나 늘었다. 원래 가중치는 0번 채널에
                        # 그대로 넣고 σ 채널은 0 으로 둔다 — 시작 시점엔 1일차
                        # 디노이저와 **정확히 같게** 동작하고, 거기서부터 σ 를 배운다
                        w = tgt[k].clone().zero_()
                        w[:, : v.shape[1]] = v
                        fit[k] = w
                    else:
                        n_skip += 1
                sub.load_state_dict(fit, strict=False)
                n_ok += len(fit)
            print(f"refine 초기화: {Path(args.init_refine).name} "
                  f"(1일차 val PSNR {ck.get('val_psnr', float('nan')):.2f}, "
                  f"{n_ok}/{len(sd) * len(net.nets)} 텐서 적재"
                  + (f", 모양 안 맞아 건너뜀 {n_skip}" if n_skip else "") + ")")
    elif args.model == "dcnet":
        if args.patch:
            print(f"[주의] dcnet 은 전역 연산이라 크롭을 못 한다. --patch 를 무시한다")
            args.patch = None
        net = DCNet(model=args.refine, features=args.features, tau=args.tau).to(device)
    elif args.model.startswith("spectral"):
        # 주파수 곱셈은 이미지 크기에 묶인다 — 크롭을 끈다
        if args.patch:
            print(f"[주의] spectral 은 크롭 학습을 못 한다. --patch {args.patch} 를 무시한다")
            args.patch = None
        if in_ch == 2:
            raise SystemExit("spectral 은 --input both 를 지원하지 않는다")
        refine = {"spectral": None, "spectral_dncnn": "dncnn",
                  "spectral_unet": "unet"}[args.model]
        net = SpectralNet(shape=(256, 256), refine=refine, features=args.features).to(device)
    else:
        net = build_model(args.model, features=args.features, num_of_layers=17).to(device)
    if in_ch == 2 and not args.model.startswith("spectral"):  # 첫 층만 2채널로 (가중치는 복사해서 이어받는다)
        if args.model == "dncnn":
            get, put = lambda: net.dncnn[0], lambda m: net.dncnn.__setitem__(0, m)
        elif args.model == "drunet":
            get, put = lambda: net.head, lambda m: setattr(net, "head", m)
        else:  # unet
            blk = net.down_sample_layers[0].layers
            get, put = lambda: blk[0], lambda m: blk.__setitem__(0, m)
        first = get()
        new_conv = nn.Conv2d(2, first.out_channels, first.kernel_size, first.stride,
                             first.padding, bias=first.bias is not None).to(device)
        with torch.no_grad():
            new_conv.weight[:, :1] = first.weight
            new_conv.weight[:, 1:] = first.weight
            if first.bias is not None:
                new_conv.bias.copy_(first.bias)
        put(new_conv)

    train_ds = DeconvDataset(args.data / "train", True, args.patch, args.noise,
                             args.noise_random, args.input, args.noise_model, args.target)
    valid_ds = DeconvDataset(args.data / "val", False, None, args.noise, False, args.input, args.noise_model, args.target)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    valid_loader = DataLoader(valid_ds, batch_size=1, num_workers=0)

    crit = {"l2": nn.MSELoss(), "charbonnier": Charbonnier(),
            "model_loss": ModelLoss(args.model_loss_weight),
            "charbonnier_ssim": CharbonnierSSIM(args.ssim_weight)}[args.loss]
    # 주파수 층은 이득이 수만까지 가야 하므로 학습률을 따로 크게 준다.
    # 같은 lr 을 쓰면 최대 4.8 배에서 멈춘다 (정답 44,074).
    spec_params, other_params = [], []
    for n_, p_ in net.named_parameters():
        # log_lam_map 도 주파수별 계수다 — 2일차에서 같은 lr 로는 안 움직였다
        is_spec = "spec." in n_ or "log_lam_map" in n_
        (spec_params if is_spec else other_params).append(p_)
    groups = [{"params": other_params, "lr": args.lr}]
    if spec_params:
        groups.append({"params": spec_params, "lr": args.lr_spectral})
        print(f"주파수 층 {sum(p_.numel() for p_ in spec_params):,}개 → lr {args.lr_spectral}")
    optim = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-5)

    total_steps = args.epochs * len(train_loader)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(total_steps - args.warmup, 1), eta_min=args.lr * 0.02)
    warm = torch.optim.lr_scheduler.LinearLR(optim, 0.01, 1.0, total_iters=max(args.warmup, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(
        optim, [warm, cosine], milestones=[max(args.warmup, 1)])

    run = args.out / f"{time.strftime('%m%d-%H%M')}_deconv-{args.input}{'_' + args.tag if args.tag else ''}"
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()},
                                                ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"run    : {run}")
    if args.init_model:
        # 손실을 바꿔 이어서 학습한다. Charbonnier 로 PSNR 을 벌어 두고 SSIM 을 얹는 식.
        # SSIM 을 처음부터 걸면 덜 학습된 모델을 "정답과 맞든 아니든 국소 대비를 키우는"
        # 쪽으로 밀어 PSNR 이 떨어진다 (실측: ep00 18.73 -> ep05 17.53).
        ck0 = torch.load(args.init_model, map_location="cpu", weights_only=False)
        sd0 = ck0.get("state_dict", ck0)
        want = (f"model={ck0.get('model')} refine={ck0.get('refine')} "
                f"features={ck0.get('features')} unroll_iters={ck0.get('unroll_iters')} "
                f"sigma_map={ck0.get('sigma_map')}")
        try:
            miss = net.load_state_dict(sd0, strict=False)
        except RuntimeError:
            # 텐서 모양이 다르면 수백 줄이 쏟아진다. 필요한 것만 보여준다
            raise SystemExit(
                f"[중단] 구조가 맞지 않는다. 체크포인트는 {want} 로 만들어졌다. "
                f"--init-model 은 같은 인자로 돌릴 때만 쓸 수 있다") from None
        if miss.missing_keys or miss.unexpected_keys:
            raise SystemExit(
                f"[중단] 구조가 다르다 — 없는 키 {len(miss.missing_keys)}개, "
                f"남는 키 {len(miss.unexpected_keys)}개. 체크포인트는 {want} 다")
        print(f"이어서 학습: {Path(args.init_model).name} "
              f"(ep {ck0.get('epoch')}, val {ck0.get('val_psnr', float('nan')):.2f} dB / "
              f"{ck0.get('val_ssim', float('nan')):.4f})")

    print(f"model  : {args.model} f{args.features} | 입력 {args.input} ({in_ch}ch) | loss {args.loss}")
    if args.noise_model == "challenge":
        print("노이즈 : 1일차 4종 (gaussian/rician/uniform/salt&pepper) 을 흐림 뒤에 — 3일차 조건")
    else:
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
                        "val_psnr": psnr, "val_ssim": ssim, "target": args.target,
                        "tau": args.tau, "refine": args.refine, "unroll_iters": args.unroll_iters,
                        "sigma_map": args.sigma_map, "share_weights": args.share_weights,
                        "lam_map": args.lam_map, "refine_iters": args.refine_iters,
                        "noise_stats": args.noise_stats},
                       run / "checkpoints" / "checkpoint_best.ckpt")
            mark = "  <- best"
        print(f"[ep {ep:02d}] loss {hist[-1]['loss']:.5f}  val PSNR {psnr:.3f}  SSIM {ssim:.4f}"
              f"  {hist[-1]['sec']:.0f}s{mark}", flush=True)
        (run / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")

    print(f"\n총 {time.time() - t0:.0f}s. best epoch {best['epoch']} — val PSNR {best['psnr']:.3f} SSIM {best['ssim']:.4f}")
    print(f"checkpoint: {run / 'checkpoints' / 'checkpoint_best.ckpt'}")


if __name__ == "__main__":
    main()
