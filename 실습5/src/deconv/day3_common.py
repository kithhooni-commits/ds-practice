"""3일차 공용 — 데이터 로딩과 **val 에서 튜닝하는** 규약.

## test 는 채점에만 쓴다

배포 안내가 명시한다 — `K` 는 validation set 에서 sweep 해서 고르고,
`noise_meta.json` 은 결과 분석에만 쓴다. 학습이 아니더라도 **하이퍼파라미터를 test 로
고르면 test 를 쓴 것**이다. 그래서 이 모듈은 두 가지를 분리한다.

    load_val()    val 100장에 forward + 노이즈를 걸어 만든다. 튜닝은 여기서만
    load_test()   배포된 test_deconv_noise. 최종 채점에만

val 의 노이즈는 **파일명 기반 seed 로 고정**한다. 1일차·2일차와 같은 방식이고,
매번 값이 흔들리면 튜닝이 무의미해진다.

`test_deconv_noise` 는 `dipole conv → noise` 순서로 만들어졌고 (배포 안내),
노이즈 4종·σ 가 1일차와 파일별로 동일하다. val 을 같은 방식으로 만들면 분포가 맞는다.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))
from data import RandomNoiseSimulator  # noqa: E402

from challenge import forward  # noqa: E402

DEFAULT_DATA = Path(os.environ.get("DS_DATA", ROOT / "data" / "dataset"))
NZ = ["gaussian", "rician", "uniform", "salt_and_pepper"]
_SIM = RandomNoiseSimulator()


def load_val(data: Path = DEFAULT_DATA, n: int = 0, device=None):
    """val clean 에 forward + 고정 seed 노이즈를 걸어 (측정치, 정답) 을 만든다.

    **튜닝은 여기서만 한다.** 반환 형식은 load_test 와 같다.
    """
    files = sorted(glob.glob(str(data / "val" / "*.npy")))
    if n:
        files = files[:n]
    out = []
    for f in files:
        gt = np.load(f).astype(np.float64)
        g = torch.from_numpy(forward(gt).astype(np.float32))[None, None]
        g = _SIM(g, seed=zlib.crc32(Path(f).name.encode()))
        b = torch.from_numpy(gt.astype(np.float32))[None, None]
        if device is not None:
            g, b = g.to(device), b.to(device)
        out.append(("val", g, b))
    return out


def load_test(data: Path = DEFAULT_DATA, n: int = 0, device=None, with_type: bool = True):
    """배포된 test_deconv_noise. **채점에만 쓴다.**

    with_type 이면 noise_meta.json 의 종류를 같이 돌려준다 — 배포 안내가 '결과 분석에만'
    쓰라고 한 정보다. 표를 종류별로 쪼개는 데만 쓰고 복원에는 쓰지 않는다.
    """
    src = data / "test_deconv_noise"
    meta = json.loads((src / "noise_meta.json").read_text(encoding="utf-8"))
    if n:
        meta = meta[:n]
    out = []
    for r in meta:
        g = torch.from_numpy(np.load(src / r["file"]).astype(np.float32))[None, None]
        b = torch.from_numpy(np.load(data / "test_label" / r["file"]).astype(np.float32))[None, None]
        if device is not None:
            g, b = g.to(device), b.to(device)
        out.append((r["noise_type"] if with_type else "test", g, b))
    return out


def score(items, fn) -> tuple[float, float]:
    from metrics import calculate_psnr, calculate_ssim

    ps, ss = [], []
    with torch.no_grad():
        for _, g, gt in items:
            e = fn(g)
            ps.append(calculate_psnr(e, gt).item())
            ss.append(calculate_ssim(e, gt).item())
    return float(np.mean(ps)), float(np.mean(ss))


def tune(val_items, configs, make_fn, label: str = "튜닝"):
    """val 에서 가장 좋은 설정을 고른다. configs 는 (이름, 인자) 목록."""
    print(f"[{label} — val {len(val_items)}장]")
    print(f"{'설정':<26}{'PSNR':>10}{'SSIM':>10}")
    print("-" * 46)
    best = (None, -1.0, 0.0)
    for name, cfg in configs:
        p, s = score(val_items, make_fn(cfg))
        if p > best[1]:
            best = (cfg, p, s)
        print(f"{str(name):<26}{p:>10.2f}{s:>10.4f}")
    print(f"→ 선택: {best[0]}   val {best[1]:.2f} / {best[2]:.4f}\n")
    return best[0]


def report(test_items, fn, label: str) -> tuple[float, float]:
    """test 에서 최종 점수. 노이즈 종류별로 쪼개 보여준다 (결과 분석 목적)."""
    from metrics import calculate_psnr, calculate_ssim

    rows = []
    with torch.no_grad():
        for nz, g, gt in test_items:
            e = fn(g)
            rows.append((nz, calculate_psnr(e, gt).item(), calculate_ssim(e, gt).item()))
    print(f"[{label} — test {len(rows)}장]")
    print(f"{'noise':<18}{'n':>4}{'PSNR':>10}{'SSIM':>10}")
    print("-" * 42)
    for nz in NZ + ["ALL"]:
        s = [r for r in rows if nz == "ALL" or r[0] == nz]
        if s:
            print(f"{nz:<18}{len(s):>4}{np.mean([r[1] for r in s]):>10.2f}"
                  f"{np.mean([r[2] for r in s]):>10.4f}")
    return float(np.mean([r[1] for r in rows])), float(np.mean([r[2] for r in rows])), rows
