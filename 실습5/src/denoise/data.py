"""데이터셋과 노이즈 시뮬레이터.

노이즈 4종의 정의와 sigma 범위는 과제 제공 코드와 동일하게 맞췄다. 학습셋에는
정답(clean)만 있고 noisy 는 매 epoch 새로 합성한다 — 같은 이미지도 매번 다른
노이즈를 보게 되므로 사실상 무한 증강이다.

제공 코드와 다르게 한 곳:
  * 파일명 추출을 `split("/")` 대신 `Path(...).name` 으로 한다.
    Windows 에서 glob 이 역슬래시를 돌려주기 때문에 원본은 경로 전체가 이름이 된다.
  * 학습 시 랜덤 크롭(기본 128)과 rot90 을 추가했다. 6GB GPU 에서 256² 배치 16 은
    올라가지 않고, DnCNN 은 완전 합성곱이라 패치로 학습해도 추론은 256² 로 된다.
"""

from __future__ import annotations

import glob
import random
import zlib
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# 과제 제공 값
NOISE_RANGES: dict[str, tuple[float, float]] = {
    "gaussian": (0.0, 0.1),
    "rician": (0.0, 0.15),
    "uniform": (0.0, 0.2),
    "salt_and_pepper": (0.0, 0.2),
}


class DataKey(IntEnum):
    Label = 0
    Noisy = 1
    Name = 2


# ---------------------------------------------------------------- 노이즈 4종


def gaussian_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    return img + torch.randn_like(img) * sigma


def rician_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    noise_real = torch.randn_like(img) * sigma
    noise_imag = torch.randn_like(img) * sigma
    return torch.abs(img + noise_real + 1j * noise_imag)


def uniform_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    return img + (torch.rand_like(img) * 2.0 - 1.0) * sigma


def salt_and_pepper_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    noisy = img.clone()
    total = img.numel()
    for prob, value in ((sigma / 2, img.max()), (sigma / 2, 0.0)):
        n = int(total * prob)
        coords = [torch.randint(0, dim, (n,)) for dim in img.shape]
        noisy[tuple(coords)] = value
    return noisy


NOISE_FN = {
    "gaussian": gaussian_noise,
    "rician": rician_noise,
    "uniform": uniform_noise,
    "salt_and_pepper": salt_and_pepper_noise,
}


class RandomNoiseSimulator:
    """이미지마다 4종 중 하나를 뽑고 해당 범위에서 sigma 를 뽑아 적용한다."""

    def __init__(self, noise_ranges: dict[str, tuple[float, float]] | None = None) -> None:
        self.noise_ranges = dict(noise_ranges) if noise_ranges is not None else dict(NOISE_RANGES)
        self.names = list(self.noise_ranges)

    def _sample(self, rng) -> tuple[str, float]:
        name = rng.choice(self.names)
        low, high = self.noise_ranges[name]
        return name, rng.uniform(low, high)

    def __call__(self, img: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        if seed is None:
            name, sigma = self._sample(random)
            return NOISE_FN[name](img, sigma)

        # 검증/테스트: 파일마다 항상 같은 노이즈가 나오도록 고정
        name, sigma = self._sample(random.Random(seed))
        state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            return NOISE_FN[name](img, sigma)
        finally:
            torch.random.set_rng_state(state)

    def describe(self, seed: int) -> tuple[str, float]:
        return self._sample(random.Random(seed))


def resolve_test_noisy(data_root: Path) -> Path:
    """`test_noise_only` 의 실제 위치를 찾는다.

    배포 zip 을 풀면 `dataset/test_noise_only/test_noise_only/` 로 한 겹 더 들어가지만,
    Colab 안내대로 Drive 에 올리면 `dataset/test_noise_only/` 한 겹이다. 둘 다 받는다.
    """
    root = Path(data_root)
    base = root / "test_noise_only"
    nested = base / "test_noise_only"
    if nested.is_dir() and any(nested.glob("*.npy")):
        return nested
    if base.is_dir() and any(base.glob("*.npy")):
        return base

    # 어디가 틀렸는지 바로 보이게 실제로 뭐가 있는지 찍어 준다
    lines = [f"test_noise_only 안에 .npy 가 없다.", f"  data root : {root}"]
    if not root.is_dir():
        lines.append("  -> 이 경로 자체가 없다. --data 나 DS_DATA 를 확인할 것")
    else:
        entries = sorted(x.name + ("/" if x.is_dir() else "") for x in list(root.iterdir())[:20])
        lines.append(f"  안에 있는 것: {', '.join(entries) or '(비어 있음)'}")
        if base.is_dir():
            sub = sorted(x.name + ("/" if x.is_dir() else "") for x in list(base.iterdir())[:20])
            lines.append(f"  test_noise_only/ 안: {', '.join(sub) or '(비어 있음)'}")
        else:
            lines.append("  test_noise_only/ 가 없다 — data root 가 dataset 폴더를 가리키는지 확인할 것")
    raise FileNotFoundError(chr(10).join(lines))


def name_seed(name: str) -> int:
    return zlib.crc32(name.encode())


# ---------------------------------------------------------------- 데이터셋


class DenoiseDataset(Dataset):
    def __init__(
        self,
        dirs: list[str | Path],
        training_mode: bool,
        noisy_dir: str | Path | None = None,
        patch: int | None = None,
        max_images: int | None = None,
        pattern: str = "*.npy",
    ) -> None:
        super().__init__()
        files: list[str] = []
        for d in dirs:
            files += glob.glob(str(Path(d) / pattern))
        files = sorted(files)
        if max_images is not None:
            files = files[:max_images]
        if not files:
            raise FileNotFoundError(f"no {pattern} under {dirs}")

        self.files = files
        self.training_mode = training_mode
        self.noisy_dir = Path(noisy_dir) if noisy_dir is not None else None
        self.patch = patch
        self.sim = RandomNoiseSimulator()

    @staticmethod
    def _load(path: str | Path) -> torch.Tensor:
        img = torch.from_numpy(np.load(str(path))).float()
        return img.unsqueeze(0) if img.dim() == 2 else img

    def _augment(self, label: torch.Tensor, noisy: torch.Tensor | None = None):
        if random.random() < 0.5:
            label = torch.flip(label, dims=[1])
            noisy = torch.flip(noisy, dims=[1]) if noisy is not None else None
        if random.random() < 0.5:
            label = torch.flip(label, dims=[2])
            noisy = torch.flip(noisy, dims=[2]) if noisy is not None else None
        k = random.randint(0, 3)
        if k:
            label = torch.rot90(label, k, dims=[1, 2])
            noisy = torch.rot90(noisy, k, dims=[1, 2]) if noisy is not None else None
        return label, noisy

    def _crop(self, label: torch.Tensor, noisy: torch.Tensor | None = None):
        p = self.patch
        if p is None or label.shape[-1] <= p:
            return label, noisy
        y = random.randint(0, label.shape[-2] - p)
        x = random.randint(0, label.shape[-1] - p)
        label = label[:, y : y + p, x : x + p]
        noisy = noisy[:, y : y + p, x : x + p] if noisy is not None else None
        return label, noisy

    def __getitem__(self, idx: int):
        path = self.files[idx]
        name = Path(path).name
        label = self._load(path)

        if self.noisy_dir is None:
            if self.training_mode:
                label, _ = self._augment(label)
                label, _ = self._crop(label)
                # 크롭 뒤에 노이즈를 얹는다. salt&pepper 는 픽셀 수에 비례하므로
                # 크롭 전에 얹으면 패치마다 밀도가 흔들린다.
                noisy = self.sim(label)
            else:
                noisy = self.sim(label, seed=name_seed(name))
        else:
            noisy_file = self.noisy_dir / name
            if not noisy_file.exists():
                raise FileNotFoundError(noisy_file)
            noisy = self._load(noisy_file)
            if self.training_mode:
                label, noisy = self._augment(label, noisy)
                label, noisy = self._crop(label, noisy)

        return label, noisy, name

    def __len__(self) -> int:
        return len(self.files)


def make_loader(
    dirs: list[str | Path],
    training_mode: bool,
    batch: int,
    num_workers: int = 0,
    shuffle: bool | None = None,
    **kwargs,
) -> tuple[DataLoader, DenoiseDataset]:
    ds = DenoiseDataset(dirs, training_mode=training_mode, **kwargs)
    loader = DataLoader(
        ds,
        batch_size=batch,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        shuffle=training_mode if shuffle is None else shuffle,
        drop_last=training_mode,
    )
    return loader, ds
