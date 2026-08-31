# 실습5 — 반도체 이미지 노이즈 제거

삼성 DS2과정 **Image Restoration Challenge**. 4종의 노이즈로 망가진 흑백 반도체
이미지를 복원한다. 제출물은 test 100장에 대한 **PSNR_total / SSIM_total 두 숫자**다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kithhooni-commits/ds-practice/blob/main/%EC%8B%A4%EC%8A%B55/colab_denoising.ipynb)

**설명 자료(사전 지식 없이 읽는 배경) →** [explainer.html](explainer.html)
**출처 →** Day 1–3 강의자료 (Jongho Lee, SNU LIST) + 배포 코드 `code_denoising/`

> **3일 구조.** 이 챌린지는 3일짜리다 — **1일차 denoising**, 2일차 deconvolution,
> 3일차 둘의 결합(`g = h * f + n`). 이 문서는 1일차를 다룬다.
> 처음에 강의자료만 보고 과제를 dipole deconvolution 으로 잡았던 것은 오판이 아니라
> **2일차를 먼저 본 것**이었다. 그때 만든 forward 시뮬레이터·TKD·Wiener·Tikhonov·
> 노이즈 추정은 [`src/deconv/`](src/deconv/) 에 그대로 있고 2일차에 쓴다.
> label-free 가산점도 실재한다 — 아래 결과 참고.

## 문제 한 줄

이미지마다 **어떤 노이즈가 얼마나 실렸는지 모른 채** 하나의 네트워크로 4종을 전부
지워야 한다. 종류별로 최적 필터가 정반대다 — salt & pepper 에는 median 이,
gaussian 에는 mean/adaptive 가 맞다.

## 데이터

`dataset/` (1.9GB, git 미포함). 전부 `.npy`, float32, 256×256, 값 범위 대략 [0, 1].

| 폴더 | 장수 | 내용 |
|---|---|---|
| `train/` | 7,268 | clean. noisy 는 학습 중에 합성한다 |
| `val/` | 100 | clean. 검증용 noisy 는 파일명 기반 고정 seed 로 합성 |
| `test_noise_only/` | 100 | **손상 이미지** (+ `noise_meta.json`) |
| `test_label/` | 100 | test 정답. **제공된다** |

### 노이즈 4종

이미지마다 4종 중 하나를 균등하게 고르고, 아래 범위에서 σ 를 균등하게 뽑는다.

| 종류 | σ 범위 | 성질 |
|---|---|---|
| `gaussian` | 0 – 0.1 | `x + N(0, σ)` |
| `rician` | 0 – 0.15 | `\|x + N(0,σ) + iN(0,σ)\|` — 어두운 영역에서 값이 위로 들린다 |
| `uniform` | 0 – 0.2 | `x + U(-σ, σ)` |
| `salt_and_pepper` | 0 – 0.2 | 픽셀의 σ 비율을 0 또는 max 로 덮어쓴다 — **임펄스** |

`test_noise_only/noise_meta.json` 에 test 100장의 종류와 σ 가 다 적혀 있다.
이 저장소의 코드는 그 정보를 **표를 종류별로 쪼개는 데만** 쓰고 복원에는 쓰지 않는다.
채점 세트에 메타가 없을 수도 있고, 무엇보다 종류를 모르고도 되는 게 더 강한 결과다.

## 과제 스펙

| 항목 | 값 |
|---|---|
| 평가 지표 | PSNR, SSIM (구현이 배포 코드에 포함 — 그대로 써야 한다) |
| 제출 | `denoising_challenge_score.xlsx` — 성함 / PSNR_total / SSIM_total, 소수 둘째자리 |
| 제공 모델 | DnCNN 17층 · 64채널 · GroupNorm + SiLU · 전역 잔차 (593,024 params) |
| 비교 대상 | mean 3×3 / median 3×3 / adaptive 5×5 (conventional method) |
| 발표 | 파이프라인 설명 · before/after 예시 · 방법 선택 근거 |

## 넘어야 할 기준선

배포된 예시 로그(`log_denoising_example/00012_train`, DnCNN 10 epoch)의 test 성적.

**PSNR (dB)**

| noise | n | noisy | DnCNN | mean | median | adaptive |
|---|---|---|---|---|---|---|
| gaussian | 25 | 27.668 | 32.044 | 26.316 | 27.382 | 30.078 |
| rician | 25 | 24.079 | 28.900 | 23.669 | 24.080 | 27.046 |
| salt_and_pepper | 25 | 17.304 | 29.692 | 23.196 | 28.496 | 21.164 |
| uniform | 25 | 29.631 | 31.406 | 26.263 | 26.779 | 30.521 |
| **ALL** | 100 | 24.671 | **30.510** | 24.861 | 26.684 | 27.202 |

**SSIM ALL**: noisy 0.6659 · DnCNN **0.8950** · mean 0.7615 · median 0.8076 · adaptive 0.7919

여기서 읽히는 것:

- **salt & pepper 가 문제다.** 입력이 17.3 dB 로 혼자 7~12 dB 아래고, conventional
  중에서는 median 만 (28.5 dB) 통한다. mean 은 23.2, adaptive 는 21.2 로 오히려
  gaussian 때보다 못하다. 임펄스는 "작은 흔들림"이 아니라 "몇 픽셀이 통째로 거짓말"이라
  평균 계열이 그 거짓말을 이웃에 퍼뜨린다.
- **conventional 은 전체 평균에서 입력을 거의 못 넘는다.** mean 24.86 vs noisy 24.67.
  종류를 모르는 채 하나의 필터를 고정하면 이런 결과가 된다.
- **DnCNN 은 그 문제를 통째로 우회한다.** 종류별로 다르게 반응하도록 학습되니까.

## 접근

배포 코드와 **같은 데이터 · 같은 노이즈 합성 · 같은 지표**를 쓰고, 학습 쪽만 손봤다.

| 바꾼 것 | 이유 | 결과 |
|---|---|---|
| **Charbonnier** loss (L2 → smooth L1) | L2 는 s&p 의 극단값 몇 픽셀에 gradient 가 끌려간다 | 채택 |
| **cosine LR, 40 epoch** (기본 10 epoch, plateau×0.88) | 10 epoch 에서는 감쇠가 사실상 안 걸린다 | 채택 |
| **patch 128 랜덤 크롭 + rot90** | 크롭 자체가 증강이고 step 수도 2배가 된다 | 채택 |
| 추론 시 **8× self-ensemble** | dihedral 8종으로 추론해 되돌려 평균. 학습 비용 0 | **+0.30 dB** |
| 입력에 **median 3×3 채널 추가** (`DnCNNPlus`) | s&p 에서 median 이 mean 을 5 dB 앞선다. "이미 임펄스가 지워진 버전"을 같이 주면 유리할 것이라 봤다 | **기각 (−0.39 dB)** |

**구조는 배포된 DnCNN 그대로가 최선이었다.** 기준선 대비 개선은 전부 학습 절차에서 나왔다.

### 파이프라인 검증

학습 결과를 믿으려면 우리가 재는 숫자가 채점 숫자와 같아야 한다.
`check_baselines.py` 는 mean/median/adaptive 를 우리 로더·우리 지표로 다시 재서
배포 예시 로그와 대조한다.

```
PSNR 최대 차이: 0.0000 dB — 일치
```

4종 × 4방법 전부 소수 넷째자리까지 일치한다. 여기가 맞은 뒤에 학습을 돌렸다.

### 배포 코드에서 우회한 것

`DataWrapper` 의 `_name = self.file_list[idx].split("/")[-1]` 는 Windows 에서
glob 이 역슬래시를 돌려주므로 **경로 전체가 파일명이 된다.** 그러면 noisy 파일 매칭이
깨지고, 검증 노이즈 seed(`crc32(name)`)도 달라진다. Colab 에서는 드러나지 않는다.
이 저장소는 `Path(...).name` 을 쓴다.

## 결과

### 제출값

```
PSNR_total 34.56    SSIM_total 0.9489
```

배포 기준선(DnCNN 10 epoch) 30.51 / 0.8950 대비 **+4.05 dB / +0.0539**.
A100 에서 60 epoch, 순정 DnCNN + 8× self-ensemble.

### epoch 수가 답이었다

같은 구조·같은 레시피로 학습 길이만 바꿨다.

| | val PSNR | test PSNR | test SSIM |
|---|---|---|---|
| 40 epoch (RTX 2060) | 33.754 | 34.13 | 0.9445 |
| **60 epoch (A100)** | **34.259** | **34.56** | **0.9489** |

처음에는 A100 이니 `--patch 256 --batch 32` 를 권했는데 그게 더 나빴다. 배치를 2배로
키우면 epoch 당 iteration 이 절반이 되어 **총 step 이 오히려 줄고**, patch 256 은
이미지 전체라 랜덤 크롭 증강이 사라진다. DnCNN 은 완전 합성곱이고 수용영역이 35px 라
학습 패치를 추론 크기에 맞출 이유도 없다. **A100 의 이득은 배치가 아니라 step 수에서 나온다.**

| 파이프라인 | clean 사용 | PSNR | SSIM |
|---|---|---|---|
| 복원 안 함 (입력) | — | 24.67 | 0.6659 |
| mean 3×3 | — | 24.86 | 0.7615 |
| adaptive 5×5 | — | 27.20 | 0.7919 |
| median 3×3 | — | 26.68 | 0.8076 |
| **label-free (test 100장만)** | **0장** | **30.08** | **0.8882** |
| 배포 기준선 DnCNN 10 epoch | 7,268장 | 30.51 | 0.8950 |
| DnCNN + median 채널 + SE (40ep) | 7,268장 | 33.75 | 0.9424 |
| DnCNN + 학습 레시피 + SE (40ep) | 7,268장 | 34.13 | 0.9445 |
| **DnCNN 60ep + SE (제출)** | 7,268장 | **34.56** | **0.9489** |

### 노이즈 종류별

| noise | 입력 PSNR | 제출 모델 | Δ | 입력 SSIM | 제출 모델 |
|---|---|---|---|---|---|
| gaussian | 27.67 | 34.95 | +7.28 | 0.6835 | 0.9426 |
| rician | 24.08 | 31.28 | +7.20 | 0.7104 | 0.9096 |
| uniform | 29.63 | 35.22 | +5.59 | 0.7950 | 0.9550 |
| **salt & pepper** | **17.30** | **36.78** | **+19.48** | **0.4746** | **0.9885** |

self-ensemble 기여는 34.02 → 34.56 으로 **+0.54 dB** (로컬 40ep 에서도 +0.55 로 일관).

### median 채널 가설은 기각됐다

s&p 입력이 17.3 dB 로 혼자 10 dB 아래고 고전 필터 중 median 만 통하니(28.5 vs mean 23.2),
median 결과를 두 번째 입력 채널로 주면 도움이 될 것이라는 가설이었다. ablation 결과:

| | 전체 | s&p 구간 |
|---|---|---|
| 순정 DnCNN | **34.13 / 0.9445** | **35.95 / 0.9842** |
| + median 채널 | 33.75 / 0.9424 | 34.74 / 0.9821 |

**넣은 이유였던 s&p 에서 오히려 1.21 dB 를 잃었다.** median 이 임펄스를 지우면서 미세
계조도 뭉개고, 네트워크가 그 뭉개진 채널에 일부 의존하게 된 것으로 보인다. 17층 3×3 이면
임펄스는 스스로 처리할 수 있었다 — 네트워크의 능력을 과소평가하고 불필요한 유도 편향을
넣은 셈이다. ablation 을 안 돌렸으면 0.39 dB 를 손해 본 채 "median 채널 덕분"이라고
발표할 뻔했다.

### 되지 않은 것 — σ 게이트

입력이 이미 50~62 dB 인 이미지에서는 모델이 손해다 (100장 중 8장). 정답을 보고 매번
유리한 쪽을 고르면 **+0.52 dB**. 그런데 σ̂ 로는 그 8장을 못 고른다 — MAD 추정기가
이미지의 미세 질감을 노이즈로 착각해, 실제 σ=0.001 인 이미지를 0.037 로 추정한다.
임계값을 어디에 두든 손해였다 (τ=0.004 에서 −0.017 dB, τ=0.02 에서 −2.89 dB).

blind 상태라는 제약이 실제 비용을 발생시킨다는 구체적 증거다. `analyze_gate.py` 로 재현된다.

## Label-free 파이프라인 (가산점)

clean 이미지를 loss 에 **한 번도 쓰지 않고** 학습한다 (Noise2Void, `train_n2v.py`).

어떤 픽셀의 값을 주변 값으로 덮어써 입력에서 지운 다음 그 자리를 맞히게 하고, 정답으로
덮어쓰기 전의 **noisy 값**을 준다. 노이즈가 픽셀마다 독립이면 주변에서 그 픽셀의 노이즈를
알아낼 방법이 없으니, 네트워크가 맞힐 수 있는 건 구조뿐이다.

우리 노이즈 4종은 전부 픽셀 독립이라 전제를 만족한다. 다만 L2 loss 는 조건부 **평균**을
학습하므로 노이즈 평균이 0 이어야 하는데 **s&p 는 0/max 로 덮으니 아니고, rician 은
절댓값 때문에 위로 들린다.** 그래서 L1 을 쓴다 — 조건부 **중앙값**은 임펄스에 끌려가지 않는다.

**체크포인트 선택도 label-free 로 한다.** val noisy 의 마스킹 loss 로 고른다. clean 기반
PSNR 로 고르면 파이프라인 전체가 label-free 가 아니게 되기 때문이다.

```
label-free (test 100장만, clean 0장)   PSNR 30.08   SSIM 0.8882
배포 기준선 (clean 7,268장 supervised)  PSNR 30.51   SSIM 0.8950
```

정답을 한 장도 안 쓰고 기준선에 **0.43 dB** 까지 붙었다. median 3×3 필터(26.68)는 3.4 dB 앞선다.

`--source train` (train clean 으로 noisy 를 합성해 그 noisy 만 학습) 도 해 봤지만 step 2000 이
최고점이고 그 뒤로 하락했다 (진단 PSNR 25.9). 데이터가 73배 많은데 3 dB 나쁘다. 평가 대상
100장에 적응하는 쪽이 일반 디노이저를 배우는 것보다 유리했다.

## 다음 — DRUNet

DnCNN 의 한계는 파라미터가 아니라 **수용영역**이다. 17층 3×3 은 35픽셀밖에 못 본다.

| 모델 | 파라미터 | 수용영역 | 6GB GPU 속도 |
|---|---|---|---|
| DnCNN | 0.59M | 35px | 212 ms/iter (patch128 b16) |
| DRUNet | 32.6M | ~180px | 729 ms/iter (patch128 b16, peak 1.68GB) |

DRUNet 은 U-Net + res block 으로 3번 다운샘플해 같은 깊이로 훨씬 넓게 본다. 메모리는
오히려 DnCNN 보다 적게 쓰지만 연산량이 3.4배라 로컬에서는 60 epoch 에 5.5시간이다 —
A100 에서 돌리는 게 맞다.

**성능보다 중요한 선택 이유는 3일차다.** DRUNet 은 DPIR(Plug-and-Play Image Restoration
with Deep Denoiser Prior)의 디노이저 프라이어로 설계된 구조다. 3일차의 deconvolution +
denoising 결합에서 데이터 정합 단계와 사전지식 단계를 번갈아 푸는 구조를 쓰려면,
사전지식 자리에 들어갈 디노이저가 바로 이것이다. `sigma_map=True` 로 노이즈 세기를
채널로 받는 모드도 넣어 뒀다 — 그 반복에서 매 단계 "이번엔 이만큼만 지워라"라고
지시하는 데 쓴다. 1일차는 σ 를 모르므로 blind 로 쓴다.

## 폴더 구조

```
실습5/
├── README.md
├── index.html          GitHub Pages 진입점
├── explainer.html      배경 설명 자료 (비전공자용)
├── src/
│   ├── denoise/        ← 이번 과제
│   │   ├── metrics.py       PSNR·SSIM (배포 구현 그대로)
│   │   ├── filters.py       mean / median / adaptive (배포 구현 그대로)
│   │   ├── data.py          노이즈 4종 합성 + 데이터셋
│   │   ├── models.py        DnCNN / DnCNNPlus / DRUNet
│   │   ├── train.py         supervised 학습
│   │   ├── train_n2v.py     label-free 학습 (Noise2Void)
│   │   ├── evaluate.py      test 평가 · 제출값 산출
│   │   ├── check_baselines.py  배포 로그와 지표 대조
│   │   ├── analyze_gate.py  σ 게이트 검증 (기각된 아이디어)
│   │   ├── report_per_image.py  이미지 100장 개별 표 → figures/per_image.csv
│   │   ├── make_figures.py  발표용 그림
│   │   └── make_ppt.py      발표 슬라이드 생성
│   └── deconv/         dipole deconvolution 실험 (과제 전제를 잘못 잡았던 흔적)
├── data/               dataset · code_denoising · log_denoising_example (git 미포함)
├── runs/               학습 로그·체크포인트 (git 미포함)
└── figures/            비교 그림
```

### 실행

```bash
cd src/denoise
python check_baselines.py                              # 지표가 배포 로그와 맞는지 먼저 확인
python train.py --model dncnn --epochs 40               # supervised 학습
python train_n2v.py --source test --model dncnn        # label-free 학습
python evaluate.py <run>/checkpoints/checkpoint_best.ckpt --self-ensemble --figures
python make_figures.py <run>/checkpoints/checkpoint_best.ckpt
python make_ppt.py --name "이름"                        # 발표 슬라이드
```

## 진행 상황

- [x] 데이터셋 · 배포 코드 확보, 과제 전제 정정 (deconvolution → denoising)
- [x] 노이즈 4종 합성 · 데이터로더 · 지표 이식
- [x] 배포 예시 로그와 지표 일치 확인 (0.0000 dB)
- [x] conventional 기준선 재현 (mean / median / adaptive)
- [x] DnCNN / DnCNNPlus 학습 코드
- [x] 40 epoch 학습 · test 평가 · 제출값 확정 (34.13 / 0.9445)
- [x] median 채널 ablation — 기각
- [x] before/after · error map · 노이즈 종류별 비교 그림
- [x] label-free 파이프라인 (30.08 / 0.8882, clean 0장)
- [x] 발표 자료 (`실습5_denoising_발표.pptx`, 13장)
- [x] A100 60 epoch 재학습 — 34.56 / 0.9489 로 제출값 확정
- [ ] DRUNet 학습 (A100) — 수용영역 35px → 180px, 3일차 재사용
- [ ] label-free 를 순정 DnCNN 으로 재실행 — 구조를 맞춰 공정 비교

## 아직 모르는 것

- 채점이 여기 `test_label` 로 이뤄지는지, 아니면 비공개 세트가 따로 있는지.
  전자면 이 저장소 숫자가 곧 제출값이고, 후자면 일반화가 더 중요해진다
- `noise_meta.json` 사용이 허용 범위인지. 지금은 안 쓰는 쪽으로 만들었으니 어느 쪽이든 안전하다
