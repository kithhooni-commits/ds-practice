# 실습5 — 반도체 이미지 노이즈 제거

삼성 DS2과정 **Image Restoration Challenge**. 4종의 노이즈로 망가진 흑백 반도체
이미지를 복원한다. 제출물은 test 100장에 대한 **PSNR_total / SSIM_total 두 숫자**다.

**설명 자료(사전 지식 없이 읽는 배경) →** [explainer.html](explainer.html)
**출처 →** Day 1–3 강의자료 (Jongho Lee, SNU LIST) + 배포 코드 `code_denoising/`

> **전제 수정 기록.** 강의자료만 보고 이 과제를 dipole deconvolution(QSM 계열, 커널의
> 0 영역을 메우는 문제)으로 잡았었다. 실제 배포된 데이터와 코드를 받아 보니 과제는
> **denoising** 이었다. 커널도, orientation 제한도, label-free 가산점도 없다.
> 그때 만든 dipole 코드와 실험은 지우지 않고 [`src/deconv/`](src/deconv/) 에 남겨 뒀다
> — 그쪽은 그쪽대로 굴러가고, 왜 빗나갔는지가 발표에서 할 이야기 중 하나다.

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

| 바꾼 것 | 이유 |
|---|---|
| 입력에 **median 3×3 채널 추가** (`DnCNNPlus`) | s&p 에서 median 이 mean 을 5 dB 앞선다. 네트워크가 임펄스를 스스로 배우게 두는 대신 "이미 임펄스가 지워진 버전"을 같이 준다. 파라미터는 576개(첫 층 채널 하나)만 는다 |
| **Charbonnier** loss (L2 → smooth L1) | L2 는 s&p 의 극단값 몇 픽셀에 gradient 가 끌려간다 |
| **cosine LR, 40 epoch** (기본 10 epoch, plateau×0.88) | 10 epoch 에서는 감쇠가 사실상 안 걸린다 |
| **patch 128 랜덤 크롭** | 6GB GPU 에 256² batch 16 이 안 올라간다. DnCNN 은 완전 합성곱이라 패치로 배우고 256² 로 추론해도 된다 |
| **rot90 증강** 추가 | 기본은 flip 만. 방향 다양성이 부족하다 |
| 추론 시 **8× self-ensemble** | dihedral 8종으로 추론해 되돌려 평균. 학습 비용 0 |

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

> 학습 진행 중 — 40 epoch × 2 (dncnn_plus 제출 후보 / dncnn 동일 레시피 ablation).
> 끝나면 이 절에 test 표와 제출값을 채운다.

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
│   │   ├── models.py        DnCNN / DnCNNPlus
│   │   ├── train.py         학습
│   │   ├── evaluate.py      test 평가 · 제출값 산출
│   │   └── check_baselines.py  배포 로그와 지표 대조
│   └── deconv/         dipole deconvolution 실험 (과제 전제를 잘못 잡았던 흔적)
├── data/               dataset · code_denoising · log_denoising_example (git 미포함)
├── runs/               학습 로그·체크포인트 (git 미포함)
└── figures/            비교 그림
```

### 실행

```bash
cd src/denoise
python check_baselines.py                              # 지표가 배포 로그와 맞는지 먼저 확인
python train.py --model dncnn_plus --epochs 40         # 학습
python evaluate.py ../../runs/<run>/checkpoints/checkpoint_best.ckpt --self-ensemble --figures
```

## 진행 상황

- [x] 데이터셋 · 배포 코드 확보, 과제 전제 정정 (deconvolution → denoising)
- [x] 노이즈 4종 합성 · 데이터로더 · 지표 이식
- [x] 배포 예시 로그와 지표 일치 확인 (0.0000 dB)
- [x] conventional 기준선 재현 (mean / median / adaptive)
- [x] DnCNN / DnCNNPlus 학습 코드
- [ ] 40 epoch 학습 · test 평가 · 제출값 확정
- [ ] before/after 그림, 노이즈 종류별 비교
- [ ] 발표 자료

## 아직 모르는 것

- 채점이 여기 `test_label` 로 이뤄지는지, 아니면 비공개 세트가 따로 있는지.
  전자면 이 저장소 숫자가 곧 제출값이고, 후자면 일반화가 더 중요해진다
- `noise_meta.json` 사용이 허용 범위인지. 지금은 안 쓰는 쪽으로 만들었으니 어느 쪽이든 안전하다
