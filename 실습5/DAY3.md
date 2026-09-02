# 3일차 — 현황과 이어서 할 일

`g = dipole(f) + n`. 흐림과 노이즈가 겹쳤다.

## 통과 기준과 현재 위치

| | PSNR | SSIM |
|---|---|---|
| 통과 기준 (조교) | 26 | **0.83** |
| 입력 (blur + noise) | 8.02 | −0.0187 |
| 배포 baseline (End2End U-Net) | 25.01 | 0.8149 |
| 다른 조 (2-step + 4× SE) | 26.73 | 0.8215 |
| **우리 제출값 (test, 4× SE)** | **29.25** | **0.8777** |

**둘 다 통과했다.** 배포 baseline 대비 +4.24 dB · +0.063.

체크포인트: `Drive/MyDrive/ds_day3/0902-0418_deconv-measure_u_drunet_sig.ckpt`
(val 28.44 / 0.8103, ep56)

### 노이즈 종류별 (test 100장, 25장씩)

| | PSNR | SSIM |
|---|---|---|
| salt & pepper | 35.26 | 0.9784 |
| gaussian | 30.03 | 0.8830 |
| uniform | 29.23 | 0.9029 |
| **rician** | **22.47** | **0.7464** |

rician 만 7 dB 뒤진다. **여기가 유일하게 남은 큰 구멍이다.**

## 무엇을 쓰고 있나

전개형(unrolled) — 데이터 정합과 사전지식을 번갈아 4번.

```
x₀ = Wiener(g, λ₀)                        학습 없음
반복 4회:
    σ  = estimate_sigma(g)                측정치의 널 원뿔에서. 정답 안 씀
    z  = DRUNet(x, σ)                     사전지식. 학습된다
    x  = (D·G + λZ)/(D² + λ)              물리 제약. 닫힌 해, 학습 파라미터는 λ 뿐
```

`--model unrolled --refine drunet --features 48 --unroll-iters 4 --sigma-map --share-weights`
`--noise-model challenge --input measure --loss charbonnier`

평가는 항상 `--self-ensemble` (4×: 좌우·상하·180°).

## 지금 돌고 있는 것 (2026-09-02 밤)

둘 다 `--mirror /content/drive/MyDrive/ds_day3` 로 best 를 Drive 에 실시간 복사한다.

| 태그 | 무엇 | 노리는 것 |
|---|---|---|
| `stats3` | `--noise-stats` + v1 이어받기, 40ep lr 1e-4 | rician 22.47 → 25+, 전체 +0.6 dB |
| `v1_more` | v1 이어받기, 60ep lr 1e-4 | v1 이 수렴 전이었다. +0.3~0.5 dB |

## 아침에 할 일

```python
# 1. Drive 의 체크포인트 점수 확인
import glob, torch
for c in sorted(glob.glob("/content/drive/MyDrive/ds_day3/*.ckpt")):
    ck = torch.load(c, map_location="cpu", weights_only=False)
    print(f"{c.split('/')[-1]:<56}{ck.get('val_psnr',0):>8.2f}{ck.get('val_ssim',0):>9.4f}")
```

```bash
# 2. 최고 모델을 test 로 채점 (이 숫자가 제출값이다)
python eval_day3.py --data "$DATA" --ckpt "<최고>" --self-ensemble --sigma-ablation

# 3. 융합 — 구조가 다르면 틀리는 방식도 달라 평균이 둘 다보다 좋다. 무게는 val 에서
python fuse_day3.py --data "$DATA" --ckpts "<A>" "<B>" --self-ensemble

# 4. 그림 (지금 발표의 8·9·10번 그림은 학습 모델이 안 들어간 옛날 것이다)
python figures_day3.py --data "$DATA" --ckpt "<최고>" --self-ensemble --out "$FIG"

# 5. 발표 다시 뽑기
python make_ppt3.py --psnr <값> --ssim <값> --name "이름"
```

`make_ppt3.py` 의 `R = {...}` 에 노이즈별 수치와 σ ablation 값이 하드코딩돼 있다.
새 숫자가 나오면 거기도 같이 고칠 것.

## 왜 이 구조인가 — 발표의 줄기

이 발표의 중심은 "점수가 높다"가 아니라 **"각자 최고인 도구를 합치면 왜 지는가"** 다.

```
1일차 정답 (디노이저)   37.42 dB
2일차 정답 (Wiener)    109.86 dB
둘을 이어 붙이면         21.06 dB   <- 배포 baseline 25.01 에도 진다
```

이유 셋. 전부 `day3_common.load_val` 로 직접 잰 값이다.

1. **2일차 답은 노이즈가 0이라 통했다.** `1/D` 로 역산하면 노이즈 에너지가
   **142,438배 (+51.5 dB)** 로 폭발한다.
2. **흐림이 대비를 죽인다.** `std(f)=0.222 → std(h*f)=0.092` (0.41배). 노이즈는
   1일차와 같은데 신호만 줄어 SNR 이 +8.6 dB → +0.9 dB 가 된다.
3. **주파수의 16% 는 복원 불가.** `|D|<0.1` 은 역산 후 SNR 이 −20 dB 이하다.

**오라클 위너(정답을 알고 만든 최고의 선형 필터)조차 19.80 dB 에서 끝난다.**
그 위는 전부 비선형 사전지식의 몫이고, 그래서 3일차는 도구를 조립하는 문제가 아니라
**역산이 포기한 원뿔 안을 이미지 사전지식으로 메우는 문제**다.

## σ 조건화 — 성능의 절반

| σ 를 어떻게 주는가 | PSNR | SSIM |
|---|---|---|
| 추정 σ (정상) | 29.25 | 0.8777 |
| **σ = 0 (없다고 알려줌)** | **15.64** | **0.3948** |
| σ 2배 (과대평가) | 22.45 | 0.7845 |

`|D| < 0.02` 인 주파수엔 dipole 이 신호를 보내지 않으므로 거기 남은 것은 전부
노이즈다. 파세발로 `E|G|² = N·σ²`. **정답도 `noise_meta.json` 도 쓰지 않는다.**
3일차 σ 는 장마다 0.0007~0.13 으로 **200배** 차이난다.

## rician 이 약한 이유 (남은 숙제)

val 40장에서 잰 값:

| | 평균 편향 | 널원뿔 첨도 |
|---|---|---|
| gaussian | +0.00014 | 2.91 |
| **rician** | **+0.0396** | **10.39** |
| uniform | −0.00010 | 3.84 |
| salt & pepper | −0.00388 | 3.38 |

rician 은 정류라 밝기를 위로 민다. dipole 은 DC 를 1/3 로 보존하므로 역산에서 3배가
되어 복원 이미지에 **0.119** 의 오차로 남는다 — 이미지 std 가 0.222 인데 그렇다.
상수 오프셋 0.119 만으로 PSNR 이 18.5 dB 수준이니 22.47 이 설명된다.

**그런데 첨도가 rician 을 가른다** (10.39 vs 2.9~3.8). `estimate_noise_stats` 가
`(σ, 왜도, 첨도)` 를 돌려주고 `--noise-stats` 로 조건에 넣는다. 그것이 `stats3` 다.

## 절대 지킬 것

**test 는 채점에만 쓴다.** 학습이 아니어도 하이퍼파라미터(K, λ)를 test 로 고르면
test 를 쓴 것이다. `day3_common.py` 가 규약이다 — `load_val` 로 튜닝, `load_test` 로
채점, `noise_meta.json` 은 표를 종류별로 쪼개는 데만.

`check_rules.py` 가 조교 지침 4개를 코드로 검증한다. 제출 전에 한 번 돌릴 것.

배포 metric 이 아닌 구현이 잡히면 `day3_common` 이 즉시 멈춘다 (`src/deconv` 에
초기 단계의 skimage 기반 metrics 가 있었고 실제로 한 번 밟았다).

## 실패 기록 — 반복하지 말 것

| 시도 | 결과 | 원인 |
|---|---|---|
| 2일차 답(K→0)을 3일차에 | −24 dB | 노이즈를 +51.5 dB 증폭 |
| DC-Net (2일차 최고 구조) | 14.80 | hard DC 가 `G/D = F + N/D` 를 못박는다. 좋은 τ 가 없다 |
| `--patch 128` | 18.6 | dipole 은 전역 연산. 조각의 FFT 는 커널이 다르다 |
| Wiener → 디노이저 | 16.88 | 순서가 반대. 역산이 노이즈를 유색으로 만든다 |
| 2단 분해 (twostage) | 15.6 | 역필터가 오차를 1/D 로 증폭해 40 dB 디노이징을 요구한다 |
| SSIM 을 처음부터 손실에 | 17.5 | 덜 학습된 모델을 "맞든 아니든 대비를 키우는" 쪽으로 민다 |
| 언샤프 후처리로 SSIM 보정 | 무효 | 잔차까지 키워 σx 만 커진다. val 이 amount=0 을 고른다 |
| end-to-end DRUNet | 24.6 | 전개형보다 3.8 dB 아래. 물리 구조가 실제로 이득이다 |
| 8× self-ensemble | 금지 | 90도 회전이 B0 를 돌려 연산자를 바꾼다. **4× 만** |

## 파일 지도

| 파일 | 역할 |
|---|---|
| `src/deconv/day3_common.py` | val/test 분리 규약. 모든 튜닝의 관문 |
| `src/deconv/unrolled.py` | 전개형 · `estimate_sigma` · `estimate_noise_stats` · `self_ensemble` |
| `src/deconv/train_deconv.py` | 학습. `--sigma-map` `--noise-stats` `--init-model` `--mirror` |
| `src/deconv/eval_day3.py` | test 채점. `--self-ensemble` `--sigma-ablation` |
| `src/deconv/fuse_day3.py` | 모델 융합. 무게는 val 에서 |
| `src/deconv/figures_day3.py` | 발표 그림 4종 |
| `src/deconv/make_ppt3.py` | 발표 15장 |
| `src/deconv/check_rules.py` | 조교 지침 4개 검증 |
| `src/deconv/twostage.py` | 실패 기록 (전달 곡선 분석이 안에 있다) |
| `src/deconv/sharpen.py` | 실패 기록 (왜 후처리로 SSIM 을 못 사는지) |
| `colab_day3.ipynb` / `colab_day3_v2.ipynb` | 콜랩. v2 는 자족적 |

## 발표 (15장)

`실습5_day3_발표.pptx`. 요구사항 1(파이프라인) · 2(before/after/difference/GT) ·
3(왜 그 방법인가) · 4(label-free, 보너스) + 시도별 요약 한 페이지 + test 규칙 슬라이드.

**8·9·10번 그림은 아직 학습 모델이 안 들어간 옛날 것이다.** 위 4번을 돌려 갱신할 것.
