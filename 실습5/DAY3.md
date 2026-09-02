# 3일차 — 현황과 다음 수

`g = dipole(f) + n`. 흐림과 노이즈가 겹쳤다. 목표는 1등이고, 배포 baseline 25.01 을
넘는 정도로는 부족하다.

## 지금 어디까지 왔나

| 방법 | PSNR | 어디서 잰 값 | 분류 |
|---|---|---|---|
| 전개형 (unrolled, unet f32, blind) | **25.91** | val, ep37 에서 정체 | Supervised |
| 배포 baseline (End2End U-Net 13.4M) | 25.01 | test | Supervised |
| 1일차 디노이저 → Wiener | 21.06 | test | Supervised |
| **오라클 위너 (최고의 선형 필터)** | **19.80** | val | — |
| median 3×3 → Wiener | 19.20 | test | Others |
| 1일차 label-free → Wiener | 18.87 | test | Self-supervised |
| Wiener 단독 | 13.79 | test | Others |
| DC-Net | 14.80 | val, 실패 | — |

전개형은 2527초 돌려 **ep37 에서 정체**했다. 에폭을 늘리는 것으로는 안 오른다.

## 왜 1일차 + 2일차를 이어 붙여도 안 되는가

`src/deconv/` 에서 숫자로 확인했다.

1. **2일차의 109 dB 는 노이즈가 0이라서 나온 값이다.** 답이 `1/D` 로 나누는 것이었는데,
   `n≠0` 이면 노이즈 에너지가 **142,438배 (+51.5 dB)** 로 폭발한다.
2. **흐림이 대비를 죽인다.** `std(f)=0.222 → std(h*f)=0.092` (0.41배). 노이즈 σ 는 1일차와
   똑같은데 신호만 7.6 dB 줄었다. 1일차 SNR +8.6 dB, 3일차 +0.9 dB. 1일차 디노이저가
   본 적 없는 난이도다.
3. **주파수의 16% 는 진짜로 복원 불가다.** `|D|<0.1` 구간의 역산 후 SNR 은 −20 dB 이하.
   2일차엔 이 부분도 되살렸지만 (노이즈가 0이니까) 3일차엔 못 한다.

**결론: 선형 방법은 19.80 dB (오라클 위너) 에서 끝난다.** 그 위는 전부 비선형 사전지식
— 즉 딥러닝 — 의 몫이다. 3일차는 "1일차 도구 + 2일차 도구 조립" 문제가 아니라,
**선형 역산이 포기한 원뿔 안을 이미지 사전지식으로 메우는** 문제다.

재현: `python /tmp/why2.py` 대신 `src/deconv/` 에서 `day3_common.load_val` 로 직접.

## DC-Net 은 왜 3일차에서 죽나 (2일차 42.93 → 3일차 14.80)

hard DC 는 `|D|>τ` 인 주파수를 `G/D` 로 **못박는다**. 그런데 `G = D·F + N` 이므로

```
G/D = F + N/D      <- 노이즈가 1/|D| 배로 증폭된 채 고정된다
```

네트워크는 그 주파수를 고칠 권한이 없다. 2일차엔 `N=0` 이라 완벽했다.

| τ | 못박는 주파수 | 경계 노이즈 증폭 |
|---|---|---|
| 0.005 (2일차 최적 방향) | 99.2% | 200배 |
| 0.05 | 92.0% | 20배 |
| 0.2 | 66.4% | 5배 |

τ 를 키우면 노이즈 증폭은 줄지만 못박는 주파수가 줄어 DC 의 의미가 사라지고, 줄이면
증폭이 폭발한다. **3일차엔 좋은 τ 가 없다.**

고칠 방법은 못박지 말고 무게를 두는 것 — `x = (D·G + λZ)/(D²+λ)`, 즉 soft DC 다.
**전개형이 곧 DC-Net 의 노이즈 대응 일반형이다** (λ→0 이면 DC-Net 이 된다). 그래서
DC-Net 을 따로 살릴 게 아니라 전개형을 키우는 게 맞다.

## 이미 넣어둔 무기 (코드에 있음, 아직 안 돌림)

### 1. σ 조건화 — `--sigma-map` (기대 이득 큼)

3일차 σ 는 이미지마다 **0.0007 ~ 0.13, 200배** 차이가 난다. blind 모델 하나로 그 범위를
덮으려면 평균에 타협해야 한다.

`unrolled.estimate_sigma()` 가 **측정치만 보고** σ 를 읽는다 — `|D|<0.02` 인 주파수엔
신호가 실려올 수 없으니 거기 남은 건 전부 노이즈다 (파세발). val 40장 상대오차
**중앙값 1.9%**. 정답도 `noise_meta.json` 도 안 쓴다.

DRUNet 의 `sigma_map` 채널로 매 단계 넣는다. `--refine drunet` 필요.

### 2. 사전지식 자리를 1일차 디노이저로 초기화 — `--init-refine` (기대 이득 큼)

전개형의 사전지식 자리가 할 일이 정확히 "노이즈 지우기"다. 1일차 DRUNet(37.42 dB)을
출발점으로 주면 무작위 초기화보다 훨씬 나은 곳에서 시작한다. ep37 정체를 뚫을 후보 1순위.

### 3. 4× self-ensemble — `--self-ensemble` (공짜, 학습 불필요)

dipole 이 견디는 대칭을 **직접 확인**했다:

| 변환 | 오차 | |
|---|---|---|
| 좌우 뒤집기 | 7e-16 | 쓸 수 있다 |
| 상하 뒤집기 | 7e-16 | 쓸 수 있다 |
| 180도 회전 | 9e-16 | 쓸 수 있다 |
| 전치 / 90도 회전 | **3.17** | **못 쓴다** — B0 방향이 돌아가 연산자가 바뀐다 |

1일차의 8× 를 그대로 가져오면 안 된다. 4개만 쓴다. 1일차에서 8× 가 +0.9 dB 였으니
4× 는 +0.3~0.5 dB 기대.

### 4. 용량 — `--refine drunet --features 64`

지금은 unet f32. 1일차에서 DRUNet 이 DnCNN 을 2.9 dB 이겼다. 전개형 안에서도 같을 것.

## 다음에 돌릴 것 (우선순위 순)

전부 Colab A100. `--patch` 는 절대 주지 말 것 (dipole 은 전역 연산 — 크롭하면 죽는다).

```bash
# ① 주력. σ 조건화 + DRUNet + 1일차 가중치 warm start
python train_deconv.py --model unrolled --refine drunet --features 64 \
  --sigma-map --init-refine /content/ckpt/day1_drunet.ckpt \
  --unroll-iters 4 --noise-model challenge --input measure \
  --epochs 120 --batch 4 --lr 2e-4 --clip-grad 1.0 --tag u_drunet_sig

# ② 대조군. σ 조건화만 빼서 얼마를 벌었는지 본다 (발표 ablation 에 필요)
python train_deconv.py --model unrolled --refine drunet --features 64 \
  --unroll-iters 4 --noise-model challenge --input measure \
  --epochs 120 --batch 4 --lr 2e-4 --clip-grad 1.0 --tag u_drunet_blind

# ③ 단계마다 다른 가중치. 용량 4배, 단계별 역할 분화
python train_deconv.py --model unrolled --refine drunet --features 48 \
  --sigma-map --no-share-weights --unroll-iters 4 \
  --noise-model challenge --input measure \
  --epochs 120 --batch 4 --lr 2e-4 --clip-grad 1.0 --tag u_noshare

# ④ 방법 B 대조군. 측정치 영역에서 노이즈만 지우고 Wiener 가 역산
python train_deconv.py --model drunet --features 64 --target measure \
  --noise-model challenge --input measure \
  --epochs 120 --batch 4 --lr 2e-4 --clip-grad 1.0 --tag methodB
```

평가 (K 는 반드시 val 에서 — 아래 규칙 참조):

```bash
python eval_day3.py --ckpt <ckpt> --self-ensemble            # ①②③
python eval_day3.py --ckpt <ckpt> --sweep-K --self-ensemble  # ④ (--target measure)
```

## 아직 안 해본 것 (여력 되면)

- **반복 횟수를 추론 때 늘리기.** `share_weights=True` 면 학습 4회 → 추론 6~8회 가능.
  `UnrolledNet.n_iter` 를 바꿔서 val 로 고르면 된다. 공짜.
- **모델 융합.** ①과 ④의 출력을 평균. 보통 +0.3~0.5 dB.
- **label-free 전개형.** 사전지식 자리에 1일차 N2V 디노이저를 넣고 얼려두면 정답 없이
  3일차가 풀린다 → **보너스 점수**. `pnp.py` 로 이미 되지만 (18.87) 전개형 λ 만
  학습하면 더 오를 것. 정답을 안 쓰므로 규칙 위반 아님.
- **σ 추정을 label-free 체크포인트 선택에 사용.** 1일차에서 쓴 방식.

## 절대 지킬 것

**test set 은 채점에만 쓴다.** 학습이 아니어도 하이퍼파라미터(K, λ)를 test 로 고르면
test 를 쓴 것이다. 배포 안내도 "K 는 validation set 에서 sweep", "`noise_meta.json` 은
결과 분석에만" 이라고 명시했다.

`day3_common.py` 가 규약이다.

```python
load_val()   # val clean 에 forward + 파일명 seed 노이즈. 튜닝은 여기서만
load_test()  # 배포된 test_deconv_noise. 채점에만
tune(...)    # val 에서 설정을 고른다
report(...)  # test 로 최종 점수. 노이즈 종류별로 쪼개 보여준다
```

`combine_day3.py` 는 test 에서 스윕하던 스크립트라 **폐기**했다 (실행하면 에러).

## 파일 지도

| 파일 | 역할 |
|---|---|
| `src/deconv/day3_common.py` | val/test 분리 규약. 모든 튜닝의 관문 |
| `src/deconv/run_day3.py` | 학습 없는 조합 전부 비교 (val 튜닝 → test 채점) |
| `src/deconv/train_deconv.py` | 학습. `--sigma-map`, `--init-refine` 추가됨 |
| `src/deconv/unrolled.py` | 전개형 + `estimate_sigma` + `self_ensemble` |
| `src/deconv/eval_day3.py` | test 채점. `--self-ensemble`, `--sweep-K`(val) |
| `src/deconv/figures_day3.py` | 발표 그림 4종 |
| `src/deconv/dcnet.py` | 2일차용. 3일차엔 구조적으로 안 맞음 (위 참조) |
| `figures/day3_*.png` | 열화 사슬 / 방법 격자 / diff+zoom / 취약점 |

## 발표에 쓸 분석 (배포 tips 요구사항)

배포 노트북이 요구한 것: synthetic 학습 쌍과 test 결과를 **visualize**, **difference map**,
**zoom-in**, 어떤 노이즈·이미지에 취약한지, 1·2일차 conventional 과 비교, few-shot 또는
self-supervised.

`figures_day3.py` 가 만든 것으로 다 덮인다. 거기서 나온 분석:

- **Rician 이 가장 약하다** (18.92 dB). 정류 편향이 DC 를 밀어 올리는데 dipole 은 DC 를
  1/3 로 보존하므로 편향이 그대로 살아남는다.
- **salt & pepper 에서만 median 이 딥러닝을 이긴다** (23.23 vs 19.55). 1일차 디노이저는
  *선명한* 입력 위의 임펄스만 봤지 *흐릿한* 입력 위의 임펄스는 본 적이 없다.
- σ<0.05 에서 23.91 dB, σ≥0.10 에서 17.64 dB — **6 dB 차이**. σ 조건화가 필요한 근거.
- diff 맵의 **X자 무늬** = 매직앵글 영널 원뿔. 어떤 K 를 써도 남는다.
- **순서가 중요하다.** 전처리→Wiener 21.06 vs Wiener→전처리 16.88. 노이즈가 흐림 뒤에
  붙었으니 측정치 위에서 백색이고, 역산 **전에** 지워야 한다.

## 실패 기록 (반복하지 말 것)

| 시도 | 결과 | 원인 |
|---|---|---|
| 2일차 답(K→0)을 3일차에 | −24 dB | 노이즈를 +51.5 dB 증폭 |
| DC-Net (hard DC) | 14.80 | `G/D = F + N/D` 를 못박음. 좋은 τ 가 없다 |
| `--patch 128` | 18.6 | dipole 은 전역 연산. 조각의 FFT 는 커널이 다르다 |
| Wiener → 디노이저 | 16.88 | 순서가 반대. 역산이 노이즈를 유색으로 만든다 |
| 8× self-ensemble | (금지) | 90도 회전이 B0 를 돌려 연산자를 바꾼다. 4× 만 |
| 에폭만 늘리기 | ep37 정체 | 용량과 조건화가 병목이지 학습량이 아니다 |
