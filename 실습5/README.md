# 실습5 — 정답 없이 흐린 이미지 되돌리기

삼성 DS2과정 프로젝트 **Image Restoration Challenge**. 번짐(dipole convolution)과 잡음으로
망가진 흑백 이미지를 복원하는 파이프라인을 만든다. 테스트 데이터에는 **정답 이미지가 없다.**

**설명 자료(사전 지식 없이 읽는 배경) →** [explainer.html](explainer.html)
**출처 →** Day 1–3 강의자료 (Jongho Lee, SNU LIST)

## 문제 한 줄

주파수 영역에서 **커널이 정확히 0이 되는 영역**이 있어 단순 역연산이 발산한다.
그 자리를 무엇으로 메울 것인가가 이 과제의 전부다.

## 열화 모델

```
g = h * f + n
```

| 기호 | 의미 | 주어지는가 |
|---|---|---|
| `f` | 깨끗한 원본 | 학습셋만 (7,368장) |
| `h` | 2D dipole 커널 | **모델 제공** |
| `n` | 잡음 | **성질 미제공** — 직접 추정해야 한다 |
| `g` | 손상 이미지 | 테스트 100장 |

dipole 커널은 푸리에 영역에서 magic-angle cone 위의 값이 0이다. `G/H`를 그대로 계산하면
그 cone 방향으로 줄무늬(streaking)가 폭발한다.

## 과제 스펙

| 항목 | 값 |
|---|---|
| 학습/검증 clean 이미지 | 7,368장 |
| 테스트 | corrupted 100장, clean 라벨 없음 |
| 데이터 생성 제한 | clean 1장당 **orientation ≤ 6 × 노이즈 ≤ 2 = 최대 12장** |
| 평가 지표 | PSNR, SSIM |
| 채점 | 암호화된 evaluator 포함 test code, Colab에서 실행 |
| 발표 | 파이프라인 설명 · before/after 예시 · 방법 선택 근거 |
| 가산점 | **label-free 파이프라인** |

## 규칙에 숨은 힌트

생성 제한 `6 × 2`는 임의의 숫자가 아니다.

- **orientation 6개** — 3개 이상의 방향 데이터를 합치면 커널의 0 영역이 서로 다른 자리에
  놓여 k-space의 빈칸이 사라진다 (COSMOS 계열).
- **같은 orientation에 노이즈 2장** — 장면은 같고 잡음만 다른 쌍. Noise2Noise가 요구하는
  데이터 형태와 정확히 일치한다. 정답 없이 잡음을 지울 수 있다.

즉 규칙 자체가 **label-free 경로를 열어 두고 가산점까지 걸어 놓은** 구조다.

## 접근 후보

| 안 | 내용 | 쓸모 |
|---|---|---|
| A | Wiener filter / TKD / L2 정규화 | 기준선. 학습 없이 바닥 성능을 확정한다 |
| B | End-to-end 신경망 (corrupted → clean) | 단순하지만 블랙박스, 분포 밖에서 무너진다 |
| C | 2-stage (denoise → deconvolution) | 중간 산출물을 볼 수 있어 진단·발표에 유리 |
| D | Self-supervised denoise + multi-orientation deconv | 가산점 경로. C의 label-free 버전 |

Day 3 자료가 경고하는 딥러닝의 실패 양상 — 학습 분포 밖 과소추정, 비교 불공정성,
QSM Challenge 2.0에서 고전 기법이 딥러닝을 이긴 사례 — 는 발표에서 짚을 지점이다.

## 폴더 구조

```
실습5/
├── README.md          이 파일
├── index.html         GitHub Pages 진입점
├── explainer.html     배경 설명 자료 (비전공자용)
├── src/               파이프라인 코드
├── data/              로컬 데이터 (git 미포함)
└── figures/           복원 전후 예시, 그래프
```

## 진행 상황

- [x] 강의자료 3일치 정리, 과제 스펙 확정
- [x] 배경 설명 자료 작성
- [ ] 데이터셋 · evaluator 코드 확보 (위치 미확인)
- [ ] forward 시뮬레이터 구현 (dipole convolution + 노이즈)
- [ ] 잡음 특성 추정 — 평탄 영역/히스토그램
- [ ] A안 기준선: Wiener / TKD
- [ ] D안: Noise2Noise 디노이저 + multi-orientation 디컨볼루션
- [ ] PSNR/SSIM 비교표, before/after 그림
- [ ] 발표 자료

## 아직 모르는 것

- 학습 데이터와 test code를 어디서 받는지
- 잡음의 종류와 세기 (설계상 미제공 — 데이터에서 추정해야 한다)
- 제공되는 dipole 모델의 정확한 형태와 orientation 파라미터화 방식
