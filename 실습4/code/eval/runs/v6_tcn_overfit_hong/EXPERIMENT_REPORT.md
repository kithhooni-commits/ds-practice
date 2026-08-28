# 🧪 TCN 적절성 검증 실험 — "현재 User(hong)에 Overfitting" 테스트

**질문**: v5_tcn_optimized의 F1(0.379)이 낮다. TCN이라는 접근 자체가 적절한지, 아니면 학습
데이터가 이 평가 영상/사람과 안 맞아서 저평가된 것인지 먼저 확인하고 싶다.

**방법**: `benchmark.mp4`(사람: hong) 자체에서 학습 데이터를 뽑아(부족하면 augmentation) TCN을
이 사람·이 영상에 overfitting시켜본다. 분포가 완전히 일치하는 상황에서도 안 되면, 문제는 데이터가
아니라 다른 곳에 있다는 뜻이다.

---

## 🚨 0. 먼저 발견한 것 — 평가 파이프라인의 TCN 로더가 계속 죽어 있었다

작업을 시작하자마자 `eval/evaluate_video.py`를 직접 실행해 로그를 확인했더니:

```
⚠️ TCN 로드 실패 (룰베이스로 폴백): name 'sys' is not defined
```

`TCNMotionClassifier._load()`가 `sys.path.insert(...)`를 쓰는데, 이 파일 맨 위에 `import sys`가
**없었다.** 그 결과 `try/except Exception`에 걸려 **항상 조용히 rule-base로 폴백**되고 있었다 —
`run_pipeline.py`는 이 서브프로세스의 stdout을 `capture_output=True`로 삼키고 성공 시에는 출력하지
않으므로, 기존에 기록된 `v4_tcn_hybrid`/`v5_tcn_optimized`의 결과를 볼 때는 이 경고가 전혀 보이지
않았다.

**영향 범위 — 무엇이 틀렸고 무엇은 안 틀렸는지**

| 지표 | 영향 여부 | 이유 |
|---|---|---|
| TP/FP/FN, Precision, Recall, **F1** | **영향 없음** | 펀치가 "언제 나가는가"는 버전과 무관하게 항상 같은 룰베이스 물리 트리거(`PunchEvaluator.try_punch`)가 결정한다. 엔진 선택은 트리거 이후 "종류"만 바꾸므로 리더보드의 F1 숫자 자체는 유효하다. |
| **kind_accuracy, confusion matrix, by_action 분포** | **영향 있음 (오염됨)** | "v4_tcn_hybrid"·"v5_tcn_optimized"라는 이름으로 기록된 종류 판정은 사실 **TCN이 한 번도 호출되지 못한 채 나온 rule-base 결과**였다. `presentation.md`에 정리한 confusion matrix도 이 오염된 값을 근거로 작성되었다 — 재확인/수정이 필요하다. |

**고친 것** (`eval/evaluate_video.py`):
1. `import sys` 추가 (근본 원인).
2. `_load()`가 커스텀 `--tcn-model-dir`(가중치만 있는 디렉터리)을 받을 때도 `tcn_model.py`와
   그 의존 모듈(`real_data.py`)을 찾을 수 있도록, 항상 기본 `motion_learning/` 경로도
   `sys.path`에 추가하도록 수정(원래는 커스텀 디렉터리 하나만 넣어서, 그 안에 모듈 사본이
   없으면 `ModuleNotFoundError`로 또 폴백되는 2차 버그가 있었다).
3. `evaluate_video.py`·`run_pipeline.py`에 `--tcn-model-dir` 옵션을 추가해, 기존 배포 가중치
   (`motion_learning/boxing_tcn.pth`)를 건드리지 않고도 실험용 모델을 평가할 수 있게 함.

이 수정 이후 기존 `v5_tcn_optimized`의 설정(`v5_tcn_optimized.json`, 원래 4인 데이터로 학습된
배포 모델)으로 다시 돌려 **정상적으로 모델이 로드되는 것을 확인**했고, 그 결과를
`eval/runs/v5b_tcn_loadfix_reference/`에 별도로 저장했다(공식 `v5_tcn_optimized` 기록은
프로젝트의 불변성 정책에 따라 덮어쓰지 않음).

---

## 1. 학습 데이터 구성 — `benchmark.mp4` 자체에서 뽑기

- **소스**: `eval/video/benchmark_landmarks.jsonl`(90초 전체 프레임, 캐시됨) +
  `eval/video/benchmark_labels.json`(정답 펀치 29개: L/R × STRAIGHT/HOOK/UPPERCUT).
- **피처**: `heuristic_7j_v1` 17차원. `evaluate_video.py::TCNMotionClassifier.push()`와
  **완전히 동일한 수식**으로 재계산해, 학습·평가 간 피처 정의 불일치가 결과에 섞이지 않게 했다.
- **양성 샘플**: 29개 정답 시각에서 60프레임 causal 윈도(실시간 버퍼와 동일한 left-pad 규칙)를
  추출 → `("L","STRAIGHT")→LEFT_JAB` 등으로 라벨 매핑.
- **음성 샘플**: 90초 프로토콜의 "펀치가 없어야 하는" 구간(준비·숨고르기 4회·마무리 → IDLE,
  **풋워크 → OTHER**)에서 400ms 간격으로 조밀하게 추출. 풋워크 오검출이 전 버전 공통 문제였으므로
  이를 명시적으로 학습시키는 것이 핵심 의도였다.
- **왜 증강이 필요했나**: 정답 29개 중 가장 적은 클래스가 3개(R_HOOK, L/R_UPPERCUT)뿐이라
  증강 없이는 학습 자체가 불가능했다.

## 2. Augmentation 방법 — 적용한 4가지 (전부 `train_report.json`에 파라미터 기록)

| # | 방법 | 구체적 내용 | 목적 |
|---|---|---|---|
| 1 | **좌우 미러링** | 17차원 중 left/right 열을 서로 맞바꾸고 x축 속도(vx) 부호를 반전. 라벨도 `LEFT_JAB↔RIGHT_JAB` 등으로 교체 | 복싱 동작의 좌우 대칭성을 이용해 샘플을 그대로 2배로 늘림(`LEFT_HOOK`이 `RIGHT_HOOK`의 증강 데이터가 되는 식) |
| 2 | **타이밍 앵커 지터** | 정답 시각 `t_ms` 기준 `{-150, 0, +150}ms` 오프셋에서도 윈도를 추출 | 실시간 트리거는 GT 순간에 정확히 발사되지 않는다(실측 평균 지연 169.7ms). 그 변동성을 학습에 반영 |
| 3 | **시간축 워핑** | 60프레임 윈도를 선형보간으로 `0.9배/1.0배/1.1배` 길이로 리샘플 후 causal left-pad로 재정렬 | 같은 펀치를 조금 빠르게/느리게 수행한 변형을 모사 |
| 4 | **가우시안 피처 지터** | 채널별 robust 표준편차(MAD 기반)의 **5%**를 시그마로 하는 가우시안 잡음을 프레임마다 독립적으로 추가 (양성 2회, 음성 1회 복제) | 포즈 추정 자체의 프레임 단위 떨림(landmark jitter)을 모사 |

**적용 순서**: (미러 유/무) × (속도 워핑 3종) = 6가지 "깨끗한" 변형 → 각 변형에 가우시안 잡음을
추가로 덧씌워 복제.

**최종 데이터셋 규모** (`motion_learning/overfit_hong/train_report.json`):

| 항목 | 값 |
|---|---:|
| 증강 전 양성 base 윈도 | 87개 (29 정답 × 3 타이밍 오프셋) |
| 증강 전 음성 base 윈도 | 113개 (400ms 간격 샘플링) |
| **증강 후 최종 샘플** | **2,922개** |
| IDLE / OTHER | 960 / 396 |
| LEFT_JAB / RIGHT_JAB | 432 / 432 |
| LEFT_HOOK / RIGHT_HOOK | 189 / 189 |
| LEFT_UPPERCUT / RIGHT_UPPERCUT | 162 / 162 |

## 3. 학습

`train_tcn_real.py`(기존 4인 배포 모델 학습 스크립트)와 **동일한 하이퍼파라미터**로 학습해
비교 가능성을 유지했다: Causal TCN(채널 32/32/32, dilation 1-2-4), AdamW lr=1e-3,
batch=16, 60 epoch, median/MAD 스케일러. 가중치는 기존 배포 모델을 덮어쓰지 않고
`motion_learning/overfit_hong/`에 별도 저장.

**학습셋 자기 재현 정확도: 99.59%** — 증강된 2,922개 샘플 중 2,910개를 맞혔다.
→ **모델·피처의 capacity 자체는 충분하다는 것이 확인됐다.** "외울 수 있는가?"라는 질문에는
명확히 "그렇다"가 답이다.

## 4. End-to-End 평가 — benchmark.mp4 전체에 다시 꽂아서 채점

진짜 질문은 "학습 데이터를 외웠는가"가 아니라 "실제 평가 파이프라인(룰베이스 트리거가 실시간으로
윈도를 끊어 보내는 상황)에 꽂았을 때 더 잘하는가"이다. `run_pipeline.py`로 동일한 90초 벤치마크를
3가지 조건으로 다시 채점했다 (트리거 로직·config는 전부 동일, **종류 분류기만** 다름):

| 조건 | kind 분류기 | Precision | Recall | F1 | **kind_accuracy** | side_accuracy | 비동작 FP |
|---|---|---:|---:|---:|---:|---:|---:|
| 룰베이스만 (버그로 인해 이전까지 "v5_tcn_optimized"로 기록됐던 실제 값) | 각도·속도 임계값 | 0.3793 | 0.3793 | 0.3793 | **0.3636** | 0.3636 | 7회 |
| **v5b — 기존 배포 TCN** (4인 데이터, 로더 버그 수정 후 정상 로드) | Causal TCN | 0.3793 | 0.3793 | 0.3793 | **0.4545** | 0.3636 | 7회 |
| **v6 — 이 실험의 overfit TCN** (hong 1인, benchmark.mp4 자체로 학습) | Causal TCN | 0.3793 | 0.3793 | 0.3793 | **0.4545** | 0.3636 | 7회 |

(전체 수치: `eval/runs/v5b_tcn_loadfix_reference/metrics.json`, `eval/runs/v6_tcn_overfit_hong/metrics.json`)

**핵심 관찰**

1. **Precision/Recall/F1은 세 조건에서 완전히 동일하다.** 당연한 결과다 — "펀치가 언제 나가는가"는
   세 조건 모두 똑같은 룰베이스 물리 트리거가 결정하고, 분류기는 트리거 *이후* "무슨 종류인가"만
   바꾼다. **즉 이 프로젝트의 헤드라인 지표(F1=0.379)는 분류기(rule vs TCN) 선택과 무관하며,
   개선하려면 트리거 로직(언제 펀치로 인정할지) 자체를 건드려야 한다** — 이전 버전 비교(v1~v5)에서
   "TCN이 F1을 못 넘었다"는 관찰은 애초에 TCN이 F1에 영향을 줄 수 있는 지표가 아니었다는 뜻이다.
2. **TCN은 kind_accuracy에서 룰베이스보다 실제로 낫다** (36.4% → 45.5%, +9.1%p). 버그가 없었다면
   v4/v5도 이 정도 이득을 보여줬어야 했다 — 로더 버그 때문에 그 이득이 전부 가려져 있었다.
3. **"이 사람에 overfitting"은 추가 이득을 주지 않았다.** v5b(4인 혼합 학습)와 v6(hong 1인,
   같은 영상으로 99.6%까지 overfit)의 kind_accuracy가 **정확히 같다**(5/11). 11개의 정답 매칭
   이벤트 중 단 1개(HOOK 오분류 방향)만 다르고, 나머지는 동일한 예측을 냈다.

## 5. 결론 — TCN 방법 자체는 유효하지만, 데이터 재학습으로는 더 못 올린다

- **TCN이 "적절한 접근인가"에 대한 답**: 부분적으로 그렇다. 종류 분류(kind)에서는 룰베이스보다
  분명히 낫다(+9.1%p). 다만 원래 기대했던 "F1을 크게 끌어올릴 레버"는 아니다 — F1은 트리거
  문제이고, TCN은 트리거에 관여하지 않는 구조이기 때문이다(`DEVLOG.md`에 기록된 원래 설계
  원칙과 일치).
- **"현재 User에 overfitting하면 나아지는가"에 대한 답**: **아니다.** 분포를 완벽히 맞추고
  (같은 영상·같은 추출 파이프라인), 29개뿐인 정답을 4종 augmentation으로 2,922개까지 불려
  학습셋 자체는 99.6%까지 외웠음에도, 실제 평가에서의 kind_accuracy는 4인 혼합 학습 모델과
  **한 치도 다르지 않았다.** 즉 지금까지의 저성능은 "학습 데이터가 이 사람/이 영상과 안 맞아서"가
  아니라, 더 구조적인 원인 때문이다. 가장 가능성 높은 후보:
  1. 트리거가 실제로 발사되는 시점(GT 대비 평균 169.7ms, 최대 333ms 지연)이 "깨끗한 펀치 프로토타입"
     구간이 아니라 이미 애매해진/회수 중인 구간을 보고 있을 수 있다 — 분류기가 무엇이든 이 지점에서는
     구분이 어렵다.
  2. 17차원 heuristic 피처 자체가 이 경계 구간에서 STRAIGHT/HOOK/UPPERCUT을 가를 신호를
     충분히 담지 못할 수 있다(`presentation.md` §1-2에서 지적한, 모든 버전·모든 엔진에 공통된
     "STRAIGHT 계열 오분류"가 이를 뒷받침한다).
- **권장 다음 단계**:
  1. (가장 중요, 당장 가능) **v4_tcn_hybrid·v5_tcn_optimized를 이번에 고친 평가기로 재실행**해,
     실제로 TCN이 호출된 수치로 공식 리더보드/`presentation.md`의 confusion matrix를 교체한다 —
     지금 기록은 "rule-base를 TCN이라는 이름으로 잘못 라벨링한 값"이다.
  2. F1(트리거) 자체를 올리려면 `punch_core.js`의 윈도/쿨다운 파라미터를 건드려야 한다 —
     이건 애초에 TCN의 책임 범위가 아니다.
  3. kind_accuracy를 더 올리려면 "더 많은 같은 분포의 데이터"가 아니라 "트리거 시점의 피처
     표현력"을 개선해야 한다 — 예: 펀치 궤적의 각속도/곡률 채널 추가, 또는 트리거 시점이 아니라
     피크 속도 시점(이미 rule-base가 내부적으로 기억하고 있는 `peak` 시점)의 윈도를 분류기에
     넘기는 구조 변경.

---

## 부록 — 재현 방법

```bash
# 1) benchmark.mp4 자체로 학습 데이터 구성 + augmentation + 학습 (overfit 모델)
python motion_learning/train_tcn_benchmark_overfit.py
# → motion_learning/overfit_hong/{boxing_tcn.pth, boxing_tcn_scaler.json, train_report.json}

# 2) 로더가 실제로 이 모델을 쓰는지 진단 (트리거마다 모델의 실제 예측/확신도 확인)
python motion_learning/diagnose_overfit_hong.py

# 3) 전체 90초 벤치마크로 end-to-end 채점
python eval/run_pipeline.py --version v6_tcn_overfit_hong --engine tcn \
  --config eval/configs/v6_tcn_overfit_hong.json \
  --tcn-model-dir motion_learning/overfit_hong --overwrite

# (비교 기준) 기존 배포 모델을 로더 버그 수정 후 다시 채점
python eval/run_pipeline.py --version v5b_tcn_loadfix_reference --engine tcn \
  --config eval/configs/v5_tcn_optimized.json
```

**수정된 파일**: `eval/evaluate_video.py`(`import sys` 추가, `--tcn-model-dir` 옵션,
커스텀 모델 디렉터리에서도 `tcn_model.py`/`real_data.py`를 찾도록 경로 보강),
`eval/run_pipeline.py`(`--tcn-model-dir` passthrough). 기존 배포 가중치(`motion_learning/boxing_tcn.pth`)·
공식 런 기록(`v1~v5`)은 건드리지 않음.
