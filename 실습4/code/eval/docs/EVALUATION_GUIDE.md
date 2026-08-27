# 📊 Boxing Motion Recognition Evaluation Guide

본 문서는 실시간 AR 섀도우 복싱 시스템의 **동작 인식 정확도(Accuracy), 반응 속도(Timing Latency), 안정성(Robustness)**을 체계적으로 검증하기 위한 **투트랙(Two-Track) 평가 파이프라인 가이드**입니다.

---

## 🏗️ 1. 평가 파이프라인 아키텍처 (Two-Track Framework)

```text
[Track 1: 프로그램 합성 궤적 (Synthetic)]           [Track 2: 정해진 프로토콜 실촬영 영상 (Real Video)]
   synth_dataset.py                                    benchmark.mp4 + benchmark_labels.json
         │                                                      │
         ▼                                                      ▼
 datasets/ (11개 케이스 JSONL + labels)                evaluate_video.py (MediaPipe 추적)
         │                                                      │
         └───────────────────────┬──────────────────────────────┘
                                 ▼
                     run_suite.py / scoring.py
                                 │
                                 ▼
               output/suite_report.json / annotated.mp4
             (Precision, Recall, F1, Confusion Matrix)
```

---

## 📁 2. 디렉토리 구조 및 역할

```text
iter3/eval/
├── datasets/                  # Track 1 합성 궤적 및 라벨 데이터셋 (케이스별 저장)
├── docs/                      # 평가 및 녹화 가이드 문서
│   ├── EVALUATION_GUIDE.md    # 전체 평가 파이프라인 가이드 (본 문서)
│   └── RECORDING_GUIDE.md     # 60초 표준 벤치마크 영상 촬영 및 라벨링 가이드
├── models/                    # MediaPipe 포즈 랜드마커 모델 (.task)
├── output/                    # 평가 리포트(JSON), 시각화 비디오(MP4), 펀치 통계(CSV)
├── synth_dataset.py           # [Track 1] 2-Link 기구학 기반 합성 궤적 생성기
├── run_suite.py               # [Track 1] 전체 데이터셋 일괄 평가 및 요약 테이블 출력
├── evaluate_video.py          # [Track 2] 비디오 파일 프레임별 추적 및 펀치 검출기
├── scoring.py                 # Ground Truth 매칭 및 정량 메트릭 계산 엔진
└── extract_landmarks.py       # 비디오에서 순수 관절 랜드마크만 JSONL로 추출하는 유틸리티
```

---

## 🚀 3. 실행 방법 (Usage)

> **전제 조건**: Conda `pjt-4` 환경 활성화
> ```bash
> conda activate pjt-4
> ```

### 🎯 Track 1: 합성 노이즈 궤적 벤치마크 실행

합성 데이터셋을 생성하고 전체 11개 시나리오(총 68회 펀치 + 3개 음성 케이스)에 대해 알고리즘을 즉시 일괄 검증합니다.

```bash
# 1. 합성 궤적 데이터셋 생성 (필요 시)
python iter3/eval/synth_dataset.py

# 2. 전체 스위트 실행 및 요약 결과 확인
python iter3/eval/run_suite.py
```

* **출력 예시**:
  ```text
  case                   정답   예측  TP  FP  FN      P      R     F1     종류   Δt(ms)
  -------------------------------------------------------------------------------
  clean_hook_L            8    8   8   0   0  1.000  1.000  1.000   1.00       50
  clean_hook_R            8    8   8   0   0  1.000  1.000  1.000   1.00       50
  clean_straight_L        8    8   8   0   0  1.000  1.000  1.000   1.00       43
  clean_straight_R        8    8   8   0   0  1.000  1.000  1.000   1.00       39
  clean_uppercut_L        8    8   8   0   0  1.000  1.000  1.000   1.00       16
  clean_uppercut_R        8    8   8   0   0  1.000  1.000  1.000   1.00       16
  combo_mixed            12   12  12   0   0  1.000  1.000  1.000   0.92       32
  rapid_jab_combo         8    8   8   0   0  1.000  1.000  1.000   0.88       39
  short_reach             0    0   0   0   0      -      -      -      -        -
  sweep_rotation          0    0   0   0   0      -      -      -      -        -
  too_slow                0    0   0   0   0      -      -      -      -        -
  -------------------------------------------------------------------------------
  합계                     68   68  68   0   0  1.000  1.000  1.000
  ```

---

### 📹 Track 2: 실촬영 벤치마크 영상 평가 실행

`RECORDING_GUIDE.md`에 따라 녹화된 영상을 입력하여 오프라인 평가 및 분석 비디오를 생성합니다.

```bash
# 기본 분석 및 점수 측정
python iter3/eval/evaluate_video.py iter3/eval/video/benchmark.mp4 --labels iter3/eval/video/benchmark_labels.json

# 관절 궤적 및 판정 이벤트가 오버레이된 비디오 생성
python iter3/eval/evaluate_video.py iter3/eval/video/benchmark.mp4 \
  --labels iter3/eval/video/benchmark_labels.json \
  --annotate iter3/eval/output/annotated_benchmark.mp4 \
  --report iter3/eval/output/benchmark_report.json
```

---

## 📈 4. 평가 지표 (Metric Definitions)

| 지표명 | 수식 / 계산 방식 | 이상적 목표 | 해석 및 비즈니스 의미 |
| :--- | :--- | :---: | :--- |
| **Precision (정밀도)** | $\frac{TP}{TP + FP}$ | **$\ge 0.95$** | 펀치라고 판정한 것 중 실제 펀치인 비율 (헛스윙/스텝 오검출 방지) |
| **Recall (재현율)** | $\frac{TP}{TP + FN}$ | **$\ge 0.95$** | 실제 수행한 펀치를 놓치지 않고 인식한 비율 |
| **F1-Score** | $2 \times \frac{P \times R}{P + R}$ | **$\ge 0.95$** | 정밀도와 재현율의 조화 평균 (종합 인식 성능) |
| **Kind Accuracy** | $\frac{\text{종류 일치 } TP}{TP}$ | **$\ge 0.90$** | 검출된 펀치 중 잽/훅/어퍼컷 종류를 정확히 분류한 비율 |
| **Side Accuracy** | $\frac{\text{팔 일치 } TP}{TP}$ | **$\ge 0.98$** | 왼손 / 오른손 구별 정확도 |
| **Timing Error ($\Delta t$)**| $|t_{\text{pred}} - t_{\text{label}}|$ 평균 | **$< 50\text{ ms}$** | 펀치 최고 정점 시점과 알고리즘 트리거 시점 간의 지연 편차 |
| **Negative Case FP**| Negative 케이스 검출수 | **$0\text{ 건}$** | 느린 손동작, 하프 리치, 횡이동 시 펀치 오작동이 없어야 함 |

---

## 🔄 5. 알고리즘 수정 및 회귀 방지 (Regression Testing) 워크플로우

클라이언트(`fighter_client.html`)의 펀치 임계값(속도, 각도, 쿨다운 등)이나 판정 로직을 변경할 때는 다음 순서로 검증합니다:

1. **`evaluate_video.py`에 변경된 파라미터 반영** (`PUNCH_SPEED`, `PUNCH_EXTEND` 등)
2. **`python iter3/eval/run_suite.py` 실행**
   - Track 1의 11개 스위트가 `F1 = 1.000`, `FP = 0`, `FN = 0`으로 유지되는지 확인.
3. **Track 2 고정 비디오 재평가**
   - 실제 촬영 영상에 대한 F1 점수 및 지연시간 변화 확인.
4. 모든 지표가 기준치를 만족하면 `fighter_client.html` 및 `DEVLOG.md`에 반영.
