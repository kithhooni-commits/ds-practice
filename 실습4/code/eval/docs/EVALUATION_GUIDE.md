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
python iter4/eval/synth_dataset.py

# 2. 전체 스위트 실행 및 요약 결과 확인
python iter4/eval/run_suite.py
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
python iter4/eval/evaluate_video.py iter4/eval/video/benchmark.mp4 --labels iter4/eval/video/benchmark_labels.json

# 관절 궤적 및 판정 이벤트가 오버레이된 비디오 생성
python iter4/eval/evaluate_video.py iter4/eval/video/benchmark.mp4 \
  --labels iter4/eval/video/benchmark_labels.json \
  --annotate iter4/eval/output/annotated_benchmark.mp4 \
  --report iter4/eval/output/benchmark_report.json
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

## 🔄 5. 알고리즘/모델 변경 시 새 버전 평가 워크플로우

**철칙**: `iter4/eval/runs/<version>/` 는 **불변 아카이브**다. 이미 존재하는 버전에 덮어쓰지 않고, 항상 **새 버전 태그**로 아카이빙한다. (`run_pipeline.py` 는 기존 버전 덮어쓰기를 기본 차단하며, 재실행이 필요하면 `--overwrite` 를 명시해야 한다.)

### 5.1 언제 새 버전을 만드는가

다음 중 하나라도 해당하면 새 `vN_...` 버전을 만들어 평가한다:

* `punch_core.js` / `evaluate_video.py` 의 룰베이스 임계값 변경 (`PUNCH_SPEED`, `PUNCH_EXTEND`, `HOOK_VX`, `UPPERCUT_VY` 등)
* TCN 모델 재학습, 학습 데이터/피처 변경, `boxing_tcn.pth` / `boxing_tcn_scaler.json` 교체
* `TCN_MIN_CONF` 등 TCN 신뢰도 임계값 변경
* 검출 파이프라인 로직 변경 (트리거·잠금·좌우 판단 등)
* Kinematics 정의나 좌표계 변경

**scoring/phase 로직 자체 수정** (예: 매칭 알고리즘 개선) 은 **모든 기존 버전을 재실행**해 registry 를 갱신한다. 이 경우엔 새 버전을 만드는 게 아니라 `--overwrite` 로 전체를 다시 굽는다.

### 5.2 표준 절차

```bash
conda activate pjt-4

# 1) 새 config 를 iter4/eval/configs/vN_<slug>.json 로 만든다
#    - v1_baseline.json 을 복사해 tune 값 조정하는 게 가장 안전
#    - TCN 을 쓰면 "engine": "tcn" 을 포함, TCN_MIN_CONF 도 넣는다

# 2) 합성 스위트로 회귀 확인 (선택이지만 강력 권장)
python iter4/eval/run_suite.py
#    -> 68/68 F1=1.000 이 유지되지 않으면 룰베이스 변경이 합성 궤적을
#       망가뜨렸다는 뜻. 새 버전 아카이빙 전에 원인 파악.

# 3) 실촬영 벤치마크로 새 버전 아카이빙
python iter4/eval/run_pipeline.py \
  --version vN_<slug> \
  --engine <rule|tcn> \
  --config iter4/eval/configs/vN_<slug>.json

# 결과: iter4/eval/runs/vN_<slug>/{metrics.json, report.json,
#       summary_report.md, punches.csv} + runs_registry.json 갱신
```

### 5.3 결과 검증 체크리스트

버전을 아카이빙한 뒤 다음을 반드시 확인한다. 하나라도 실패하면 registry 오염 가능성이 있으므로 **커밋하지 말고** 원인을 찾는다.

* [ ] `--engine tcn` 인 경우 stdout 에 `🧠 [TCN Engine] PyTorch Causal TCN 모델 로드 완료` 로그가 있다. 없으면 룰베이스로 폴백된 상태로 저장됐다는 뜻이지만, 현재 파이프라인은 로드 실패 시 즉시 `SystemExit` 이므로 이 로그가 없으면 파이프라인이 애초에 중단됐어야 한다.
* [ ] `metrics.json` 에 `matches` 필드가 있고 `len(matches) == tp` 이다.
* [ ] `metrics.json.phase_analysis.skipped` 가 `true` 가 아니다. (알려지지 않은 case labels 를 실수로 쓰면 phase 분석이 스킵된다.)
* [ ] `runs_registry.json` 최신 엔트리의 `version` 이 방금 만든 태그이고 `f1/precision/recall/non_action_fp` 가 stdout 요약과 일치한다.
* [ ] 룰베이스 변경이면 혼동행렬(`confusion`) 이 이전 버전과 다르다. TCN 변경이면 `confusion` 에 `->UPPERCUT`, `->HOOK` 같은 TCN 특유 오분류가 등장한다 (혼동행렬이 rule 결과와 완전히 동일하면 TCN 이 실제로 관여하지 않은 것).

### 5.4 커밋 규칙

* 로직 변경 파일 + `runs/vN_<slug>/*` + `runs_registry.json` + `output/benchmark/*` 를 **한 커밋**으로 묶는다. 산출물이 로직과 분리 커밋되면 나중에 revert 할 때 registry 가 불일치 상태로 남는다.
* 커밋 메시지 프리픽스는 `feat(eval):` (새 알고리즘/모델), `fix(eval):` (평가 로직 버그 수정), `perf(eval):` (튠 최적화).

---

## 🤖 6. 새 버전 평가를 Agent 에게 시킬 때 프롬프트 템플릿

Agent 에게 던질 때는 **무엇을 바꿨는지 · 새 버전 태그 · 성공 판정 기준**을 명시한다. 애매하면 agent 가 기존 버전에 덮어쓰거나 rule 폴백을 못 눈치채고 커밋한다 (v4/v5 사고의 재발). 아래 두 템플릿을 상황에 맞게 골라 쓴다.

### 6.1 튠(임계값) 변경 — 룰베이스

```text
iter4 브랜치, conda env pjt-4.

<무엇을 바꿨는지 한 줄> 을 반영한 새 버전 v<N>_<slug> 를 평가·아카이빙해줘.

절차:
1) iter4/eval/configs/v1_baseline.json 을 복사해서
   iter4/eval/configs/v<N>_<slug>.json 을 만든다.
   변경 파라미터: <PUNCH_SPEED=..., PUNCH_EXTEND=..., 필요한 것만>
2) `python iter4/eval/run_suite.py` 로 합성 스위트가 68/68 F1=1.000
   유지되는지 먼저 확인. 깨지면 여기서 멈추고 원인 보고.
3) `python iter4/eval/run_pipeline.py --version v<N>_<slug>
   --engine rule --config iter4/eval/configs/v<N>_<slug>.json`
   실행. `--overwrite` 는 붙이지 말 것.
4) `iter4/eval/docs/EVALUATION_GUIDE.md` 5.3 체크리스트 전 항목 검증
   결과를 보고. 하나라도 실패면 커밋하지 말고 원인을 알려줘.
5) 모두 통과하면 로직 변경 파일 + runs/v<N>_<slug>/* +
   runs_registry.json + output/benchmark/* 를 한 커밋으로 묶어서
   `feat(eval): ...` 메시지로 커밋해줘.

성공 기준(참고): non_action_fp 가 v1_baseline(9) 대비 감소, F1 이
v1_baseline(0.3666) 이상. 이 기준을 못 맞춰도 결과는 아카이빙하되
커밋 메시지에 "회귀 관찰됨" 을 명시.
```

### 6.2 TCN 모델/신뢰도 변경 — 딥러닝

```text
iter4 브랜치, conda env pjt-4.

<TCN 재학습 / TCN_MIN_CONF 변경 / 스케일러 교체 등> 을 반영한 새
버전 v<N>_<slug> 를 평가·아카이빙해줘.

절차:
1) `iter4/motion_learning/boxing_tcn.pth` 와
   `boxing_tcn_scaler.json` 이 새 파일인지, 파일 mtime 이 이번 학습
   커밋 이후인지 확인.
2) iter4/eval/configs/v5_tcn_optimized.json 을 복사해
   iter4/eval/configs/v<N>_<slug>.json 을 만든다.
   변경 파라미터: <TCN_MIN_CONF=..., 필요하면 룰 트리거도>
   반드시 `"engine": "tcn"` 을 유지.
3) `python iter4/eval/run_pipeline.py --version v<N>_<slug>
   --engine tcn --config iter4/eval/configs/v<N>_<slug>.json`
   실행. stdout 에 `🧠 [TCN Engine] PyTorch Causal TCN 모델 로드
   완료` 로그가 반드시 있어야 한다. 없으면 즉시 중단·보고.
4) `iter4/eval/runs/v<N>_<slug>/metrics.json` 의 `confusion` 을
   `v1_baseline` 의 그것과 비교해 서로 다른지 확인. 완전히 같으면
   TCN 이 실제로 관여하지 않았다는 뜻이므로 커밋하지 말고 원인
   보고.
5) EVALUATION_GUIDE.md 5.3 체크리스트 전 항목 검증. 하나라도 실패면
   커밋 금지·원인 보고.
6) 통과하면 학습 산출물 + configs/v<N>_<slug>.json + runs/... +
   registry 를 한 커밋으로 `feat(eval): ...` 로 커밋.

성공 기준(참고): kind_accuracy 가 v1_baseline(36.4%) 을 초과. F1 은
현재 상한이 낮으니 v5(0.3793) 이상이면 진전. 못 맞춰도 결과는
아카이빙하되 커밋 메시지에 "회귀 관찰됨" 명시.
```

### 6.3 파이프라인/scoring 로직 자체를 바꿀 때 — 전체 재실행

```text
iter4 브랜치, conda env pjt-4.

<scoring 알고리즘 / phase 계산 / evaluate_video.py 파이프라인 등>
로직 자체를 바꿨어. 기존 5개 버전을 모두 --overwrite 로 재실행해
registry 를 갱신해야 한다.

절차:
1) `python iter4/eval/run_suite.py` 로 합성 스위트 회귀 확인.
   68/68 F1=1.000 유지 안 되면 여기서 멈추고 보고.
2) v1_baseline / v2_anti_sway / v3_iter4_eval (--engine rule) 과
   v4_tcn_hybrid / v5_tcn_optimized (--engine tcn) 를 각각
   `python iter4/eval/run_pipeline.py --version <v> --engine <e>
    --config iter4/eval/configs/<v>.json --overwrite`
   로 실행.
3) 각 버전에 대해 EVALUATION_GUIDE.md 5.3 체크리스트를 돌리고,
   특히 이전 F1 값과 비교해 변화량을 표로 정리해 보고.
4) 변화가 의도한 로직 개선과 부합하면 로직 변경 + 전체 runs/* +
   registry + docs 를 한 커밋으로 `fix(eval):` 또는 `feat(eval):`
   로 커밋.

주의: 로직 변경이 registry 숫자를 흔들면 EVALUATION_REPORT.md 의
표도 함께 갱신해야 한다. 커밋 전에 표와 registry 가 일치하는지
반드시 확인.
```

