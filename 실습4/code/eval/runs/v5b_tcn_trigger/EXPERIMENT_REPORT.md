# 🧪 v5b_tcn_trigger — "TCN이 트리거까지 담당하면 F1이 오르는가"

**요청**: v5에서 "펀치 종류 분류"뿐 아니라 "언제 펀치가 나가는지(트리거)"까지 TCN이 직접
결정하도록 바꿔서 평가. (1) 증강 데이터로 학습했는지 확인, (2) 진짜 TCN이 쓰였는지(룰베이스
폴백 아닌지) 검증.

**결론 먼저**: F1은 올라가지 않았다. **정직하게(학습에 안 쓴 15개 사건만으로) 채점하면
0.32로 v5(0.379)보다 오히려 떨어진다.** 이유는 DEVLOG 19차가 이미 경고했던 그 문제 —
"TCN 자체를 트리거로 쓰면 노이즈에 훨씬 약하다" — 가 그대로 재현됐기 때문이다.

---

## 1. 증강 데이터 확인 — 있었고, 그대로 재사용해 학습을 실행했다

기존에 이 저장소에는 `train_tcn_v6b_trigger.py`가 **이미 작성돼 있었지만 실행되지 않은 채**
남아 있었다(모델 파일 없음). 코드를 확인한 결과 `train_tcn_benchmark_overfit.py`의 증강 4종을
그대로 재사용하도록 설계돼 있었다:

1. 좌우 미러링 (`mirror_sequence`)
2. 타이밍 앵커 지터 (`{-150,0,+150}ms`)
3. 시간축 워핑 (`0.9/1.0/1.1`배)
4. 가우시안 피처 지터 (MAD 기반 5%)

**이번에 실제로 실행**(`python motion_learning/train_tcn_v6b_trigger.py`)해 아래 결과를 확인했다:

| 항목 | 값 |
|---|---:|
| TRAIN_GT (학습에 쓴 정답) | 14개 |
| TEST_GT (학습에서 완전 제외) | 15개 |
| Purge 반경 | ±2500ms |
| 양성 base 윈도 (purge 후) | 13개 (42개 후보 중 29개 purge로 제외) |
| 음성 base 윈도 (purge 후) | 92개 |
| **증강 후 최종 학습 샘플** | **1,338개** |
| 학습셋 자기 재현 정확도 | 100.00% |

→ 산출물: `motion_learning/v6b_tcn_trigger/{boxing_tcn.pth, boxing_tcn_scaler.json, train_report.json}`
(기존 배포 모델·v6 overfit 모델 둘 다 건드리지 않음).

`v6`(overfit_hong)과의 핵심 차이: v6는 29개 정답 전부로 학습해 "얼마나 외울 수 있는가"만 봤다.
이번 실험은 **15개를 학습에서 완전히 빼고(purge까지 적용)** held-out 성능을 정직하게 재려 했다.

## 2. TCN이 실제로 쓰였는지 검증

이전에 발견됐던 `import sys` 누락 버그(TCN이 조용히 rule-base로 폴백되던 문제)는 이번 실험과
무관한 **다른 스크립트**(`eval/evaluate_video.py`)의 문제였고, git pull로 이미 수정본이
반영돼 있었다(`import sys`가 18번째 줄에 존재 확인). 이번 실험은 그 스크립트를 아예 쓰지
않고 `motion_learning/evaluate_tcn_v6b.py`를 새로 작성해 **룰베이스 트리거를 완전히 우회**하고
TCN 단독으로 붙였으므로, 별도로 다음을 직접 확인했다:

| 확인 항목 | 결과 |
|---|---|
| `model.load_state_dict()` 예외 없이 로드 | ✅ (`v6b_tcn_trigger/boxing_tcn.pth`) |
| 파라미터 수 | 20,556개 (0이면 로드 실패 의심 — 정상) |
| 90초 전체, 매 프레임 forward pass 성공 (`verified_tcn_forward_success`) | ✅ True (2,654프레임 전부) |
| 예측에 등장한 클래스 다양성 | `STRAIGHT_L/R, HOOK_L/R, UPPERCUT_L/R` 전부 등장 (룰베이스라면 나올 수 없는 조합·분포) |
| confidence 분포 | 0.446 ~ 0.9999 (상수/기본값이 아니라 실제 softmax 출력) |

→ **rule-base 폴백이 아니라 TCN이 실제로 매 프레임 추론해 트리거를 결정했다는 것을 확인.**
룰베이스(`punch_core.py`/`try_punch`)는 이 실행 경로에서 아예 import되지도 않는다
(`evaluate_tcn_v6b.py`가 참조하는 모듈에 없음).

## 3. 결과 — 왜 F1이 오히려 떨어졌나

### 정직한 채점 (TEST_GT 15개, leakage 없음) — **핵심 지표**

| 지표 | v5_tcn_optimized (룰베이스 트리거) | **v5b_tcn_trigger (TCN 트리거)** |
|---|---:|---:|
| Precision | 0.379 | **0.20** |
| Recall | 0.379 | **0.80** |
| **F1** | **0.379** | **0.32** |
| 예측 개수(90초 전체) | 29 | **60** |

`eval/runs/v5b_tcn_trigger/metrics.json` 참고 (`all_predictions_full_90s`에 60개 예측 전부 기록,
`raw_trace.jsonl`에 2,654프레임 각각의 예측 클래스·confidence 전부 기록 — 재현/디버그용).

### 무슨 일이 일어났나

- **Recall은 크게 좋아졌다** (0.379 → **0.80**, 15개 중 12개 적중). TCN이 시계열 패턴을 보므로
  룰베이스 물리 임계값보다 실제 펀치를 더 잘 잡아낸다는 것 자체는 사실이다.
- 하지만 **90초 동안 60번을 쏘았다** (GT는 29개, 이 중 실제 정답으로 인정되는 건 TEST_GT 15개뿐).
  대가로 Precision이 0.20까지 무너졌다 — **DEVLOG 19차가 TCN 자체 트리거를 폐기했던 바로 그
  이유가 재현됐다**: "분류기가 그 순간 뭐라고 보는가"만으로 트리거를 걸면, `punch_core`의
  "실제 속도·뻗음이 물리 임계값을 넘어야 창이 열린다"는 물리적 게이트가 없어서 프레임 단위
  요동(flicker)에 훨씬 약하다. edge-trigger + 390ms 쿨다운을 넣었는데도 60번이나 발사됐다는 것은,
  클래스 예측 자체가 프레임마다 자주 바뀐다는 뜻이다.
- **(참고, leakage 있음) 전체 29개 GT 기준으로 채점하면 F1=0.4719**로 v5보다 높게 나온다
  (`metrics.json`의 `full_90s_leaky_metrics`). 하지만 이 중 14개는 모델이 학습 때 이미 본
  시각대(TRAIN_GT)라 낙관적으로 부풀려진 값이다 — **정직한 지표가 아니므로 참고용으로만
  남긴다.** (헤드라인으로 쓰면 안 됨 — 실제 배포 시 이 정도 일반화는 기대할 수 없다.)

## 4. 결론 — 이 접근은 채택하지 않는 게 맞다

- **"TCN이 트리거까지 담당하면 F1이 오르는가"에 대한 답: 아니다.** 정직한 held-out 평가에서
  F1이 0.379 → 0.32로 **오히려 떨어진다.** Recall은 개선되지만 Precision 붕괴가 그보다 크다.
- TCN을 트리거에 쓰려면 **물리적 게이트(속도·뻗음 최소 조건) 없이 분류기 확신도만으로 문을
  여는 구조 자체가 근본 원인**이다 — 이는 학습 데이터를 더 늘리거나 모델을 더 키운다고 풀리는
  문제가 아니라(v6 실험에서 이미 "이 사람에 overfitting해도 kind_accuracy는 그대로"임을 확인한
  것과 같은 종류의 결론), 트리거 자체의 설계 문제다.
- **권장**: 기존 설계 원칙(트리거=룰베이스, 분류=TCN)을 유지한다. TCN을 트리거에 쓰고 싶다면
  "TCN 확신도"와 "룰베이스 물리 게이트"를 **AND로 결합**하는 하이브리드(둘 다 통과해야 발사)를
  다음으로 시도해볼 수 있다 — 이번 실험처럼 TCN 단독으로 완전히 대체하는 것은 재현 결과상
  권장하지 않는다.

## 재현 방법

```bash
# 1) leakage 방지 train/test 분할로 학습 (증강 4종 포함, 이미 존재하던 스크립트를 실행만 함)
python motion_learning/train_tcn_v6b_trigger.py
# -> motion_learning/v6b_tcn_trigger/{boxing_tcn.pth, boxing_tcn_scaler.json, train_report.json}

# 2) 룰베이스 트리거를 완전히 우회하고 TCN 단독으로 90초 전체 causal 재생 + 채점
python motion_learning/evaluate_tcn_v6b.py
# -> eval/runs/v5b_tcn_trigger/{metrics.json, raw_trace.jsonl}
```

**신규 파일**: `motion_learning/evaluate_tcn_v6b.py`. 기존 `train_tcn_v6b_trigger.py`는 이번에
처음 실행했을 뿐 수정하지 않음. `eval/evaluate_video.py`(룰베이스 트리거 경로)는 이 실험과
무관하며 건드리지 않음.
