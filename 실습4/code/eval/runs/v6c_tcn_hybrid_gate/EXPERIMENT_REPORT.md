# 🧪 v6c — v6b에 "TCN 확신도 AND 룰베이스 물리조건" 게이트 추가

**요청**: v6b(TCN이 트리거까지 전담)에서, 다른 세션이 v5c에서 쓴 패턴처럼 "TCN 확신도 AND
룰베이스 물리조건"으로 트리거 조건을 바꿔서 평가.

**먼저 알려드릴 것**: 작업을 시작해보니, **바로 이 실험이 다른 세션에서 `v5c_tcn_hybrid_gate`로
이미 수행돼 있었습니다** — `motion_learning/v6b_tcn_trigger`(제가 만든 leakage-safe 모델)를
그대로 쓰고, `eval/evaluate_video.py`에 `TCNHybridTriggerEvaluator`(`--engine tcn_hybrid`)까지
이미 구현·배선돼 있었습니다. 그래서 이번 v6c는 엔진을 새로 만들 필요 없이, **그 엔진을 그대로
불러 v6c라는 이름으로 정식 채점**했고, 추가로 **게이트 자체의 순수 효과**를 분리해 보는
ablation을 하나 더 돌렸습니다.

---

## 0. 무엇이 바뀌었나

```python
class TCNHybridTriggerEvaluator(TCNTriggerEvaluator):
    def _physics_gate_ok(self, label, feat):
        side = "L" if label.startswith("LEFT_") else "R"
        speed = feat[10] if side == "L" else feat[11]   # l_speed / r_speed
        reach = feat[2] if side == "L" else feat[3]      # l_reach / r_reach
        return speed > TCN_HYBRID_ARM_GATE and reach > TCN_HYBRID_EXTEND_GATE
```

v6b(`TCNTriggerEvaluator`)의 확정(confirm)·엣지 트리거·쿨다운 로직은 그대로 두고, "확정 후보로
인정할지" 판단에 **한 줄**을 더 추가한 것 — TCN이 펀치라고 확신하는 것(`conf ≥ MIN_CONF`) **AND**
그 순간 해당 팔이 실제로 물리적으로 뻗어나가는 중인가(`speed > PUNCH_ARM` 수준,
`reach > PUNCH_EXTEND` 수준). **둘 다** 있어야 후보로 인정한다. TCN 혼자서도, 물리 조건 혼자서도
열 수 없다.

---

## 1. 결과 — 세 가지를 나란히 비교 (전부 같은 leakage-safe 모델, 같은 test-only 채점)

모델은 전부 `motion_learning/v6b_tcn_trigger`(benchmark.mp4의 15개 정답을 학습에서 제외하고
학습한 그 모델) 하나다. **달라지는 건 트리거 조건과 디바운스 파라미터뿐.**

| 버전 | 트리거 조건 | confirm/cooldown | TP | FP | FN | Precision | Recall | **F1** | kind_acc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **v6b** | TCN 확신도만 | 2프레임 / 350ms | 10 | 10 | 5 | 0.500 | 0.667 | **0.571** | 0.600 |
| **v6c_ablation**(게이트만 추가, 나머지 동일) | TCN **AND** 물리조건 | 2프레임 / 350ms | 11 | 10 | 4 | 0.524 | 0.733 | **0.611** | 0.545 |
| **v6c**(게이트 + v5c와 동일 튠) | TCN **AND** 물리조건 | 3프레임 / 600ms | 10 | 3 | 5 | 0.769 | 0.667 | **0.714** | 0.500 |

(모두 15개 test-only 정답 기준, leakage-free. `eval/evaluate_v6c_held_out.py`)

### 해석 — 개선 효과를 두 부분으로 쪼개서 봐야 한다

1. **물리 게이트 자체의 순수 효과** (v6b → v6c_ablation, 디바운스 파라미터는 건드리지 않음):
   FP는 그대로(10)지만 **FN이 5→4로 줄어 recall이 올랐다**(0.667→0.733). 즉 이 설정에서
   게이트는 "오검출을 막는" 효과보다는, TCN 단독 모드에서 쿨다운에 걸려 막혔던 **진짜 펀치를
   오히려 하나 더 살려준** 효과가 더 크게 나타났다. F1은 0.571→0.611로 소폭 개선.
2. **디바운스 재튜닝의 효과** (v6c_ablation → v6c, 게이트는 고정, confirm 2→3프레임·
   cooldown 350→600ms): **FP가 10→3으로 크게 줄었다.** v6c의 F1 0.714 중 절반 이상의
   개선분은 사실 **"더 엄격하게, 더 오래 기다려야 확정한다"는 디바운스 튠 자체**에서 나온다.

**결론**: "TCN AND 룰베이스 물리조건"이라는 아이디어는 방향이 맞다(게이트만 추가해도 효과가
있다 — 0.571→0.611). 하지만 v5c/v6c가 보여준 가장 큰 숫자(0.714)는 게이트 하나의 효과가
아니라 **게이트 + 디바운스 재튜닝을 같이 바꾼 결과**다. 이 둘을 섞어서 "게이트가 F1을 0.571→0.714로
올렸다"고 말하면 과장이다.

---

## 2. 전체 90초(참고, leakage 있음) — 다른 세션의 v5c와 재현성 확인

| | 예측 | TP | FP | FN | F1 | 비동작 FP |
|---|---:|---:|---:|---:|---:|---:|
| v5c_tcn_hybrid_gate(다른 세션) | 30 | 17 | 13 | 12 | 0.5763 | 0 |
| **v6c(이 리포트)** | 30 | 17 | 13 | 12 | **0.5763** | **0** |

**숫자가 완전히 일치한다** — 같은 모델·같은 엔진·같은 config를 독립적으로 다시 돌려도 동일한
결과가 나온다는 재현성 확인이다(당연한 결과이지만, 파이프라인이 안정적이라는 근거로 기록해 둔다).

---

## 3. ⚠️ 중요한 방법론 차이 — "test-only" 숫자가 세션마다 다르게 나왔다

다른 세션이 남긴 `v5c_tcn_hybrid_gate/metrics.json`의 `test_gt_only_honest_metrics`는
**FP=20, F1=0.4444**로 기록돼 있다. 저는 **같은 예측 30개로 FP=3, F1=0.7143**을 얻었다 —
같은 데이터에서 이렇게 다른 숫자가 나온 이유를 반드시 설명해야 한다.

- **다른 세션의 방식**: 예측 30개 **전부**를 놔두고, 정답만 15개로 줄여서 채점했다. 그러면
  train 구간(직선/훅/어퍼컷 앞쪽 반복)에서 **정확하게 잘 맞춘 예측**까지 "어느 test 정답과도
  안 맞으니 FP"로 잡힌다 — train 구간에서 잘하는 건 당연한데 그걸 오검출로 벌점 받는 셈이다.
- **제 방식**(`eval/evaluate_v6b_held_out.py`·`evaluate_v6c_held_out.py`): 예측도 **test
  시간대에 속하는 것만** 추려서 채점한다(정답과 똑같이). Train 구간 예측은 잘했든 못했든
  평가 범위 밖으로 아예 빼버린다 — held-out test 평가의 표준적인 방식이다.

**제 방식이 맞는 방법론이라고 판단합니다.** "이 시간대에서 처음 보는 패턴을 얼마나 잘
맞히는가"를 보려면, 채점 자체를 그 시간대로 한정해야 합니다 — 안 그러면 "학습 구간에서
잘했다"는 (당연한, 그리고 무의미한) 사실이 "테스트 구간에서 못했다"는 주장과 뒤섞입니다.
다만 다른 세션의 `metrics.json`을 제가 수정하지는 않았습니다 — 숫자가 다른 이유를 이 리포트에
남기는 것으로 대신합니다.

---

## 4. TCN이 실제로 쓰였는지 검증

`evaluate_v6c_held_out.py`가 실행마다 자동 확인한다 (v6b와 동일한 방식):

```
[검증] tcn_hybrid 엔진이 실제로 TCN을 썼는가?
  - 프로세스 종료 코드: 0
  - '🧠 [TCN ... 로드 완료' 로그 존재: True
  - '로드 실패'(폴백) 로그 존재: False
  => ✅ TCN이 정상적으로 로드되어 전체 트리거/분류를 직접 수행했다.
```

---

## 부록 — 재현

```bash
# v6c (v5c와 동일 튠: confirm=3, cooldown=600ms, gate 포함)
python eval/evaluate_v6c_held_out.py
# → eval/runs/v6c_tcn_hybrid_gate/{report.json, punches.csv, metrics_test_only.json}

# ablation: confirm/cooldown은 v6b와 동일하게 두고 게이트만 추가
python -m eval.run_pipeline_style... # 또는 config만 바꿔 evaluate_v6b_held_out.py 의
                                      # run_engine(engine="tcn_hybrid",
                                      #            config_path="eval/configs/v6c_ablation_gate_only.json", ...)
```

**새로 만든 파일**: `eval/configs/v6c_tcn_hybrid_gate.json`, `eval/configs/v6c_ablation_gate_only.json`,
`eval/evaluate_v6c_held_out.py`. `eval/evaluate_v6b_held_out.py`는 엔진/config/out_dir을
인자로 받도록 일반화(기존 v6b 동작은 그대로 유지). **엔진 코드(`TCNHybridTriggerEvaluator`
등)는 건드리지 않았다** — 이미 다른 세션이 구현해 둔 것을 그대로 재사용.
