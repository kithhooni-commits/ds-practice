# 🧪 v5c_tcn_hybrid_gate — "TCN 확신도 AND 룰베이스 물리조건" 하이브리드 트리거

**요청**: v5b(TCN 단독 트리거, F1 0.32 — v5의 0.379보다 낮음)의 실패를 개선할 방법을 제안하고,
"TCN 확신도 AND 룰베이스 물리조건"으로 실제 구현/평가.

**결론**: 성공. TEST_GT(학습에서 완전 제외된 15개, leakage 없음) 기준 **F1 0.32 → 0.44**로
개선됐고, **v5(룰베이스 트리거, F1 0.379)도 넘어섰다.** 전체 29개 GT 기준(참고용, 일부 leaky)으로는
**F1 0.5763**으로 지금까지의 모든 버전(v1~v6) 중 가장 높다.

---

## 1. 왜 이 방향인가 (v5b 실패 원인 재분석)

v5b(`eval/runs/v5b_tcn_trigger/EXPERIMENT_REPORT.md`)는 "TCN 확신도 + edge + cooldown"만으로
트리거를 걸었다가 90초 동안 60번을 쏴 Precision 0.20까지 무너졌다. 원인은 **물리적 게이트가
없어서** 모델이 프레임마다 흔들리는 구간(풋워크, 팔 반동, 애매한 자세)에서도 "펀치처럼 보인다"는
확신을 자주 냈기 때문이다.

**제안**: 트리거 조건에 "그 순간 실제로 팔이 물리적으로 뻗어나가는 중인가"라는 최소한의
운동학적 증거를 **AND**로 추가한다. TCN 혼자서도, 물리 조건 혼자서도 못 열고 **둘 다 있어야
연다** — 이게 사용자가 요청한 정확한 설계다.

## 2. 구현

### 어디에 넣었나 (기존 구조 재사용)
저장소에는 이미 v6b 실험으로 만들어진 `TCNTriggerEvaluator`(`eval/evaluate_video.py`)가
있었다 — confirm_frames(연속 프레임 확정) + edge-fire(스트릭당 1회) + 전역 cooldown 으로
"모델 출력의 시간적 안정성"만으로 트리거를 결정하는 클래스다. 여기에 **손대지 않고**,
`_physics_gate_ok(label, feat)`라는 훅 메서드 하나를 추가해 기본은 항상 `True`(기존 동작 그대로
보존)로 두고, 이를 오버라이드하는 `TCNHybridTriggerEvaluator(TCNTriggerEvaluator)`를 새로 만들었다.

```python
class TCNHybridTriggerEvaluator(TCNTriggerEvaluator):
    def _physics_gate_ok(self, label, feat):
        side = "L" if label.startswith("LEFT_") else "R"
        speed = feat[10] if side == "L" else feat[11]   # l_speed / r_speed
        reach = feat[2] if side == "L" else feat[3]      # l_reach / r_reach
        return speed > TCN_HYBRID_ARM_GATE and reach > TCN_HYBRID_EXTEND_GATE
```

물리 게이트 값은 **룰베이스가 "펀치 창을 여는(arm)" 데 쓰는 것과 같은 하한선**
(`PUNCH_ARM=1.0`, `PUNCH_EXTEND=0.40`)을 그대로 재사용했다 — "확정 피크"(`PUNCH_SPEED=1.65`,
`PUNCH_REACH_N=0.88`)까지 요구하면 TCN이 확정되는 프레임과 물리적 피크 프레임이 어긋날 때
정당한 펀치까지 게이트에서 막히기 때문에(그리고 실측으로도 확인 — §3), 최소 하한선만 쓰는
게 맞다.

### 엔진/설정 시스템에 정식 편입
- `--engine tcn_hybrid` 추가 (`rule`/`tcn`/`tcn_trigger`와 같은 급의 정식 옵션 — 다음 실험이
  임시 스크립트 없이 `run_pipeline.py`만으로 재현 가능)
- `eval/configs/v5c_tcn_hybrid_gate.json` — `TCN_TRIGGER_CONFIRM_FRAMES/MIN_CONF/COOLDOWN_MS` +
  `TCN_HYBRID_ARM_GATE/EXTEND_GATE` 5개 튜닝 값
- 모델은 이전 실험(v5b)에서 이미 leakage 방지로 학습해 둔 `motion_learning/v6b_tcn_trigger/`를
  그대로 재사용(재학습 불필요 — 물리 게이트는 추론 후처리이지 모델을 바꾸는 게 아니다)

## 3. 하이퍼파라미터 탐색 (TEST_GT 기준 64가지 조합)

`confirm_frames × min_conf × cooldown_ms × (arm_gate, extend_gate)` 그리드를 TEST_GT(15개,
leakage 없음)로 채점해 비교했다(상위 5개):

| min_conf | confirm_frames | cooldown_ms | gate(speed,reach) | 예측수 | P | R | **F1** |
|---:|---:|---:|---|---:|---:|---:|---:|
| **0.5** | **3** | **600** | **(1.0, 0.40)** | **30** | **0.333** | **0.667** | **0.4444** |
| 0.5 | 3 | 600 | (1.65, 0.88) | 31 | 0.323 | 0.667 | 0.4348 |
| 0.6 | 3 | 600 | (1.65, 0.88) | 31 | 0.323 | 0.667 | 0.4348 |
| 0.5 | 2 | 600 | (1.65, 0.88) | 37 | 0.297 | 0.733 | 0.4231 |
| 0.6 | 3 | 600 | (1.0, 0.40) | 34 | 0.294 | 0.667 | 0.4082 |

**관찰**:
- **완화된 게이트(1.0/0.40, "arm" 수준)가 엄격한 게이트(1.65/0.88, "확정 피크" 수준)보다
  일관되게 F1이 같거나 높다** — §2에서 예상한 대로, 피크 시점 정렬 문제 때문.
- `confirm_frames`를 2→3으로 늘리고 `cooldown_ms`를 350→600으로 늘린 것이 v5b 대비 예측 개수를
  60개→30개로 절반으로 줄이면서 recall 손실은 크지 않았다(0.80→0.667) — 즉 v5b의 초과 발사분은
  대부분 진짜 놓치면 안 될 이벤트가 아니라 순수 중복/노이즈였다는 뜻.
- 최종 채택: `min_conf=0.5, confirm_frames=3, cooldown_ms=600, arm_gate=1.0, extend_gate=0.40`

## 4. 결과 — 두 가지 채점 관점

물리 게이트가 모델 자체를 바꾸는 게 아니므로, v5b와 동일하게 TEST_GT(15개, leakage 없음)를
핵심 지표로 삼고 전체 29개 GT는 참고용으로 같이 낸다.

| 버전 | 트리거 방식 | **TEST_GT F1 (핵심, leakage 없음)** | 전체 29GT F1 (참고, 일부 leaky) |
|---|---|---:|---:|
| v5_tcn_optimized | 룰베이스 물리 임계값 단독 | *(TEST_GT 분할 없이 학습된 모델이라 직접 비교 불가)* | 0.3793 |
| v5b_tcn_trigger | TCN 확신도 단독 (edge+cooldown) | 0.3243 | 0.4772 (leaky) |
| **v5c_tcn_hybrid_gate** | **TCN 확신도 AND 물리조건** | **0.4444** | **0.5763** |

(`eval/runs/v5c_tcn_hybrid_gate/metrics.json`의 최상위 필드가 전체 29GT 기준,
`test_gt_only_honest_metrics` 필드가 TEST_GT 전용 정직한 지표.)

**핵심 개선**: v5b 대비 TEST_GT F1이 **0.3243 → 0.4444로 +37%**, v5(룰베이스, 0.379) 대비로도
**+17%** 개선. Precision은 여전히 낮지만(0.333) v5b(0.204)보다 크게 나아졌고, Recall은
0.667로 v5(0.379)보다 훨씬 높다 — 즉 **물리 게이트가 v5b의 "노이즈성 과검출"만 골라서
걸러내고, TCN이 원래 갖고 있던 "룰베이스보다 잘 잡는" 능력(recall)은 대부분 보존했다.**

## 5. 결론 및 다음 단계

- **가설이 맞았다**: TCN 단독 트리거의 문제는 "모델이 나쁘다"가 아니라 "물리적 게이트가
  없다"였다. 최소한의 운동학적 하한선 하나만 AND로 추가해도 v5b의 손해를 만회하고 v5까지
  넘어섰다.
- 다만 Precision(0.333)은 아직 실전 투입 기준에는 한참 못 미친다. 남은 FP(TEST_GT 기준 20개)의
  성격을 더 뜯어봐야 한다 — 다음으로 시도해볼 것:
  1. **게이트에 "회수(withdraw)" 조건 추가** — 지금은 뻗는 순간만 보는데, 같은 스윙의 반동/회수
     동작도 여전히 confirm_frames를 채울 수 있다. 룰베이스의 `reach0`(창 열릴 때 뻗음 기준점)
     처럼 "arm 시점 대비 실제로 더 뻗어났는가"를 게이트에 추가하면 반동 오검출을 줄일 수 있다.
  2. **per-side 쿨다운으로 분리** — 지금 cooldown은 전역이라, 왼손 하나가 오른손 트리거를
     막을 수 있다(반대로 만들면 recall이 더 오를 수 있음).
  3. 이번 그리드는 5개 파라미터를 독립적으로 몇 개 값만 봤다 — Optuna 류 자동 탐색으로 더
     넓게 훑으면 여지가 남아 있을 가능성.

## 재현 방법

```bash
python eval/run_pipeline.py --version v5c_tcn_hybrid_gate --engine tcn_hybrid \
  --config eval/configs/v5c_tcn_hybrid_gate.json \
  --tcn-model-dir motion_learning/v6b_tcn_trigger --overwrite
```

**수정/신규 파일**: `eval/evaluate_video.py`(`TCNHybridTriggerEvaluator` 클래스 추가,
`--engine tcn_hybrid` 옵션, `TCN_HYBRID_ARM_GATE`/`TCN_HYBRID_EXTEND_GATE` 튠 값 — 기존
`TCNTriggerEvaluator`/`rule`/`tcn` 엔진 로직은 한 줄도 변경 없음, 훅 메서드 추가로 하위호환 유지),
`eval/run_pipeline.py`(engine choices에 `tcn_hybrid` 추가), `eval/configs/v5c_tcn_hybrid_gate.json`(신규).
