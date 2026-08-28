# 🧪 v6b — TCN이 트리거까지 직접 담당 (leakage-free 평가)

**요청 2건에 대한 답**
1. v6(TCN 분류만)에서 **트리거도 TCN이 담당**하도록 바꿔 평가 → 했음, 결과는 §2·§3.
2. benchmark.mp4 + augmentation 데이터에서 **test에 해당하는 데이터는 학습에서 제외**(leakage 방지) → 했음, 분할 방법은 §1.
3. (지난 대화에서 나온) "TCN으로 된 게 맞는지 검증" → §4에서 직접 확인.

---

## 0. 아키텍처 변경 — 무엇이 바뀌었나

v1~v6는 전부 "트리거는 항상 rule-base, TCN은 트리거가 열린 다음 종류만 답한다" 구조였다
(지난 질문에서 설명한 구조 그대로). **v6b는 이 경계를 없앴다.**

```
[v1~v6]  좌표 → 속도·뻗음 계산(rule) → PUNCH_ARM/SPEED/REACH_N 임계값 통과? → (rule 또는 TCN으로 종류만 결정)
[v6b]    좌표 → 17차원 피처 → TCN 10-클래스 예측 → 연속 프레임 안정성만으로 트리거+종류 동시 결정
```

`eval/evaluate_video.py`에 `TCNTriggerEvaluator`(`--engine tcn_trigger`)를 새로 추가했다.
**`PUNCH_ARM`·`PUNCH_SPEED`·`PUNCH_REACH_N` 같은 키네마틱 임계값을 어디에도 쓰지 않는다.**
대신:

1. **확정(confirm)**: 같은 펀치 클래스가 `CONFIRM_FRAMES=2`프레임 연속으로, 매번 `MIN_CONF=0.5`
   이상의 확신도로 나와야 "확정".
2. **엣지 트리거**: 확정되는 그 프레임에서 1회만 발사. 같은 스트릭이 계속돼도 재발사 안 함
   (`already_fired_this_streak`) — DEVLOG가 기록한 "레벨 트리거라 무한 재발동" 버그를
   의도적으로 피한 설계.
3. **스트릭 리셋**: 예측이 펀치가 아니거나 확신도가 떨어지면 스트릭 종료 → 다음 확정은 새 이벤트.
4. **전역 쿨다운**: `COOLDOWN_MS=350`ms — 스트릭이 짧게 깨졌다가 같은 스윙으로 바로
   재확정되는 경우의 중복 발사를 한 번 더 막는다.

---

## 1. Leakage 방지 — Train/Test 분할

`motion_learning/train_tcn_v6b_trigger.py`. 단순 "앞부분=train, 뒷부분=test"는 한쪽 펀치
종류가 test에서 통째로 빠질 수 있어서, **클러스터(직선/훅/어퍼컷)마다 마지막 1~2회를 test로
빼는 방식**을 썼다(콤보 구간은 이벤트가 너무 촘촘해 쪼갤 수 없어 통째로 test).

| | 구성 | 개수 |
|---|---|---:|
| **TRAIN** | 직선 앞 3+3, 훅 앞 2+2, 어퍼 앞 2+2 (L/R 각각) | 14개 |
| **TEST**  | 직선/훅/어퍼 각 클러스터의 마지막 1회(L+R) + 콤보 9개 전체 | 15개 |

**Leakage 방지 규칙**: 학습 후보(양성 anchor든, 휴식/풋워크에서 뽑은 음성 샘플이든) 중
**테스트 이벤트 시각에서 ±2500ms 이내**는 전부 버린다(60프레임≈2000ms 윈도보다 넉넉한
버퍼). 실행 결과:

```
양성 base 윈도: 13개 유지, 29개(14개 train 이벤트 × 3개 타이밍 오프셋 중 다수) purge로 제외
음성 base 윈도: 92개 유지, 21개 purge로 제외
증강 후 최종 학습 샘플: 1,338개
```

**채점도 거꾸로** — 모델은 90초 전체를 causal하게(과거만 보며) 한 번 재생하지만(이건 누설이
아니다, 가중치 갱신이 없는 순전파일 뿐), **점수는 test 시각대에 속하는 예측·정답만 사용**한다
(`eval/evaluate_v6b_held_out.py`). Train 구간에서 나온 예측은 점수 계산에서 완전히 제외된다.

---

## 2. 결과

### 2-1. 전체 90초 (참고용 — **leakage-free 아님**, train 구간 포함)

| 지표 | 값 |
|---|---:|
| 예측 / 정답 | 40 / 29 |
| TP / FP / FN | 17 / 23 / 12 |
| Precision / Recall / **F1** | 0.425 / 0.586 / **0.4928** |
| kind_accuracy | 0.647 |
| **비동작 구간(준비·휴식4회·풋워크·마무리) 오검출** | **0회 (!)** |

**가장 눈에 띄는 변화**: 준비/휴식/풋워크/마무리 — 40개 예측 전부가 "동작이 있어야 하는"
구간(직선/훅/어퍼컷/콤보)에서만 나왔다. v1~v6 전부 9~11회씩 나던 풋워크 오검출이 **0이 됐다.**
대신 각 동작 구간 안에서 "한 번 휘두른 걸 여러 번 잡는" 과검출(예: 직선 구간 GT 8개인데 예측 14개)은
여전히 있다 — 이건 다른 문제(쿨다운/디바운스 튜닝)다.

### 2-2. TEST-ONLY (진짜 결과 — **leakage-free**, 15개 정답만)

| 지표 | 값 |
|---|---:|
| 예측 / 정답 | 20 / 15 |
| TP / FP / FN | **10 / 10 / 5** |
| Precision / Recall / **F1** | 0.500 / 0.667 / **0.5714** |
| kind_accuracy (TP 10개 중) | **0.600** |
| 혼동 | `STRAIGHT→STRAIGHT 6, HOOK→UPPERCUT 1, STRAIGHT→UPPERCUT 3` |

### 2-3. v1~v6 전체 비교

| 버전 | 트리거 | 분류 | F1 | kind_acc | 비동작 FP |
|---|---|---|---:|---:|---:|
| v1_baseline | rule | rule | 0.367 | – | 9 |
| v2_anti_sway | rule | rule | 0.346 | – | 4 |
| v3_iter4_eval | rule | rule | 0.370 | – | 5 |
| v4_tcn_hybrid | rule | TCN(게이팅) | 0.367 | – | 9 |
| v5_tcn_optimized | rule | TCN(게이팅) | 0.379 | – | 7 |
| v6_tcn_overfit_hong | rule | TCN(이 영상에 overfit) | 0.379 | 0.455† | 7 |
| **v6b_tcn_trigger (전체 90s, 참고)** | **TCN** | **TCN** | **0.493** | 0.647 | **0** |
| **v6b_tcn_trigger (TEST-ONLY, 진짜 성능)** | **TCN** | **TCN** | **0.571** | 0.600 | **0**(test 구간 내) |

† v6의 45.5%는 트리거가 이미 rule-base로 골라준 순간에 대해서만 평가된 값이라 v6b와
직접 비교에는 주의가 필요하다(전제가 다름).

**v6b(test-only)가 역대 최고 F1(0.571)을 기록했다** — 이전 최고였던 v5(0.379)보다
+0.19 높다. 비동작 구간 오검출도 처음으로 0을 달성했다.

---

## 3. 이 결과를 어떻게 읽어야 하나 — 과장하지 않기

1. **test GT가 15개뿐이다.** TP가 한두 개만 바뀌어도 F1이 크게 흔들린다(예: TP 1개만
   늘면 recall이 0.667→0.733). "역대 최고"는 맞지만 통계적으로 불안정한 숫자라는 것도
   같이 말해야 한다.
2. **동작 구간 내 과검출은 해결된 게 아니다.** §2-1에서 본 "직선 구간 GT 8개, 예측 14개"처럼,
   한 번의 스윙이 여러 번 잡히는 문제는 v1~v6와 마찬가지로 남아 있다. 이번에 없어진 건
   "비동작 구간"오검출 한 종류뿐이다.
3. **왜 비동작 구간 오검출이 0이 됐는지**: rule-base는 "속도/뻗음이 임계값을 넘으면" 무조건
   펀치로 인정했고, 풋워크의 상체 흔들림이 가끔 그 임계값을 넘었다(이전 대화에서 분석한
   내용). TCN은 애초에 "이 17차원 패턴이 펀치 클래스처럼 보이는가"를 전체 모양으로 판단하므로,
   풋워크의 흔들림 패턴 자체가 학습된 펀치 패턴과 뚜렷이 다르면 걸리지 않는다 — **단, 이건
   "설계상 원천적으로 불가능하다"가 아니라 "이번 학습 데이터·이번 영상에서는 그랬다"**이며,
   다른 사용자/다른 조명/다른 카메라 각도에서도 유지되는지는 확인되지 않았다.

---

## 4. "TCN이 실제로 쓰인 게 맞는가" 검증

`eval/evaluate_v6b_held_out.py`가 실행할 때마다 자동으로 확인한다:

```
[검증] TCN 트리거 엔진이 실제로 쓰였는가?
  - 프로세스 종료 코드: 0 (0이면 SystemExit 안 걸렸다는 뜻)
  - '🧠 [TCN Trigger Engine]' 로그 존재: True
  - '로드 실패'(폴백) 로그 존재: False
  => ✅ TCN 트리거 엔진이 정상적으로 로드되어 전체 트리거/분류를 직접 수행했다.
```

이게 의미 있는 검증인 이유: `evaluate_video.py`는 이제(다른 세션에서 수정됨)
`--engine tcn`/`tcn_trigger`를 요청했는데 모델 로드가 실패하면 **조용히 rule-base로
폴백하지 않고 `SystemExit`로 파이프라인 자체를 중단시킨다.** 지난번 v4/v5를 오염시켰던
`import sys` 누락 버그가 고쳐진 것과 별개로, **"예외로 죽지 않고 끝까지 돌았다"는 사실 자체가
이미 TCN이 로드됐다는 증거**이고, 로그 문자열 대조는 2차 확인이다. 또한 `TCNTriggerEvaluator`는
애초에 rule-base 코드 경로(`PunchEvaluator.try_punch`, `arm_kinematics`)를 **호출조차 하지
않는 별개 클래스**이므로(§0 참고), "트리거가 진짜 TCN인가"라는 질문에는 코드 구조 자체가
답이다 — rule 임계값 변수(`PUNCH_ARM` 등)가 `TCNTriggerEvaluator.process()` 안에 한 번도
나타나지 않는다.

---

## 5. 다음 후보

1. **동작 구간 내 과검출 억제** — 같은 스윙이 여러 번 잡히는 문제. `CONFIRM_FRAMES`를
   늘리거나 `COOLDOWN_MS`를 늘려보되, 실제 최소 콤보 간격(433ms, 76500→76933)보다는
   짧게 유지해야 한다.
2. **test 셋 확장** — 15개는 너무 적다. 추가로 녹화하거나, k-fold 식으로 클러스터별
   홀드아웃 위치를 바꿔가며 여러 번 평가해 분산을 직접 측정하는 것을 권장.
3. **다른 사용자/영상에서 "비동작 FP=0"이 재현되는지 확인** — 지금은 단일 영상·단일
   사용자 결과다.

---

## 부록 — 재현

```bash
# 1) leakage-free 학습 (train 14개 / test 15개 분할, purge ±2500ms)
python motion_learning/train_tcn_v6b_trigger.py
# → motion_learning/v6b_tcn_trigger/{boxing_tcn.pth, boxing_tcn_scaler.json, train_report.json}

# 2) 전체 90초 재생 + TCN 로드 검증 + test-only 채점
python eval/evaluate_v6b_held_out.py
# → eval/runs/v6b_tcn_trigger/{report.json, punches.csv, metrics_test_only.json}
```

**수정된 파일**: `eval/evaluate_video.py`(`TCNTriggerEvaluator` 신규, `--engine tcn_trigger`,
`--tcn-model-dir` 복원 + 경로 보강), `eval/run_pipeline.py`(`--tcn-model-dir`,
`tcn_trigger` 엔진 choice 추가). 기존 v1~v6 기록·배포 가중치(`motion_learning/boxing_tcn.pth`)는
건드리지 않음.
