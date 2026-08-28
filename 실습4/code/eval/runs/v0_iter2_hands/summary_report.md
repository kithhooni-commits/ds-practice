# 🥊 Boxing Action Recognition Benchmark Report

**생성 일시**: 2026-08-27 13:56:13  
**평가 대상**: `C:\hong\project-4\iter4\eval\video\benchmark_landmarks.jsonl`  
**허용 오차 윈도우**: `±400ms`

---

## 1. 📊 종합 정량 성적표 (Overall Benchmark Score)

| 핵심 지표 (Metric) | 측정 수치 | 판정 기준 (Target) | 달성 여부 |
| :--- | :---: | :---: | :---: |
| **F1-Score (종합)** | **0.2140** | $\ge 0.850$ | 🟡 REVIEW |
| **Precision (정밀도)** | **0.1198** (29/242) | $\ge 0.800$ | 🟡 REVIEW |
| **Recall (재현율)** | **1.0000** (29/29) | $\ge 0.850$ | 🟢 PASS |
| **종류 분류 정확도 (Kind Acc)** | **41.4%** | $\ge 90.0\%$ | 🟡 REVIEW |
| **비동작/풋워크 오검출 (Non-Action FP)** | **114회** | $\le 3회$ | 🔴 과검출 발생 |
| **평균 타격 지연 시간 (Timing MAE)** | **101.2 ms** | $\le 50.0 ms$ | 🟡 REVIEW |

---

## 2. 📋 90초 프로토콜 구간별 검출 상세 (Phase Breakdown)

| 프로토콜 구간 | 시간대 | 성격 | 검출 횟수 | 상태 |
| :--- | :---: | :---: | :---: | :---: |
| **1. 준비 (Calibration)** | `00~06s` | 휴식/준비 | **17회** | ⚠️ 17회 오검출 |
| **2. 직선 펀치 (Straight)** | `06~19s` | 동작 | **35회** | 🟢 정상 |
| **⏸ 숨고르기 (Rest 1)** | `19~23s` | 휴식/준비 | **11회** | ⚠️ 11회 오검출 |
| **3. 훅 펀치 (Hook)** | `23~35s` | 동작 | **33회** | 🟢 정상 |
| **⏸ 숨고르기 (Rest 2)** | `35~40s` | 휴식/준비 | **13회** | ⚠️ 13회 오검출 |
| **4. 어퍼컷 (Uppercut)** | `40~52s` | 동작 | **33회** | 🟢 정상 |
| **⏸ 숨고르기 (Rest 3)** | `52~57s` | 휴식/준비 | **14회** | ⚠️ 14회 오검출 |
| **5. 풋워크 (Footwork)** | `57~70s` | 휴식/준비 | **35회** | ⚠️ 35회 오검출 |
| **⏸ 숨고르기 (Rest 4)** | `70~75s` | 휴식/준비 | **14회** | ⚠️ 14회 오검출 |
| **6. 실전 콤보 (Combos)** | `75~85s` | 동작 | **27회** | 🟢 정상 |
| **7. 마무리 (Cooldown)** | `85~90s` | 휴식/준비 | **10회** | ⚠️ 10회 오검출 |

---

## 3. 🎯 펀치 종류별 혼동 행렬 (Confusion Matrix)

```text
  • HOOK->HOOK               : 2회
  • HOOK->STRAIGHT           : 5회
  • STRAIGHT->HOOK           : 6회
  • STRAIGHT->STRAIGHT       : 10회
  • UPPERCUT->HOOK           : 1회
  • UPPERCUT->STRAIGHT       : 5회
```

---

## 4. 💡 주요 분석 및 개선 제안 (Actionable Insights)

1. **비동작 구간(풋워크/휴식) 과검출 방지**:
   * 현재 풋워크/휴식 구간에서 총 **114회**의 오검출이 발생했습니다.
   * `punch_core.js`의 `PUNCH_SPEED` 임계값을 `1.6 m/s -> 1.8 m/s`로 상향하거나, 상체 롤링 중 펀치 감도를 억제하는 `TILT_SUPPRESSION`을 강화하면 해결됩니다.
2. **연타 콤보 및 훅 회수 시 중복 검출 방지**:
   * 펀치 회수(Retract) 시 팔꿈치가 빠르게 당겨지는 과정이 잽으로 오인식되는 문제를 방지하기 위해 `PUNCH_EXTEND (0.40)` 가드를 높이는 것을 권장합니다.

---
*Report generated automatically by Antigravity Benchmark Pipeline.*
