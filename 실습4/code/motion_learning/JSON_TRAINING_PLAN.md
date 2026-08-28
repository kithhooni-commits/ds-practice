# Iter3 JSON 모션 데이터 학습 계획

## 1. 목적

`collected_pose`의 JSON 관절 데이터를 사용하여 다음 세 가지 방식을 같은 데이터 기준으로 비교하고, 최종적으로 게임의 실시간 모션 판정에 적용한다.

1. 휴리스틱 임계값 최적화
2. Bi-LSTM 전체 시퀀스 분류
3. Causal TCN 실시간 슬라이딩 윈도우 분류

최종 목표는 새로운 참가자에게도 동작 정확도를 유지하면서 현재 JavaScript 휴리스틱보다 오검출과 판정 지연을 줄이는 것이다.

## 2. 현재 데이터 현황

기준 경로:

```text
iter3/motion_learning/collected_pose/
```

2026-08-26 확인 결과:

| 항목 | 현재 상태 | 최종 목표 |
|---|---:|---:|
| 참가자 | 4명 | 8명 |
| 참가자별 샘플 | 50개 | 50개 이상 |
| 클래스 | 10개 | 10개 |
| 클래스별 샘플 | 20개 | 40개 이상 |
| 전체 JSON | 200개 | 400개 이상 |
| 스키마 | 버전 4 | 버전 4 유지 |
| 샘플 프레임 | 46~60프레임 | 현재 범위 유지 |
| 평균 추론 FPS | 11.27 | 가능하면 15FPS 이상 |

현재 모든 참가자가 동작별로 5개씩 수집되어 클래스 균형이 맞는다.

학습 클래스:

```text
IDLE
OTHER
LEFT_JAB
RIGHT_JAB
LEFT_HOOK
RIGHT_HOOK
LEFT_UPPERCUT
RIGHT_UPPERCUT
TWO_HAND_GUARD
ENERGY_WAVE
```

## 3. JSON에서 사용하는 정보

원본 정보는 수정하지 않고 전처리 결과와 검수 상태를 별도로 관리한다.

### 공통 원본

- `participant_id`, `session_id`, `label`
- `variant`, `repetition`
- 프레임별 `timestamp_ms`, `relative_time_ms`
- 프레임별 `phase`: `prepare` 또는 `action`
- 프레임별 `causal_target`
- 상체 15관절의 `x, y, z, visibility`
- 품질검사 결과와 실제 추론 FPS

### 휴리스틱용

```text
heuristic_7j_v1: 프레임당 17차원
```

- 좌우 팔꿈치 각도
- 좌우 팔 뻗음 거리
- 손목 xyz 속도와 속력
- 양손 거리
- 손목과 코 사이 거리
- 양쪽 팔꿈치 거리
- 평균 손목 깊이

### Bi-LSTM용

```text
chest_up_15j_temporal_v2: 프레임당 150차원
```

- 15관절 정규화 위치
- 속도
- 가속도
- 관절 가시성
- 액션 구간을 30프레임으로 재표본화

### TCN용

```text
game_7j_temporal_v2: 프레임당 70차원
```

- 게임과 동일한 핵심 7관절
- 위치, 속도, 가속도, 가시성
- 최근 8·10·12프레임의 과거 전용 윈도우
- 윈도우 마지막 프레임의 `causal_target`을 정답으로 사용

## 4. 데이터 무결성 검사

학습 전에 다음 검사를 자동 실행한다.

### 자동 제외 조건

- JSON 파싱 실패
- manifest에 없는 파일 또는 존재하지 않는 경로
- NaN·무한대 좌표
- 특징 차원 불일치
- 프레임 타임스탬프 역전
- 액션 프레임 2개 미만
- 핵심 관절 가시성 기준 미달
- 추론 FPS 8 미만
- 최대 프레임 간격 350ms 초과

자동 제외 파일은 삭제하지 않고 `excluded_hard` 목록으로 격리한다.

### Outlier 검수

클래스별로 다음 값을 계산한다.

- 손목 최고 속도·가속도
- 팔꿈치 최소·최대 각도
- 팔 뻗음 거리
- 동작 방향과 이동 거리
- 양손 거리
- 액션 프레임 수

각 특징의 Median Absolute Deviation 기반 점수가 `3.5`를 넘으면 검수 후보로 지정한다. 추가로 클래스 대표 궤적과의 DTW 거리가 상위 5%인 샘플도 검수한다.

후보는 같은 이름의 `.webm` 영상을 확인하고 다음 상태 중 하나를 부여한다.

```text
accepted      정상
relabelled    다른 정답 라벨로 수정
excluded      학습에서 제외
needs_review  추가 검수 필요
```

불완전한 공격이나 어떤 공격에도 해당하지 않는 동작은 삭제보다 `OTHER` 재라벨링을 우선한다.

## 5. 데이터 분할 원칙

동일 참가자의 데이터가 학습과 평가에 동시에 들어가면 안 된다. 파일 단위 무작위 분할은 사용하지 않는다.

### 현재 4명 데이터

탐색 실험은 Leave-One-Subject-Out 4회 교차검증으로 수행한다.

```text
3명 학습 및 내부 검증
1명 테스트
참가자를 바꾸어 총 4회 반복
```

현재 결과는 모델 구조와 파이프라인 확인용이며 최종 성능으로 사용하지 않는다.

### 최종 8명 데이터

기본 평가는 참가자 단위 8-fold Leave-One-Subject-Out으로 수행한다. 최종 모델 선정 시에는 다음 고정 분할도 함께 보관한다.

```text
학습: 6명
검증: 1명
테스트: 1명
```

최종 테스트 참가자는 임계값 조정과 모델 선택에 사용하지 않는다.

## 6. 학습 데이터 증강

증강은 학습 참가자 데이터에만 적용한다.

### 권장 증강

- 시간 속도 0.8~1.2배 변경
- 프레임 일부 누락
- 작은 좌표 노이즈
- 일부 관절 가시성 감소
- 시작 위치와 카메라 거리 변화
- 액션 앞부분만 포함한 prefix window 생성
- 좌우 반전과 라벨 교환

좌우 반전 시 반드시 다음 라벨을 함께 교환한다.

```text
LEFT_JAB       ↔ RIGHT_JAB
LEFT_HOOK      ↔ RIGHT_HOOK
LEFT_UPPERCUT  ↔ RIGHT_UPPERCUT
```

검증·테스트 데이터에는 증강을 적용하지 않는다.

## 7. 실험 1: 휴리스틱 최적화

### 목적

현재 JavaScript 규칙을 기준선으로 만들고, 사람이 수동으로 정한 임계값을 JSON 데이터에서 최적화한다.

### 입력

```powershell
.\.venv\Scripts\python.exe .\iter4\motion_learning\optimize_heuristic.py
```

### 방법

1. 동작별 각도·속도·거리 분포를 계산한다.
2. 참가자 단위 교차검증에서 임계값 후보를 탐색한다.
3. `OTHER`와 `IDLE` 오검출에 더 큰 비용을 부여한다.
4. 클래스별 임계값과 해제 조건을 별도로 최적화한다.
5. 현재 휴리스틱과 동일한 테스트 참가자에서 비교한다.

### 산출물

```text
artifacts/heuristic_thresholds.json
artifacts/heuristic_metrics.json
```

### Iter4 구현 결과

- 캡처 품질 통과 200개 중 MAD 기반 outlier 17개를 제외하고 원본 183개 사용
- 학습 참가자 데이터에 권장 증강 전체 적용: 183개 → 1,722개
- 적용 항목: 좌우 반전(라벨·`vx` 교환), 시간축 0.8/1.2배, 8% 프레임 누락 후 선형 보간, 특징 공간 소량 노이즈, 짧은 가시성 손실 보간, 카메라 거리 0.92~1.08배, 공격 prefix 0.65/0.82 window
- 4명 Leave-One-Subject-Out 교차검증 적용
- 목적함수: `Macro-F1 - 0.15 × IDLE/OTHER 공격 오검출률`
- 고정 기준선 Macro-F1 `0.2935` → 최적화 휴리스틱 `0.3516`
- `IDLE/OTHER` 공격 오검출률 `0.2286` → `0.2571` (증강 후 오검출률이 상승했으므로 임계값 비용 가중치 재조정이 필요)
- 브라우저 배포본: `server/static/models/heuristic_thresholds.json`
- 게임 컨트롤러에서 `기본 / TCN / 휴리스틱 최적화` 실시간 선택

## 8. 실험 2: Bi-LSTM 기준 모델

### 목적

완성된 액션 시퀀스를 사용하여 시간 정보가 분류 정확도에 주는 효과를 확인한다. Bi-LSTM은 미래 프레임을 사용하는 오프라인 비교 모델로 취급하며 게임의 최종 실시간 모델로 바로 사용하지 않는다.

### 입력

```powershell
.\.venv\Scripts\python.exe .\iter3\motion_learning\pose_dataset.py `
  --feature-set chest_up_15j_temporal_v2 `
  --segment action `
  --sequence-length 30
```

### 초기 모델 후보

```text
입력: 30 × 150
Bi-LSTM: 1~2층
hidden: 64 또는 128
dropout: 0.2~0.3
출력: 10클래스
loss: Cross Entropy
early stopping 기준: 검증 Macro F1
```

현재 합성 데이터용 `boxing_lstm.pth`는 6클래스·63차원 모델이므로 재사용하지 않고 새 모델을 학습한다.

### 산출물

```text
artifacts/bilstm_best.pt
artifacts/bilstm_metrics.json
artifacts/bilstm_confusion_matrix.png
```

## 9. 실험 3: Causal TCN 실시간 모델

### 목적

미래 프레임 없이 최근 관절 움직임만 사용해 게임에서 빠르게 동작을 분류한다.

### 입력

```powershell
.\.venv\Scripts\python.exe .\iter3\motion_learning\pose_dataset.py `
  --mode tcn `
  --feature-set game_7j_temporal_v2 `
  --window-size 10
```

### 비교할 윈도우

```text
8프레임
10프레임
12프레임
```

### 초기 모델 후보

```text
입력: window × 70
Causal Conv1D 채널: 48~64
kernel size: 3
dilation: 1, 2, 4
residual block: 2~3개
dropout: 0.2
출력: 10클래스
```

TCN 윈도우 수는 동작 길이에 따라 클래스별로 달라질 수 있으므로 class weight 또는 균형 샘플러를 사용한다.

### 실시간 이벤트 판정

모델 출력은 바로 공격으로 사용하지 않고 다음 상태 규칙을 적용한다.

```text
확률 0.85 이상 + 충분한 동작 에너지 → 즉시 판정
확률 0.65 이상                    → 2프레임 연속 확인
IDLE 또는 OTHER                   → 공격 이벤트 없음
공격 후 해제 자세 확인            → 다음 공격 허용
```

장풍과 가드는 클래스별 유지시간 조건을 추가한다.

### 산출물

```text
artifacts/tcn_best.pt
artifacts/tcn_model.onnx
artifacts/tcn_metrics.json
artifacts/tcn_confusion_matrix.png
```

## 10. 평가 지표

전체 정확도만으로 모델을 선택하지 않는다.

### 분류 지표

- Macro Precision, Recall, F1
- 클래스별 Recall
- 혼동행렬
- 좌우 동작 혼동률
- `IDLE/OTHER`를 공격으로 분류한 비율

### 실시간 지표

- 실제 동작 시작부터 판정까지 p50·p95 지연
- 분당 잘못 발생한 공격 이벤트 수
- 실제 공격 누락률
- 동일 동작 중복 발생률
- 대상 브라우저에서 추론 p50·p95 시간

### 모델 선택 우선순위

1. 새로운 참가자의 `OTHER/IDLE` 오검출 최소화
2. 펀치별 Recall과 좌우 구분
3. 액션 시작 후 판정 지연
4. 브라우저 추론 성능
5. 전체 정확도

## 11. 게임 적용 순서

1. 현재 휴리스틱 결과를 기준선으로 기록한다.
2. 최적화한 휴리스틱 임계값을 먼저 적용한다.
3. TCN을 ONNX 또는 TensorFlow.js 형식으로 변환한다.
4. Python 전처리와 JavaScript 전처리의 출력이 같은지 비교한다.
5. 게임에서는 `휴리스틱 시작 감지 + TCN 분류` 하이브리드 방식으로 실행한다.
6. 모델 확률, 최종 이벤트, 판정 시간을 로그로 저장한다.
7. 실제 플레이 오검출 영상을 검수 데이터로 다시 추가한다.

권장 실시간 흐름:

```text
MediaPipe Pose
  → 7관절 정규화 위치·속도·가속도·가시성
  → 동작 에너지 검사
  → 최근 8~12프레임 Causal TCN
  → 확률 및 연속 프레임 검사
  → 공격 이벤트 1회 발생
```

## 12. 추가 수집 계획

남은 4명도 동일하게 클래스별 5개 이상을 수집한다. 가능하면 참가자마다 별도 세션을 추가해 다음 변화를 포함한다.

- 다른 조명
- 다른 카메라 거리
- 다른 옷
- 빠른·느린 동작
- 작은·큰 동작
- 왼손잡이 참가자

실제 게임의 오검출률을 평가하려면 참가자별로 2~3분의 연속 `IDLE/OTHER` 플레이 데이터도 별도로 수집한다. 짧은 안내형 클립만으로는 분당 오검출 횟수를 신뢰성 있게 평가하기 어렵다.

## 13. 작업 단계

### 1단계: 데이터 완성

- [x] 4명 × 50개 JSON 수집
- [x] 10개 클래스 균형 확인
- [x] 스키마 버전 4 통일
- [ ] 나머지 4명 데이터 수집
- [ ] 연속 IDLE/OTHER 평가 데이터 수집

### 2단계: 품질검수

- [ ] 무결성 검사 스크립트 작성
- [ ] MAD·DTW outlier 후보 생성
- [ ] 영상 검수 및 재라벨링
- [ ] 데이터셋 버전 고정

### 3단계: 모델 비교

- [x] 휴리스틱 기준선 평가
- [x] 휴리스틱 임계값 최적화
- [ ] 10클래스 Bi-LSTM 학습
- [ ] 8·10·12프레임 TCN 비교
- [ ] 참가자 단위 교차검증

### 4단계: 실시간 적용

- [ ] TCN 모델 브라우저용 변환
- [ ] JavaScript 전처리 일치 검사
- [x] 휴리스틱 최적화 하이브리드 판정 적용
- [ ] 지연·오검출 측정
- [ ] 최종 모델과 임계값 확정

## 14. 재현성 관리

각 실험은 다음 정보를 함께 저장한다.

```text
dataset_version
Git commit
참가자 분할 목록
feature_set
window_size 또는 sequence_length
random_seed
모델 설정
학습 epoch
최고 검증 점수
테스트 지표
모델 파일 해시
```

원본 JSON과 영상은 변경하지 않는다. 제외·재라벨링 결과와 전처리된 데이터셋은 버전이 있는 별도 manifest로 관리한다.
