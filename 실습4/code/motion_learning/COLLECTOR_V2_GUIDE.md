# 복싱 모션 데이터 수집기 v2

실행:

```powershell
cd C:\project-4-main
.\.venv\Scripts\python.exe .\iter4\motion_learning\collector_v2_app.py
```

브라우저에서 `http://127.0.0.1:8012`를 엽니다. Windows에서는
`iter4\motion_learning\run_collector_v2.bat`를 실행해도 됩니다.

## 사용 순서

1. 참가자 ID와 세션 ID를 입력하고 **카메라 시작**을 누릅니다.
2. **녹화 시작** 후 동작합니다. 이 동안 영상, 상체 15개 관절 프레임, 기존 게임의 `punch_core.js` 판정 이벤트가 함께 기록됩니다.
3. **녹화 종료**를 누르면 검수 화면으로 전환됩니다.
4. 이벤트 행 또는 수정할 라벨을 클릭하면 영상이 그 이벤트 시점으로 이동합니다. 원하는 라벨을 선택합니다.
5. **검수 데이터 저장**을 누릅니다.

## 저장 구조

기본 저장 경로는 다음과 같습니다.

```text
iter4/motion_learning/collected_review_v2/
  {participant_id}/{session_id}/reviewed_recordings/
    {recording_id}.webm
    {recording_id}.json
  manifest.jsonl
```

JSON에는 원본 자동 라벨(`predicted_label`), 검수 라벨(`corrected_label`), 판정 시각, 판정 당시의 측정값, 그리고 상체 15개 관절의 프레임 시퀀스(`pose_frames`)가 들어갑니다. 따라서 휴리스틱 최적화·Bi-LSTM·TCN 학습용 데이터로 재사용할 수 있습니다.

`RIGHT_CROSS`는 기존 게임 런타임의 이름이고, 학습용 데이터 라벨에서는 `RIGHT_JAB`으로 정규화해 저장합니다.
