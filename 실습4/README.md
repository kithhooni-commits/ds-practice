# 실습4 — 웹캠만으로 4인 실시간 AR 복싱, 그리고 조용한 실패들

MediaPipe Pose 상체 7노드로 복싱 모션을 읽고, Face Mesh 468 랜드마크로 **단안 3D 얼굴을 복원**해
아바타에 씌우는 브라우저 게임. 서버는 FastAPI + WebSocket, 렌더는 Three.js.

**웹에서 보기 →** https://kithhooni-commits.github.io/ds-practice/실습4/
**코드 →** https://github.com/gichul-hong/project-4 (`iter3/`)

## 결론 한 줄

가장 많은 시간을 잡아먹은 것은 **틀린 알고리즘이 아니라 아무 말 없이 꺼져 있던 기능들**이었다.
자동 조준은 대상 목록이 비어 있어 통째로 무효였고, 이동은 펀치 잠금에 막혀 6초에 **0 units**를
갔고, 3D 얼굴은 두개골 **안쪽**에 배치돼 어디서도 보이지 않았다. 셋 다 예외도 경고도 없었고,
**로직 하니스는 전부 통과**시켰다. 브라우저로 페이지를 실제로 띄우는 하니스를 만들고 나서야 잡혔다.

## 숫자

| 항목 | 값 | 비고 |
|---|---|---|
| 추적 노드 | 33개 중 **7개** | 코·양어깨·양팔꿈치·양손목 |
| 3D 얼굴 | 468 랜드마크 → **852 삼각형** | 사진 1장, Python·GPU·GLB 없음 |
| 얼굴 삼각형 복원 버그 | 3288 → **852** | 간선 목록이 양방향 중복 |
| 펀치 판정 | 창 380ms · 최고속도 1.6 m/s · 뻗음 어깨폭 0.88배 | 잽 기준으로 역산 |
| 실측 펀치 간격 | 최소 **0.20s** · 중앙값 0.25s | 서버 로그 533회 |
| 이동 잠금 버그 | 6초 연타 중 전진 **0.0 → 39.4 units** | 잠금 하한 480 → 180ms |
| 자동 조준 | 옆(90°) 상대 **MISS → HIT** | 펀치 순간 스냅 |
| 이동 임계 | 사람마다 측정 (최대 기울기의 45%) | 고정값으로는 아무에게도 안 맞았다 |
| 검증 | 하니스 **11종 267개** | 로직 7 + 실제 브라우저 4 |

## 무엇을 만들었나

- **모션 인식** — 몸 기울임으로 전후좌우, 주먹 좌우 쓸기로 회전, 팔 궤적으로 잽/훅/어퍼컷 분류
- **단안 3D 얼굴 복원** — 입장 시 얼굴을 찍으면 그 자리에서 3D 메쉬로 복원해 아바타 머리에 씌운다.
  맞으면 그 자리가 눌리고 멍이 누적되며, HP가 떨어지면 코피가 나고 지친 표정이 된다
- **게임 규칙** — 최후 1인 생존, 분노 게이지(맞을수록 참) → 불꽃 오라 → 필살기
- **음향** — 오디오 파일 없이 Web Audio 실시간 합성. 데미지에 따라 소리가 달라진다

## 구조

| 경로 | 내용 |
|---|---|
| `FINDINGS.md` | 전체 분석 — 조용한 실패 4건, CV 파이프라인, 검증 전략 |
| `index.html` | GitHub Pages용 렌더 페이지 |
| `figures/` | 실행 화면 (host 관제뷰 · 1인칭) |

코드는 이 저장소에 없다. [gichul-hong/project-4](https://github.com/gichul-hong/project-4)의
`iter3/`에 있고, 팀 저장소라 여기서는 분석만 다룬다.

| 코드 위치 | 내용 |
|---|---|
| `iter3/server/static/punch_core.js` | 펀치·필살기 판정 **단일 소스** (브라우저·Node 공용, 의존성 없음) |
| `iter3/server/static/face3d.js` | 웹캠 사진 → 3D 얼굴 복원 + 피격 손상 |
| `iter3/server/static/humanoid.js` | 관절형 복서 아바타 |
| `iter3/tests/` | 하니스 11종 |

## 실행

```bash
cd iter3
python run_arena_server.py          # https://localhost:8000/arena
```

conda 불필요 — 서버는 `fastapi/uvicorn/jinja2/cryptography`만 쓴다.
MediaPipe는 브라우저가 CDN에서 받고, torch는 학습 스크립트 전용이다.

```bash
cd iter3/tests
node pose_harness.js && node effects_harness.js && node aim_harness.js
node move_harness.js && node face_harness.js && node punch_harness.js && node doc_harness.js
node page_harness.js && node match_harness.js                    # 서버 필요
node face_page_harness.js && node avatar_page_harness.js         # 서버 필요
```

## 한계

- **BiLSTM이 런타임에 연결되지 않았다.** 합성 데이터로 정확도 100%를 찍었지만 입력 규격이
  손 랜드마크 63차원이라 현재 Pose 7노드 파이프라인과 맞지 않는다. 실제 판정은 전부 규칙 기반이다.
  팀에서 실제 포즈 데이터셋(`collected_pose/`)을 모으기 시작했으므로 재학습이 다음 단계다.
- **단안 복원의 한계** — Face Mesh는 얼굴 앞면만 덮는다. 옆·뒤 텍스처가 없어 두개골 구로 메운다.
- **조명·배경에 민감** — 어두운 곳에서는 랜드마크 신뢰도가 떨어져 팔 인식이 흔들린다.
