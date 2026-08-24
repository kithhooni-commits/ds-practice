# 데이터 사이언스 실습 프로젝트

로컬 / Colab / Claude Code on the web에서 이어서 작업하기 위한 저장소.
실습 하나당 폴더 하나이고, 각 폴더는 GitHub Pages로도 공개된다.

**웹에서 보기 →** https://kithhooni-commits.github.io/ds-practice/

## 실습

| 폴더 | 내용 |
|---|---|
| [실습3](실습3/) | 이미지 생성 모델 개인화 — SD3.5-medium에 TI/LoRA/DoRA/prior를 적용하고 CLIP·DINO 지표 자체를 검증 |
| [실습4](실습4/) | 제스처 비밀번호 잠금장치 — MediaPipe Hand Landmarker + 규칙 기반 제스처 분류로 만든 브라우저 인증 데모 |
| [실습5](실습5/) | 비전 드럼 연습기 — 웹캠 손 속도로 가상 드럼 패드를 치고 Web Audio로 합성음·메트로놈 박자 오차를 낸다 |
| [실습6](실습6/) | 비전 복싱(1인용 데모) — Pose Landmarker로 잽·훅·어퍼컷·더킹·위빙을 판정하는 AI 대전 게임 |

## 구조

- `실습N/` — 실습별 코드·분석·산출물. 각 폴더에 `README.md`와 `index.html`(Pages용)
- `notebooks/` — Colab/Jupyter 노트북
- `src/` — 실습에 걸쳐 재사용하는 파이썬 스크립트
- `data/` — 로컬 데이터 (git에는 미포함, `.gitkeep`으로 구조만 유지)
- `.nojekyll` — Pages가 Jekyll 처리를 건너뛰게 한다 (한글 경로·언더스코어 파일 보호)
