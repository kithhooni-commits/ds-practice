# 데이터 사이언스 실습 프로젝트

로컬 / Colab / Claude Code on the web에서 이어서 작업하기 위한 저장소.
실습 하나당 폴더 하나이고, 각 폴더는 GitHub Pages로도 공개된다.

**웹에서 보기 →** https://kithhooni-commits.github.io/ds-practice/

## 실습

| 폴더 | 내용 |
|---|---|
| [실습3](실습3/) | 이미지 생성 모델 개인화 — SD3.5-medium에 TI/LoRA/DoRA/prior를 적용하고 CLIP·DINO 지표 자체를 검증 |
| [실습4](실습4/) | 웹캠만으로 4인 실시간 AR 복싱 — 상체 7노드 모션 인식과 단안 3D 얼굴 복원, 그리고 조용한 실패들 |
| [실습4_연습](실습4_연습/) | 실습4로 가는 길에 만든 연습 4편 — 제스처 잠금장치 · 비전 드럼 · 비전 복싱(WebRTC 대전) · 다음 단계 타당성 노트 |
| [실습5](실습5/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kithhooni-commits/ds-practice/blob/main/%EC%8B%A4%EC%8A%B55/colab_denoising.ipynb)  · Day2 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kithhooni-commits/ds-practice/blob/main/%EC%8B%A4%EC%8A%B55/colab_deconvolution.ipynb) | 정답 없이 흐린 이미지 되돌리기 — 삼성 DS2 Image Restoration Challenge, dipole 디컨볼루션과 label-free 복원 |

## 구조

- `실습N/`, `실습N-M/` — 실습별 코드·분석·산출물. 각 폴더에 `README.md`, 데모가 있으면 `index.html`(Pages용)도 함께 둔다
- `notebooks/` — Colab/Jupyter 노트북
- `src/` — 실습에 걸쳐 재사용하는 파이썬 스크립트
- `data/` — 로컬 데이터 (git에는 미포함, `.gitkeep`으로 구조만 유지)
- `.nojekyll` — Pages가 Jekyll 처리를 건너뛰게 한다 (한글 경로·언더스코어 파일 보호)
