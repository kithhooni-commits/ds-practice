# 실습3 — 이미지 생성 모델 개인화, 파이프라인과 지표 검증

SD3.5-medium에 Textual Inversion · DreamBooth-LoRA · DoRA · prior-preservation을
CustomConcept101 10개 컨셉에 적용하고, CLIP T2I/I2I와 DINO로 채점한 뒤 **그 지표 자체를 검증**한 실험.

**웹에서 보기 →** https://kithhooni-commits.github.io/ds-practice/실습3/

## 결론 한 줄

가장 큰 결함은 하이퍼파라미터가 아니라 단어 하나였다. DreamBooth 관행대로 트리거 토큰을
`sks`로 잡았는데 SD3.5의 텍스트 인코더는 이것을 **SKS 소총**으로 읽는다 — 요청하지 않은
소총과 군복이 이미지에 들어왔다. 그런데 이것을 고치자 **DINO가 10컨셉 중 8개에서 오히려
하락**했다. 세 지표를 다 써도 "폭포 앞에 소총이 떠 있다"를 잡아내지 못한다.

## 구조

| 경로 | 내용 |
|---|---|
| `FINDINGS.md` | 전체 분석 — 지표 검증 · ablation 8건 · 정성 관찰 · 한계 |
| `index.html` · `findings.html` | GitHub Pages용 렌더 페이지 |
| `viewer/viewer_*.html` | 컨셉별 생성 이미지 뷰어 (자체 완결 HTML, 서버 불필요) |
| `tables/zwx_table.md` | 트리거 토큰 평가 표 (컨셉 단위 검정 포함) |
| `data/breakdown_*.csv` | 프롬프트 단위 원점수 — T2I · I2I · DINO |
| `src/` | 학습 · 생성 · 평가 · 시각화 스크립트 |
| `prompts/` | 컨셉별 프롬프트 10개 (채점에 필요) |
| `figures/` | 발표에 쓴 비교 그림 |
| `발표_파이프라인.pptx` | 발표 자료 13슬라이드 |

## 웹에서 이어서 작업할 때

이 저장소에는 **코드 · 점수 · 프롬프트**가 들어 있고 **레퍼런스 이미지(420MB) · 학습
가중치 · 생성 이미지**는 빠져 있다. 그래서 되는 것과 안 되는 것이 갈린다.

| 스크립트 | 저장소만으로 | 필요한 것 |
|---|---|---|
| `zwx_table.py` | ✅ | `data/*.csv` — 경로 자동 탐색 |
| `eval_breakdown.py`, `evaluation.py` | ❌ | 생성 이미지 + 레퍼런스 이미지 |
| `make_viewer.py`, `compare_res.py`, `make_deck3.py` | ❌ | 생성 이미지 |
| `run_all.py`, `train_*.py`, `generate.py` | ❌ | 데이터셋 + GPU (Colab) |
| `publish.py` | ❌ | 원본 프로젝트 레이아웃 |

즉 웹에서는 **분석 · 문서 · 표 재생성**이 되고, **재학습과 뷰어 재생성은 로컬/Colab**
몫이다. `zwx_table.py`는 작업 디렉터리·`src/`·상위 폴더를 차례로 뒤져 CSV를 찾으므로
어디서 실행해도 된다.

```bash
python src/zwx_table.py --md tables/zwx_table.md   # 저장소 루트에서
cd src && python zwx_table.py                       # src/ 안에서도 동일
```

## 파이프라인

```
데이터 → 학습 → 생성 → 평가 → 분석
        TI/LoRA    후보 4장    CLIP T2I·I2I   부호검정
        /DoRA      CLIP 최고점  + DINO         뷰어
        /prior     1장 선택
```

`run_all.py`가 전 단계를 오케스트레이션하며, 출력 파일이 이미 있으면 건너뛴다 —
중간에 죽어도 이어서 돌릴 수 있다. (Colab 런타임이 끊겨 결과를 한 번 통째로 잃고
재학습으로 복구한 적이 있다.)

```bash
python run_all.py --no_4bit --variants lora,dora --only actionfigure_2
python eval_breakdown.py --all --generated <생성디렉터리> --dino --out breakdown.csv
python make_viewer.py --source v2=<생성디렉터리> --scores v2=breakdown.csv --split --out view/viewer.html
python publish.py --repo .. --slug 실습3      # 이 폴더를 다시 빌드
```

`--trigger_token`은 기본값이 `zwx`다. `sks`를 쓰지 말 것 — 위의 이유 때문이다.

## 공개본에서 빠진 것

`person_3` 컨셉의 **이미지**는 이 폴더에 없다. 레퍼런스가 실존 인물의 셀피이고 Pages에
올린 것은 누구나 가져갈 수 있으므로 뷰어에서 제외했다. 수치는 모든 표와 CSV에 그대로
들어 있다. 로컬에서 전체를 보려면 `python publish.py --repo .. --slug 실습3 --include-person3`.

학습 가중치와 원본 데이터셋도 용량 때문에 빠져 있다.
