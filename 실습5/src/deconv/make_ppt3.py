"""3일차 발표 슬라이드 생성.

1일차 `make_ppt.py` 의 헬퍼를 그대로 재사용한다. 수치는 `figures/day3_*.json` 과
`--model` 로 넘긴 최종 결과에서 읽는다.

## 이 발표의 중심

1일차는 "어떻게 더 잘 되는가", 2일차는 "왜 딥러닝이 지는가" 였다.
3일차는 **"왜 1일차 + 2일차를 이어 붙이면 안 되는가"** 다.

    1일차 정답 (디노이저)  37.42 dB
    2일차 정답 (Wiener)   109.86 dB
    둘을 이어 붙이면        21.06 dB   <- 배포 baseline 25.01 에도 진다

각자 최고인 도구인데 합치면 진다. 그 이유가 이 발표의 내용이고, 답은
"선형 역산이 포기한 주파수를 비선형 사전지식으로 메운다" 다.

## 구성 (18장)

결론을 먼저 말하고 근거를 뒤에 둔다. 요약과 목차를 앞에 놓고 파이프라인을 바로 보인다 —
"무엇을 만들었나" 를 알아야 뒤의 근거 슬라이드가 읽힌다.

     1  표지               제출값
     2  요약               문제·발견·한계·답 네 줄
     3  목차
     4  파이프라인          요구사항 1
     5-7 문제와 근거        왜 안 합쳐지나 · 선형의 천장          요구사항 3
     8-9 설계 선택          왜 전개형인가 · σ 를 라벨 없이        요구사항 3
    10-12 복원 결과         before/after/difference/GT · 격자   요구사항 2
    13-14 결과 분석         최종 결과 · σ ablation
    15  label-free         정답을 한 장도 쓰지 않고              요구사항 4 (보너스)
    16  이미지 type 취약점                                   배포 tips
    17  시도별 요약 한 페이지
    18  검증 규칙          test 는 채점에만
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))

from make_ppt import (  # noqa: E402
    ACCENT, ACCENT_SOFT, INK, INK2, MONO, MUTED, RULE, SURF, WARN, WHITE,
    H, W, Inches, MSO_SHAPE, PP_ALIGN, Presentation, Pt,
    arrow, blank, bg, card, footer, pic, table, text, title,
)

FIG = ROOT / "figures"
BASELINE = 25.01          # 배포 End2End U-Net (test)
BASELINE_S = 0.8149


def build(prs, name: str, M: dict, R: dict) -> None:
    """M: 최종 모델 결과. R: run_day3.py 가 낸 학습 없는 조합 결과."""
    p, s = M["psnr"], M["ssim"]
    gain = p - BASELINE

    # ---------------------------------------------------------- (파이프라인 본체)
    # 앞쪽(슬라이드 4)에서 호출한다. 발표에서 "무엇을 만들었나" 를 먼저 보여야
    # 뒤의 근거 슬라이드들이 읽힌다.
    def slide_pipeline():
        sl = blank(prs); bg(sl)
        title(sl, "파이프라인", "전개형 — 데이터 정합과 사전지식을 번갈아 4번",
              eyebrow="요구사항 1")
        xs = Inches(0.75)
        boxes = [("측정치 g", "h*f + n", SURF),
                 ("Wiener 초기화", "x₀ = D·G\n   /(D²+λ₀)", SURF),
                 ("사전지식\nDRUNet", "z = net(x, σ)\nσ 는 측정치에서", ACCENT_SOFT),
                 ("데이터 정합", "x = (D·G+λZ)\n     /(D²+λ)", SURF),
                 ("복원 f̂", f"{p:.2f} dB", ACCENT_SOFT)]
        for i, (h1, h2, fill) in enumerate(boxes):
            card(sl, xs, Inches(2.3), Inches(2.0), Inches(1.6), fill,
                 ACCENT if fill == ACCENT_SOFT else RULE)
            text(sl, xs + Inches(0.1), Inches(2.45), Inches(1.8), Inches(0.5),
                 h1, size=13, color=INK, bold=True, align=PP_ALIGN.CENTER)
            text(sl, xs + Inches(0.1), Inches(2.95), Inches(1.8), Inches(0.8),
                 h2, size=10.5, color=MUTED, align=PP_ALIGN.CENTER, font=MONO)
            xs += Inches(2.25)
            if i < len(boxes) - 1:
                arrow(sl, xs - Inches(0.22), Inches(2.98), Inches(0.2), Inches(0.24))
        # 되돌아가는 화살표
        text(sl, Inches(3.0), Inches(4.05), Inches(6.5), Inches(0.4),
             "↑____________ 4번 반복 ____________|", size=12, color=ACCENT,
             bold=True, align=PP_ALIGN.CENTER, font=MONO)

        card(sl, Inches(0.9), Inches(4.8), Inches(11.5), Inches(1.6), ACCENT_SOFT, ACCENT)
        text(sl, Inches(1.2), Inches(5.0), Inches(11), Inches(1.3),
             [("데이터 정합은 학습 파라미터가 없다. dipole 은 주파수 영역에서 대각 연산자라 닫힌 해로 정확히 풀린다.",
               {"bold": True, "color": INK}),
              ("네트워크는 '아는 주파수를 다시 맞히는' 일을 배울 필요가 없고, 모르는 주파수를 채우는 데만 용량을 쓴다.", {}),
              ("λ 는 단계마다 따로 학습해 '이번엔 측정치를 얼마나 믿을지'를 스스로 정한다.", {})],
             size=13, color=INK2)

    # ---------------------------------------------------------- 1. 표지
    sl = blank(prs); bg(sl)
    band = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.28), H)
    band.fill.solid(); band.fill.fore_color.rgb = ACCENT
    band.line.fill.background(); band.shadow.inherit = False

    text(sl, Inches(1.0), Inches(1.5), Inches(11), Inches(0.4),
         "삼성 DS2 · Image Restoration Challenge · Day 3", size=13, color=ACCENT,
         bold=True, font=MONO)
    text(sl, Inches(1.0), Inches(2.0), Inches(11), Inches(1.2),
         "1일차 + 2일차 = 3일차 가 아니다", size=40, color=INK, bold=True)
    text(sl, Inches(1.0), Inches(2.9), Inches(11), Inches(0.7),
         "선형 역산이 포기한 곳을 메우는 법", size=40, color=ACCENT, bold=True)
    text(sl, Inches(1.0), Inches(4.0), Inches(10), Inches(0.5),
         "g = dipole(f) + n  ·  test_deconv_noise 100장", size=15, color=MUTED)

    card(sl, Inches(1.0), Inches(4.8), Inches(4.4), Inches(1.5), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.25), Inches(5.0), Inches(4.0), Inches(0.35),
         "제출값", size=12, color=ACCENT, bold=True, font=MONO)
    text(sl, Inches(1.25), Inches(5.35), Inches(4.0), Inches(0.7),
         f"{p:.2f} dB / {s:.4f}", size=28, color=INK, bold=True)

    card(sl, Inches(5.7), Inches(4.8), Inches(4.4), Inches(1.5))
    text(sl, Inches(5.95), Inches(5.0), Inches(4.0), Inches(0.35),
         "배포 baseline 대비", size=12, color=MUTED, bold=True, font=MONO)
    text(sl, Inches(5.95), Inches(5.35), Inches(4.0), Inches(0.7),
         f"{gain:+.2f} dB", size=28, color=ACCENT if gain > 0 else WARN, bold=True)
    if name:
        footer(sl, name)

    # ---------------------------------------------------------- 2. 요약
    # 결론을 먼저 말한다. 뒤 슬라이드는 전부 이 세 줄의 근거다.
    sl = blank(prs); bg(sl)
    title(sl, "요약", "한 장으로", eyebrow="Summary")
    ys = Inches(1.7)
    for tag, head, body in [
        ("문제", "흐림과 노이즈가 겹쳤다  g = h * f + n",
         "1일차는 노이즈만, 2일차는 흐림만이었다. 3일차는 둘이 겹쳐 각자의 답이 통하지 않는다."),
        ("발견", "각자 최고인 도구를 이어 붙이면 진다 — 21.06 dB",
         "1일차 디노이저(37.42)와 2일차 Wiener(109.86)를 이어도 배포 baseline(25.01)에 진다. "
         "역산이 노이즈를 +51.5 dB 증폭하기 때문이다."),
        ("한계", "선형 방법의 천장은 19.80 dB",
         "정답을 알고 만든 최고의 선형 필터(오라클 위너)조차 그렇다. 그 위는 비선형 사전지식의 몫이다."),
        ("답", "전개형 — 물리 제약과 학습된 사전지식을 번갈아",
         "역산은 닫힌 해로 정확히 풀고, 네트워크는 역산이 포기한 주파수를 메우는 데만 용량을 쓴다. "
         "σ 는 측정치에서 읽어 장마다 다른 노이즈 세기에 맞춘다."),
    ]:
        card(sl, Inches(0.9), ys, Inches(11.5), Inches(1.05))
        text(sl, Inches(1.15), ys + Inches(0.28), Inches(1.1), Inches(0.4),
             tag, size=12, color=ACCENT, bold=True, font=MONO)
        text(sl, Inches(2.3), ys + Inches(0.12), Inches(9.9), Inches(0.35),
             head, size=14, color=INK, bold=True)
        text(sl, Inches(2.3), ys + Inches(0.5), Inches(9.9), Inches(0.5),
             body, size=11.5, color=MUTED)
        ys += Inches(1.15)

    card(sl, Inches(0.9), Inches(6.05), Inches(11.5), Inches(0.85), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.15), Inches(6.22), Inches(11), Inches(0.5),
         f"결과   {p:.2f} dB / {s:.4f}   ·   배포 baseline {BASELINE:.2f} / {BASELINE_S:.4f} 대비 "
         f"{gain:+.2f} dB · {s - BASELINE_S:+.4f}   ·   통과 기준 26 / 0.83",
         size=13, color=INK, bold=True)

    # ---------------------------------------------------------- 3. 목차
    sl = blank(prs); bg(sl)
    title(sl, "목차", None, eyebrow="Contents")
    items = [("1", "파이프라인", "무엇을 만들었나", "요구사항 1"),
             ("2", "문제와 근거", "왜 1일차 + 2일차가 안 되는가 · 선형의 천장", "요구사항 3"),
             ("3", "설계 선택", "왜 전개형인가 · σ 를 라벨 없이 읽는 법", "요구사항 3"),
             ("4", "복원 결과", "before / after / difference / GT · 노이즈별 격자", "요구사항 2"),
             ("5", "결과 분석", "어떤 노이즈·어떤 이미지에 취약한가 · σ ablation", "tips"),
             ("6", "label-free", "정답을 한 장도 쓰지 않고 푼다", "요구사항 4 · 보너스"),
             ("7", "시도별 요약", "무엇을 해봤고 무엇을 배웠나 · 검증 규칙", "")]
    ys = Inches(1.75)
    for num, head, body, req in items:
        card(sl, Inches(0.9), ys, Inches(11.5), Inches(0.62))
        text(sl, Inches(1.15), ys + Inches(0.13), Inches(0.5), Inches(0.4),
             num, size=15, color=ACCENT, bold=True, font=MONO)
        text(sl, Inches(1.8), ys + Inches(0.14), Inches(3.0), Inches(0.4),
             head, size=13.5, color=INK, bold=True)
        text(sl, Inches(4.9), ys + Inches(0.17), Inches(5.4), Inches(0.4),
             body, size=11.5, color=MUTED)
        if req:
            text(sl, Inches(10.3), ys + Inches(0.17), Inches(1.9), Inches(0.4),
                 req, size=10.5, color=ACCENT, font=MONO, align=PP_ALIGN.RIGHT)
        ys += Inches(0.72)

    # ---------------------------------------------------------- 4. 파이프라인 (요구사항 1)
    slide_pipeline()

    # ---------------------------------------------------------- 5. 문제
    sl = blank(prs); bg(sl)
    title(sl, "문제 — 두 열화가 겹쳤다", "g = h * f + n. 노이즈가 흐림 **뒤에** 붙는다",
          eyebrow="Day 3")
    rows = [("", "1일차", "2일차", "3일차"),
            ("문제", "g = f + n", "g = h * f", "g = h * f + n"),
            ("우리 답", "DRUNet", "Wiener K→0", "?"),
            ("점수", "37.42 dB", "109.86 dB", "—")]
    table(sl, Inches(0.9), Inches(1.9), Inches(11.5), Inches(1.4), rows,
          col_w=[1.2, 2, 2, 2.4], highlight_col=3)
    card(sl, Inches(0.9), Inches(3.6), Inches(11.5), Inches(2.6), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.2), Inches(3.85), Inches(11), Inches(0.4),
         "순서가 중요하다", size=15, color=ACCENT, bold=True)
    text(sl, Inches(1.2), Inches(4.3), Inches(11), Inches(1.7),
         [("노이즈가 흐림 뒤에 붙었으므로 측정치 위에서는 백색이다. 그래서 역산 전에 지워야 한다.", {}),
          ("", {}),
          ("전처리 → Wiener   21.06 dB          Wiener → 전처리   16.88 dB",
           {"bold": True, "color": INK, "font": MONO}),
          ("4.2 dB 차이. 역산이 노이즈를 유색으로 만들면 디노이저가 못 알아본다.", {})],
         size=14, color=INK2)
    footer(sl, "노이즈 4종·σ 는 1일차와 파일별로 동일하다 (100/100 일치)")

    # ---------------------------------------------------------- 3. 왜 안 합쳐지나
    sl = blank(prs); bg(sl)
    title(sl, "왜 1일차 + 2일차가 안 되는가", "숫자로 확인한 세 가지",
          eyebrow="근거")
    ys = Inches(1.85)
    for head, body, val in [
        ("2일차 답은 노이즈가 0이라서 통했다",
         "답이 1/D 로 나누는 것이었다. D 는 매직앵글 원뿔에서 0에 가까워지니\n"
         "거의 무한대를 곱하는 셈인데, n=0 이면 무한대 × 0 = 0 이라 괜찮았다.", "+51.5 dB\n노이즈 증폭"),
        ("흐림이 대비를 죽여 같은 σ 가 훨씬 어려워진다",
         "std(f)=0.222 → std(h*f)=0.092 (0.41배). 노이즈는 1일차와 똑같은데\n"
         "신호만 줄었다. 1일차 디노이저가 본 적 없는 난이도다.", "+8.6 dB\n→ +0.9 dB"),
        ("주파수의 16% 는 진짜로 복원 불가다",
         "|D|<0.1 구간은 역산 후 SNR 이 −20 dB 이하다. 2일차엔 이 부분도\n"
         "되살렸지만(노이즈가 0이니까) 3일차엔 못 한다.", "16%\n버려진다"),
    ]:
        card(sl, Inches(0.9), ys, Inches(11.5), Inches(1.35))
        text(sl, Inches(1.15), ys + Inches(0.12), Inches(7.6), Inches(0.35),
             head, size=14, color=INK, bold=True)
        text(sl, Inches(1.15), ys + Inches(0.5), Inches(7.6), Inches(0.75),
             body, size=11.5, color=MUTED)
        text(sl, Inches(9.0), ys + Inches(0.28), Inches(3.1), Inches(0.8),
             val, size=15, color=ACCENT, bold=True, font=MONO, align=PP_ALIGN.CENTER)
        ys += Inches(1.5)
    footer(sl, "재현: src/deconv/day3_common.py 의 load_val 로 직접 잰 값")

    # ---------------------------------------------------------- 4. 선형의 천장
    sl = blank(prs); bg(sl)
    title(sl, "선형 방법의 천장은 19.80 dB", "그 위는 전부 비선형 사전지식의 몫이다",
          eyebrow="왜 딥러닝인가")
    rows = [("방법", "PSNR", "성격"),
            ("Wiener 단독", "13.79", "선형"),
            ("median → Wiener", "19.20", "거의 선형"),
            ("오라클 위너 (주파수별 진짜 SNR 을 안다고 가정)", "19.80", "최고의 선형 필터"),
            ("1일차 디노이저 → Wiener", "21.06", "비선형이 한 번 들어감"),
            ("배포 baseline (End2End U-Net)", f"{BASELINE:.2f}", "학습된 비선형"),
            ("우리 최종", f"{p:.2f}", "학습된 비선형 + 물리")]
    table(sl, Inches(0.9), Inches(1.9), Inches(11.5), Inches(2.6), rows,
          col_w=[5.5, 1.6, 3.2], highlight_row=len(rows) - 1)
    card(sl, Inches(0.9), Inches(4.8), Inches(11.5), Inches(1.5), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.2), Inches(5.0), Inches(11), Inches(1.2),
         [("정답을 알고 만든 최고의 선형 필터조차 19.80 dB 다. 선형으로는 여기서 끝난다.",
           {"bold": True, "color": INK}),
          ("3일차는 도구를 조립하는 문제가 아니라, 역산이 포기한 원뿔 안을", {}),
          ("이미지 사전지식으로 메우는 문제다.", {})],
         size=14, color=INK2)

    # ---------------------------------------------------------- 6. 왜 이 구조인가 (요구사항 3)
    # ---------------------------------------------------------- 6. 왜 이 구조인가 (요구사항 3)
    sl = blank(prs); bg(sl)
    title(sl, "왜 전개형인가 — DC-Net 이 죽은 자리에서", "2일차 최고 구조가 3일차에서 실패한 이유",
          eyebrow="요구사항 3")
    text(sl, Inches(0.9), Inches(1.75), Inches(11.5), Inches(0.5),
         "hard DC 는 |D|>τ 인 주파수를 G/D 로 못박는다. 그런데 G = D·F + N 이므로", size=14, color=INK2)
    card(sl, Inches(2.5), Inches(2.3), Inches(8.3), Inches(0.75), ACCENT_SOFT, ACCENT)
    text(sl, Inches(2.6), Inches(2.45), Inches(8.1), Inches(0.5),
         "G/D  =  F  +  N/D        ← 노이즈가 1/|D| 배 증폭된 채 고정된다",
         size=15, color=INK, bold=True, font=MONO, align=PP_ALIGN.CENTER)
    rows = [("τ", "못박는 주파수", "경계 노이즈 증폭", "결과"),
            ("0.005 (2일차 최적 방향)", "99.2%", "200배", "폭발"),
            ("0.05", "92.0%", "20배", "14.80 dB"),
            ("0.2", "66.4%", "5배", "DC 가 무의미")]
    table(sl, Inches(0.9), Inches(3.35), Inches(11.5), Inches(1.4), rows,
          col_w=[3.4, 2.4, 2.6, 2.4])
    card(sl, Inches(0.9), Inches(5.05), Inches(11.5), Inches(1.3), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.2), Inches(5.25), Inches(11), Inches(1.0),
         [("τ 를 키우면 DC 가 무의미해지고 줄이면 폭발한다 — 3일차엔 좋은 τ 가 없다.",
           {"bold": True, "color": INK}),
          ("고칠 방법은 못박지 말고 무게를 두는 것. 그게 soft DC 이고,", {}),
          ("전개형이 곧 DC-Net 의 노이즈 대응 일반형이다 (λ→0 이면 DC-Net 이 된다).", {})],
         size=13.5, color=INK2)

    # ---------------------------------------------------------- 7. σ 조건화
    sl = blank(prs); bg(sl)
    title(sl, "σ 를 라벨 없이 읽어 모델에 알려준다", "3일차 σ 는 이미지마다 200배 차이난다",
          eyebrow="요구사항 3")
    text(sl, Inches(0.9), Inches(1.75), Inches(11.5), Inches(1.0),
         [("|D| < 0.02 인 주파수엔 신호가 실려올 수 없다 — dipole 이 그리로 아무것도 보내지 않는다.", {}),
          ("거기 남은 것은 전부 노이즈다. 파세발로 E|G|² = N·σ².", {}),
          ("정답도 noise_meta.json 도 쓰지 않는다.", {"bold": True, "color": INK})],
         size=14, color=INK2)
    rows = [("", "값"),
            ("σ 범위 (test 100장)", "0.0007 ~ 0.1325  (200배)"),
            ("추정 상대오차 중앙값", "1.9%"),
            ("σ<0.05 구간 성능", "23.91 dB"),
            ("σ≥0.10 구간 성능", "17.64 dB  (6.3 dB 차이)")]
    table(sl, Inches(0.9), Inches(3.0), Inches(6.4), Inches(1.7), rows, col_w=[3.2, 3.2])
    card(sl, Inches(7.6), Inches(3.0), Inches(4.8), Inches(1.7), ACCENT_SOFT, ACCENT)
    text(sl, Inches(7.85), Inches(3.2), Inches(4.3), Inches(1.4),
         [("blind 모델 하나로 200배 범위를 덮으려면", {}),
          ("평균에 타협해야 한다.", {"bold": True, "color": INK}),
          ("", {}),
          ("σ 를 알려주면 매 장에 맞는 세기로 지운다.", {})],
         size=13, color=INK2)
    card(sl, Inches(0.9), Inches(5.0), Inches(11.5), Inches(1.2), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.2), Inches(5.2), Inches(11), Inches(0.9),
         [("σ 는 이미지마다 200배 차이나는데, 그 값을 아는 것만으로 지울 세기를 매 장에 맞출 수 있다.", {}),
          ("커널을 알기에 쓸 수 있는 추정기다 — MAD 와 달리 이미지의 고주파를 노이즈로 착각하지 않는다.",
           {"bold": True, "color": INK})],
         size=13, color=INK2)

    # ---------------------------------------------------------- 8. 4장 비교 (요구사항 2)
    sl = blank(prs); bg(sl)
    title(sl, "before / after / difference / GT", "σ 중앙값에 가까운 대표 이미지 — 잘 나온 걸 고르지 않았다",
          eyebrow="요구사항 2")
    f = FIG / "day3_diff_zoom.png"
    if f.exists():
        pic(sl, f, Inches(1.3), Inches(1.6), w=Inches(10.7))
    footer(sl, "위: 복원 · 가운데: |복원−정답| (열 공통 스케일) · 아래: 72×72 확대")

    # ---------------------------------------------------------- 9. 노이즈별 격자
    sl = blank(prs); bg(sl)
    title(sl, "노이즈 4종 × 방법별 복원 결과", None, eyebrow="요구사항 2")
    f = FIG / "day3_methods_grid.png"
    if f.exists():
        pic(sl, f, Inches(1.9), Inches(1.5), h=Inches(5.4))
    footer(sl, "각 칸 왼쪽 아래가 그 장의 PSNR / SSIM")

    # ---------------------------------------------------------- 10. 취약점 분석
    sl = blank(prs); bg(sl)
    title(sl, "어디에 취약한가", "배포 안내가 요구한 분석", eyebrow="결과 분석")
    f = FIG / "day3_weakness.png"
    if f.exists():
        pic(sl, f, Inches(0.8), Inches(1.5), w=Inches(11.7))
    ys = Inches(4.5)
    for head, body in [
        ("Rician 이 가장 약하다",
         "정류 편향이 DC 를 밀어 올리는데 dipole 은 DC 를 1/3 로 보존하므로 편향이 그대로 살아남는다."),
        ("salt & pepper 에서만 median 이 딥러닝을 이긴다 (23.23 vs 19.55)",
         "1일차 디노이저는 선명한 입력 위의 임펄스만 봤지, 흐릿한 입력 위의 임펄스는 본 적이 없다."),
        ("X자 무늬는 매직앵글 영널 원뿔이다",
         "정보가 애초에 없는 주파수라 어떤 K 를 써도 남는다. 사전지식으로만 메울 수 있다."),
    ]:
        text(sl, Inches(0.9), ys, Inches(11.5), Inches(0.3), head, size=12.5,
             color=INK, bold=True)
        text(sl, Inches(0.9), ys + Inches(0.28), Inches(11.5), Inches(0.3), body,
             size=11, color=MUTED)
        ys += Inches(0.66)

    # ---------------------------------------------------------- 10b. 최종 결과
    sl = blank(prs); bg(sl)
    title(sl, "최종 결과", "test_deconv_noise 100장 · 4× self-ensemble", eyebrow="제출값")
    rows = [("", "PSNR", "SSIM", "판정"),
            ("통과 기준", "26", "0.83", "—"),
            ("입력 (blur + noise)", "8.02", "−0.0187", "출발점"),
            ("배포 baseline (End2End U-Net)", "25.01", "0.8149", "미달"),
            ("다른 조 (2-step + 4× SE)", "26.73", "0.8215", "SSIM 미달"),
            ("모델 융합 (셋을 평균)", "29.66", "0.8824", "단독보다 못함"),
            ("우리 (전개형 + σ + 4× SE)", f"{p:.2f}", f"{s:.4f}", "통과")]
    table(sl, Inches(0.9), Inches(1.85), Inches(11.5), Inches(2.2), rows,
          col_w=[5.2, 2.0, 2.0, 2.3], highlight_row=len(rows) - 1)

    nz = R.get("per_noise", {})
    rows = [("노이즈", "PSNR", "SSIM", "n")] + [
        (k, f"{v[0]:.2f}", f"{v[1]:.4f}", "25") for k, v in nz.items()]
    table(sl, Inches(0.9), Inches(4.35), Inches(6.0), Inches(1.7), rows,
          col_w=[2.6, 1.3, 1.3, 0.8])
    card(sl, Inches(7.3), Inches(4.35), Inches(5.1), Inches(1.7), ACCENT_SOFT, ACCENT)
    text(sl, Inches(7.55), Inches(4.55), Inches(4.6), Inches(1.4),
         [("rician 만 22.47 로 7 dB 뒤진다.", {"bold": True, "color": INK}),
          ("정류가 밝기를 +0.0396 밀어 올리는데", {}),
          ("dipole 은 DC 를 1/3 로 보존하므로", {}),
          ("역산에서 3배가 되어 0.119 로 남는다.", {})], size=12, color=INK2)
    footer(sl, f"배포 baseline 대비 {gain:+.2f} dB · {s - BASELINE_S:+.4f}   ·   "
               f"다른 조 대비 {p - 26.73:+.2f} dB · {s - 0.8215:+.4f}")

    # ---------------------------------------------------------- 10c. σ ablation
    sl = blank(prs); bg(sl)
    title(sl, "σ 조건화가 성능의 절반을 책임진다", "가중치는 그대로 두고 σ 입력만 바꿔 잰다 — 학습이 필요 없는 ablation",
          eyebrow="근거")
    ab = R.get("ablation", [])
    rows = [("σ 를 어떻게 주는가", "PSNR", "SSIM", "손실")] + [
        (k, f"{v[0]:.2f}", f"{v[1]:.4f}",
         "—" if abs(v[0] - p) < 1e-6 else f"{v[0] - p:+.2f} dB") for k, v in ab]
    table(sl, Inches(0.9), Inches(1.9), Inches(11.5), Inches(1.9), rows,
          col_w=[5.0, 1.8, 1.8, 2.4], highlight_row=1)
    card(sl, Inches(0.9), Inches(4.2), Inches(11.5), Inches(2.0), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.2), Inches(4.45), Inches(11), Inches(1.7),
         [("장마다 다른 σ 를 주는 것이, 하나의 값으로 뭉뚱그리는 것보다 4.22 dB 낫다.",
           {"bold": True, "color": INK}),
          ("", {}),
          ("σ 는 |D| < 0.02 인 주파수에서 읽는다. 거기엔 dipole 이 신호를 보내지 않으므로 "
           "남은 것은 전부 노이즈다.", {}),
          ("정답도 noise_meta.json 도 쓰지 않는다 — 측정치 하나에서 나온다. "
           "3일차 σ 는 장마다 200배 차이난다.", {})], size=13, color=INK2)

    # ---------------------------------------------------------- 11. label-free (요구사항 4)
    sl = blank(prs); bg(sl)
    title(sl, "label-free — 정답을 한 장도 쓰지 않고", "보너스 점수", eyebrow="요구사항 4")
    text(sl, Inches(0.9), Inches(1.7), Inches(11.5), Inches(0.9),
         [("노이즈가 흐림 **뒤에** 붙어 측정치 위에서는 백색이다. 백색이면 Noise2Void 가 그대로 통한다.",
           {"bold": True, "color": INK}),
          ("g 의 일부 화소를 이웃으로 덮고 덮은 자리에서 g 를 맞히게 한다. 가린 화소를 못 보니 "
           "이웃에서 신호를 추정할 수밖에 없고,", {}),
          ("잡음은 화소마다 독립이라 예측할 수 없으므로 최적해가 곧 노이즈 없는 h*f 다.", {})],
         size=13, color=INK2)
    xs = Inches(0.75)
    for h1, h2 in [("측정치 g", "가진 것은\n이것뿐"),
                   ("N2V 디노이저", "가린 자리에서\ng 를 맞힌다\n정답 없음"),
                   ("σ 추정", "널 원뿔에서\n라벨 불필요"),
                   ("역필터", "λ 도 측정치에서\n학습 없음"),
                   ("복원", (f"{R['lf_psnr']:.2f} dB" if "lf_psnr" in R else "학습 중"))]:
        card(sl, xs, Inches(3.0), Inches(2.0), Inches(1.5), ACCENT_SOFT, ACCENT)
        text(sl, xs + Inches(0.1), Inches(3.13), Inches(1.8), Inches(0.4),
             h1, size=13, color=INK, bold=True, align=PP_ALIGN.CENTER)
        text(sl, xs + Inches(0.1), Inches(3.55), Inches(1.8), Inches(0.85),
             h2, size=10, color=MUTED, align=PP_ALIGN.CENTER, font=MONO)
        xs += Inches(2.25)
    card(sl, Inches(0.9), Inches(4.8), Inches(11.5), Inches(1.5), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.2), Inches(5.0), Inches(11), Inches(1.2),
         [("λ 도 라벨 없이 정한다.", {"bold": True, "color": ACCENT}),
          ("고전 Wiener 의 K = 잡음파워/신호파워 를 측정치에서 계산한다 — 잡음파워는 널 원뿔의 σ², "
           "신호파워는 측정치 파워에서 잡음 몫을 뺀 것.", {}),
          ("val 에서 고른 K 대비 −0.58 dB. 정답을 한 번도 쓰지 않고 그만큼 따라간다.",
           {"bold": True, "color": INK})], size=13, color=INK2)
    footer(sl, "학습 측정치는 train 의 clean 에서 합성하되 clean 은 손실에 한 번도 들어가지 않는다 · test 는 건드리지 않는다")

    # ---------------------------------------------------------- 11b. 이미지 type 취약점
    sl = blank(prs); bg(sl)
    title(sl, "어떤 type 의 이미지에 취약한가", "노이즈 라벨은 있지만 이미지 라벨은 없다 — 이미지 자체 특징으로 가른다",
          eyebrow="결과 분석")
    f = FIG / "day3_image_types.png"
    if f.exists():
        pic(sl, f, Inches(0.7), Inches(1.6), w=Inches(11.9))
    rows = [("특징", "하위 1/3 → 상위 1/3", "상관계수", "해석"),
            ("대비 (std)", "−3.59 dB", "−0.459", "같은 σ 라도 구조가 많으면 잃을 것이 많다"),
            ("에지 밀도", "−3.52 dB", "−0.405", "평탄하면 사전지식이 강하게 작동한다"),
            ("동적범위", "−3.31 dB", "−0.310", "위와 같은 이유"),
            ("무늬 조밀도", "+0.92 dB", "+0.053", "예상과 달리 거의 무관했다")]
    table(sl, Inches(0.9), Inches(4.6), Inches(11.5), Inches(1.6), rows,
          col_w=[2.2, 2.4, 1.6, 5.3])
    footer(sl, "가장 못 살린 10장: σ 0.113(전체 0.074) · 대비 0.263(0.158) · 에지 0.083(0.050) · rician 5장")

    # ---------------------------------------------------------- 12. 시도별 요약 (한 페이지)
    sl = blank(prs); bg(sl)
    title(sl, "시도별 결과와 개선점", "한 페이지 요약", eyebrow="전체")
    rows = [("시도", "결과", "왜 그랬나 / 무엇을 배웠나"),
            ("2일차 답(K→0)을 그대로", "−24 dB", "노이즈를 +51.5 dB 증폭. n=0 이라 통했던 답이다"),
            ("Wiener 단독 (K 는 val 에서)", "13.79", "선형 역산의 한계. 정보를 버려야 노이즈가 잡힌다"),
            ("DC-Net (2일차 최고 구조)", "14.80", "hard DC 가 G/D=F+N/D 를 못박는다. 좋은 τ 가 없다"),
            ("Wiener → 디노이저", "16.88", "순서가 반대. 역산이 노이즈를 유색으로 만든다"),
            ("median → Wiener", "19.20", "s&p 에서만 딥러닝을 이긴다 (23.23)"),
            ("1일차 디노이저 → Wiener", "21.06", "오라클 선형(19.80)을 넘었다. 비선형이 필요하다는 증거"),
            ("전개형 (unet f32, blind)", "25.91", "ep37 정체. 용량과 조건화가 병목이지 학습량이 아니다"),
            ("End2End U-Net f64", "25.99", "전개형과 비슷 — 물리 구조만으로는 이득이 없었다"),
            ("2단 분해 (측정치 영역 → 역필터)", "15.6", "실패. 역필터가 오차를 1/D 로 증폭해 40 dB 디노이징을 요구한다"),
            ("σ 에 왜도·첨도를 더해 조건화", "29.59", "효과 없음. 첨도가 rician 을 가르지만 모델이 쓰지 않았다 (−0.03 dB)"),
            ("SSIM 을 처음부터 손실에", "17.5", "실패. 덜 학습된 모델을 '맞든 아니든 대비를 키우는' 쪽으로 민다"),
            ("언샤프 후처리로 SSIM 보정", "—", "실패. 잔차까지 키워 σx 만 커진다. 잃은 대비는 사후에 못 만든다"),
            ("end-to-end DRUNet", "24.6", "전개형보다 3.8 dB 아래. 물리 구조가 실제로 이득이다"),
            ("모델 융합 (서로 다른 셋을 평균)", "29.66", "실패. 같은 체크포인트의 형제라 틀리는 방식이 같다"),
            ("추론 때 반복 횟수 늘리기", "20.99", "실패. 4단계로 학습한 모델은 5단계에서 분포를 벗어난다"),
            ("전개형 + DRUNet + σ 조건화", "29.25", "σ 200배 차이를 흡수. 두 기준 모두 통과"),
            ("+ 수렴할 때까지 더 학습", f"{p:.2f}", "60 에폭이 모자랐다. ep56 까지 계속 오르고 있었다")]
    table(sl, Inches(0.6), Inches(1.7), Inches(12.1), Inches(4.6), rows,
          col_w=[3.6, 1.5, 7.0], highlight_row=len(rows) - 1, size=11)
    footer(sl, "K·λ 는 전부 validation 에서 골랐다. test 는 채점에만 썼다")

    # ---------------------------------------------------------- 13. 규칙
    sl = blank(prs); bg(sl)
    title(sl, "test set 은 채점에만 썼다", "학습이 아니어도 하이퍼파라미터를 test 로 고르면 test 를 쓴 것이다",
          eyebrow="검증 규칙")
    ys = Inches(2.0)
    for head, body in [
        ("튜닝은 val 에서만",
         "val clean 에 forward + 파일명 seed 노이즈를 걸어 (측정치, 정답) 쌍을 만든다.\n"
         "K, λ, 반복 횟수를 전부 여기서 고른다. day3_common.load_val()"),
        ("test 는 report() 한 번",
         "배포된 test_deconv_noise 는 최종 점수를 낼 때만 읽는다. day3_common.load_test()"),
        ("noise_meta.json 은 결과 분석에만",
         "배포 안내 그대로. 표를 노이즈 종류별로 쪼개는 데만 쓰고 복원에는 쓰지 않는다."),
        ("직접 찾아 고친 위반",
         "초기 스크립트가 K 를 test 에서 스윕하고 있었다. combine_day3.py 는 폐기하고\n"
         "pnp.py·eval_day3.py 의 스윕을 val 로 옮겼다."),
    ]:
        card(sl, Inches(0.9), ys, Inches(11.5), Inches(1.05))
        text(sl, Inches(1.15), ys + Inches(0.1), Inches(11), Inches(0.3),
             head, size=13.5, color=ACCENT, bold=True)
        text(sl, Inches(1.15), ys + Inches(0.42), Inches(11), Inches(0.55),
             body, size=11.5, color=MUTED)
        ys += Inches(1.2)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--psnr", type=float, required=True, help="최종 모델 test PSNR")
    ap.add_argument("--ssim", type=float, required=True)
    ap.add_argument("--lf-psnr", type=float, default=None,
                    help="label-free 최종 test PSNR (train_lf_day3.py 결과)")
    ap.add_argument("--lf-ssim", type=float, default=None)
    ap.add_argument("--results", type=Path, default=FIG / "day3_results.json")
    ap.add_argument("--out", type=Path, default=ROOT / "실습5_day3_발표.pptx")
    args = ap.parse_args()

    # eval_day3.py 가 test 100장에서 실제로 낸 값. 다시 뽑을 때는 여기만 고치면 된다.
    R = {
        "per_noise": {"gaussian": (30.15, 0.8843), "rician": (22.87, 0.7558),
                      "uniform": (29.51, 0.9084), "salt_and_pepper": (36.27, 0.9825)},
        # stats3 체크포인트로 잰 값 (test 100장). 가중치는 그대로 두고 σ 입력만 바꿨다.
        "ablation": [("추정 σ (장마다 다르게)", (29.59, 0.8817)),
                     ("전체 평균 하나로 고정", (25.37, 0.7751)),
                     ("σ 2배 (과대평가)", (22.52, 0.7890)),
                     ("σ 절반 (과소평가)", (22.92, 0.6484)),
                     ("전부 0 (노이즈가 없다고)", (15.93, 0.4040))],
    }
    if args.lf_psnr is not None:
        R["lf_psnr"] = args.lf_psnr
        R["lf_ssim"] = args.lf_ssim
    if args.results.exists():
        raw = json.loads(args.results.read_text(encoding="utf-8"))
        for k, v in raw.items():
            if isinstance(v, dict) and "label-free" in k and "Wiener" in k:
                R["lf_psnr"] = v["psnr"]

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    build(prs, args.name, {"psnr": args.psnr, "ssim": args.ssim}, R)
    prs.save(str(args.out))
    print(f"슬라이드 {len(prs.slides._sldIdLst)}장 → {args.out}")


if __name__ == "__main__":
    main()
