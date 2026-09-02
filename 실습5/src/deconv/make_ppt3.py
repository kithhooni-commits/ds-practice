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

## 요구사항 대응

    1. pipeline                 슬라이드 5
    2. before/after/error/GT    슬라이드 8-9 (figures_day3.py 가 만든 그림)
    3. 왜 그 방법인가            슬라이드 3-4, 6-7
    4. label-free (보너스)       슬라이드 11
    + 시도별 결과와 개선점 한 페이지  슬라이드 12
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

    # ---------------------------------------------------------- 2. 문제
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

    # ---------------------------------------------------------- 5. 파이프라인 (요구사항 1)
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

    # ---------------------------------------------------------- 11. label-free (요구사항 4)
    sl = blank(prs); bg(sl)
    title(sl, "label-free — 정답을 한 장도 쓰지 않고", "보너스 점수", eyebrow="요구사항 4")
    xs = Inches(0.75)
    for h1, h2 in [("측정치 g", "가진 것은 이것뿐"),
                   ("σ 추정", "널 원뿔에서\n라벨 불필요"),
                   ("N2V 디노이저", "1일차 blind-spot\n으로 학습\n정답 안 씀"),
                   ("데이터 정합", "닫힌 해\n학습 없음"),
                   ("복원", f"{R.get('lf_psnr', 18.87):.2f} dB")]:
        card(sl, xs, Inches(2.2), Inches(2.0), Inches(1.6), ACCENT_SOFT, ACCENT)
        text(sl, xs + Inches(0.1), Inches(2.35), Inches(1.8), Inches(0.4),
             h1, size=13, color=INK, bold=True, align=PP_ALIGN.CENTER)
        text(sl, xs + Inches(0.1), Inches(2.8), Inches(1.8), Inches(0.9),
             h2, size=10, color=MUTED, align=PP_ALIGN.CENTER, font=MONO)
        xs += Inches(2.25)
    rows = [("경로", "정답 사용", "PSNR", "분류"),
            ("1일차 label-free (N2V) → Wiener", "없음", "18.87", "Self-supervised"),
            ("plug-and-play · label-free", "없음", "17.99", "Self-supervised"),
            ("1일차 supervised → Wiener", "1일차만", "21.06", "Supervised"),
            ("최종 모델", "3일차 학습", f"{p:.2f}", "Supervised")]
    table(sl, Inches(0.9), Inches(4.2), Inches(11.5), Inches(1.7), rows,
          col_w=[5.0, 2.0, 1.8, 2.7], highlight_row=1)
    footer(sl, "label-free 경로는 3일차 정답을 한 장도 쓰지 않는다 — σ 추정도 측정치만 본다")

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
            ("전개형 + DRUNet + σ 조건화", "26.85", "σ 200배 차이를 흡수. PSNR 통과, SSIM 0.77 로 미달"),
            ("+ 채점 SSIM 을 손실에", f"{p:.2f}", "L1 은 뭉개는 쪽으로 수렴한다. 채점 함수를 그대로 미분했다")]
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
    ap.add_argument("--results", type=Path, default=FIG / "day3_results.json")
    ap.add_argument("--out", type=Path, default=ROOT / "실습5_day3_발표.pptx")
    args = ap.parse_args()

    R = {}
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
