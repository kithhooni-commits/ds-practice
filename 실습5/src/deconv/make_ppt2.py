"""2일차 발표 슬라이드 생성.

1일차 `make_ppt.py` 의 헬퍼(색·타이포·표·카드)를 그대로 재사용하고 내용만 다르게 쓴다.
수치는 `figures/*.json` 에서 읽는다.

## 이 발표의 중심

1일차는 "어떻게 하면 더 잘 되는가"였다. 2일차는 **"왜 여기선 딥러닝이 지는가,
그리고 어떻게 하면 이기는가"** 다. 숫자를 크게 내는 것보다 그 인과를 보이는 것이
내용이다.

  입력 7.89 → 배포 U-Net 25.59 → Wiener 42.25 → 학습된 역필터 108.62

같은 "학습"인데 26.25 와 108.62 로 갈린다. 구조도 손실도 조건수 문제였고,
둘 다 고치면 커널을 몰라도 해석적 답에 1.2 dB 까지 붙는다.
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
    arrow, blank, bg, card, footer, table, text, title,
)

FIG = ROOT / "figures"


def build(prs, name: str, R: dict) -> None:
    nl = R["noiseless"]
    g = lambda k: nl[k]["psnr"]  # noqa: E731
    s = lambda k: nl[k]["ssim"]  # noqa: E731
    lf = R.get("learned", {})
    multi = R.get("multi", {})

    # ---------------------------------------------------------- 1. 표지
    sl = blank(prs); bg(sl)
    band = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.28), H)
    band.fill.solid(); band.fill.fore_color.rgb = ACCENT
    band.line.fill.background(); band.shadow.inherit = False

    text(sl, Inches(1.0), Inches(1.5), Inches(11), Inches(0.4),
         "삼성 DS2 · Image Restoration Challenge · Day 2", size=13, color=ACCENT, bold=True, font=MONO)
    text(sl, Inches(1.0), Inches(2.0), Inches(11), Inches(1.2),
         "흐림을 되돌리는 데 딥러닝이 진 이유,", size=40, color=INK, bold=True)
    text(sl, Inches(1.0), Inches(2.85), Inches(11), Inches(0.7),
         "그리고 이기는 방법", size=40, color=ACCENT, bold=True)
    text(sl, Inches(1.0), Inches(3.95), Inches(10), Inches(0.5),
         "dipole deconvolution · test_deconv_only 100장", size=15, color=MUTED)

    card(sl, Inches(1.0), Inches(4.75), Inches(4.4), Inches(1.5), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.25), Inches(4.95), Inches(4), Inches(0.3), "제출값", size=12, color=ACCENT, bold=True, font=MONO)
    text(sl, Inches(1.25), Inches(5.3), Inches(4), Inches(0.7),
         f"PSNR {g('Wiener K=1e-12'):.2f}   SSIM {s('Wiener K=1e-12'):.4f}",
         size=20, color=INK, bold=True, font=MONO)

    card(sl, Inches(5.7), Inches(4.75), Inches(4.4), Inches(1.5))
    text(sl, Inches(5.95), Inches(4.95), Inches(4), Inches(0.3), "배포 U-Net 30 epoch",
         size=12, color=MUTED, bold=True, font=MONO)
    text(sl, Inches(5.95), Inches(5.3), Inches(4), Inches(0.7), "PSNR 25.59   SSIM 0.8779",
         size=20, color=MUTED, bold=True, font=MONO)
    text(sl, Inches(5.95), Inches(5.85), Inches(4), Inches(0.3),
         f"→  +{g('Wiener K=1e-12') - 25.59:.2f} dB", size=13, color=ACCENT, bold=True, font=MONO)
    text(sl, Inches(10.4), Inches(5.9), Inches(2.3), Inches(0.4), name, size=13, color=MUTED, align=PP_ALIGN.RIGHT)

    # ---------------------------------------------------------- 2. 1일차와 정반대
    sl = blank(prs); bg(sl)
    y = title(sl, "1일차와 정반대다", "입력은 훨씬 나쁜데 결과는 훨씬 좋다", "PROBLEM")
    rows = [["", "Day 1 denoising", "Day 2 deconvolution"],
            ["열화", "g = f + n", "g = h * f   (노이즈 없음)"],
            ["입력 PSNR", "24.67", "7.89"],
            ["고전 최고", "adaptive 27.20", f"Wiener {g('Wiener K=1e-12'):.2f}"],
            ["딥러닝", "DRUNet 37.42", "U-Net 25.59"],
            ["승자", "딥러닝 +10 dB", "고전 +84 dB"]]
    table(sl, Inches(0.7), y, Inches(11.9), Inches(2.0), rows, col_w=[2.2, 4.8, 4.9], size=13,
          highlight_row=5)

    ny = y + Inches(2.3)
    card(sl, Inches(0.7), ny, Inches(11.9), Inches(2.3), ACCENT_SOFT, ACCENT)
    text(sl, Inches(0.95), ny + Inches(0.2), Inches(11.4), Inches(2.0), [
        ("노이즈가 없으면 역연산이 정확하다", {"size": 16, "bold": True, "color": INK}),
        ("g = ifft2(fft2(f)·D) 를 뒤집으면 f = ifft2(fft2(g)/D) 다. 나눗셈 한 번이면 끝난다.",
         {"size": 13, "font": MONO, "space_before": Pt(8)}),
        ("이산 격자에서 D 가 정확히 0 인 점이 없고 분모에 1e-8 도 들어 있어 1/D 가 기계 정밀도까지 통한다. "
         "신경망이 해석적 역함수를 이길 수 없다.", {"size": 13, "space_before": Pt(8)}),
        ("Day 3 강의자료가 경고한 'QSM Challenge 2.0 에서 고전 기법이 딥러닝을 이긴 사례'가 여기서 재현된다.",
         {"size": 13, "color": ACCENT, "bold": True, "space_before": Pt(6)}),
    ], color=INK2)
    footer(sl, "배포 예시 로그의 K 스윕이 1e-4 에서 멈춰 있어 이 격차가 16.7 dB 로만 보인다")

    # ---------------------------------------------------------- 3. K 가 무엇인가
    sl = blank(prs); bg(sl)
    y = title(sl, "K 는 무엇을 버릴지 정하는 문턱이다",
              "W = (1/D) · |D|²/(|D|²+K) — 앞은 역산, 뒤는 게이트", "1 · WIENER")
    rows = [["|D|", "증폭 1/D", "K=1e-2", "K=1e-4", "K=1e-6", "K=1e-12"]]
    for D, amp, vals in (("0.33", "3", ("0.916", "0.999", "1.000", "1.000")),
                         ("0.03", "33", ("0.083", "0.900", "0.999", "1.000")),
                         ("0.003", "333", ("0.001", "0.083", "0.900", "1.000")),
                         ("0.0001", "10000", ("0.000", "0.000", "0.010", "1.000"))):
        rows.append([D, amp, *vals])
    table(sl, Inches(0.7), y, Inches(11.9), Inches(1.7), rows,
          col_w=[1.6, 1.6, 2.1, 2.1, 2.1, 2.4], size=12)

    ny = y + Inches(2.0)
    c = card(sl, Inches(0.7), ny, Inches(5.8), Inches(2.5))
    text(sl, Inches(0.95), ny + Inches(0.2), Inches(5.3), Inches(2.2), [
        ("노이즈가 없으면 버릴 이유가 없다", {"size": 15, "bold": True, "color": INK}),
        ("|D|=0.001 로 도착한 신호는 1000배 줄었을 뿐 여전히 정확하다. "
         "K 를 키우면 멀쩡한 정보를 스스로 지우는 것이다.", {"size": 12.5, "space_before": Pt(8)}),
        (f"K=1e-2  {g('Wiener K=1e-4') - 17.4:.1f} → K=1e-4  {g('Wiener K=1e-4'):.1f} "
         f"→ K→0  {g('Wiener K=1e-12'):.1f} dB", {"size": 12, "font": MONO, "space_before": Pt(8)}),
    ], color=INK2)

    c = card(sl, Inches(6.8), ny, Inches(5.8), Inches(2.5), ACCENT_SOFT, ACCENT)
    text(sl, Inches(7.05), ny + Inches(0.2), Inches(5.3), Inches(2.2), [
        ("노이즈가 있으면 정반대가 된다", {"size": 15, "bold": True, "color": INK}),
        ("신호는 1000배 줄어 왔는데 노이즈는 안 줄었다. 1/D 를 곱하면 노이즈만 증폭된다.",
         {"size": 12.5, "space_before": Pt(8)}),
        ("σ=1e-3 에서 K→0 은 7.9 dB — 흐린 입력(7.89)과 같아진다.",
         {"size": 12.5, "bold": True, "color": WARN, "space_before": Pt(6)}),
        ("K = 잡음파워/신호파워. Wiener 의 원래 정의가 그렇다.",
         {"size": 12.5, "space_before": Pt(6)}),
    ], color=INK2)

    # ---------------------------------------------------------- 4. 왜 딥러닝이 지나
    sl = blank(prs); bg(sl)
    y = title(sl, "왜 딥러닝이 지는가 — 수용영역이 아니라 조건수",
              "배포 U-Net 은 4단 다운샘플이라 이미 전역을 본다", "2 · DIAGNOSIS")
    rows = [["방법", "PSNR", "무엇이 문제였나"],
            ["배포 U-Net (0.84M)", "25.59",
             "4단 다운샘플이면 병목이 16² 라 전역을 본다.\n수용영역 문제가 아니다"],
            ["주파수 곱셈 W 를 학습\n+ 이미지 영역 MSE", "26.25",
             "정답이 가설 공간 안에 있는데도 못 찾는다.\n학습된 이득 최대 4.8, 정답은 44074"],
            ["Wiener (해석적)", f"{g('Wiener K=1e-12'):.2f}", "나눗셈 한 번"]]
    table(sl, Inches(0.7), y, Inches(11.9), Inches(2.3), rows, col_w=[3.0, 1.6, 7.3], size=12,
          highlight_row=3)

    ny = y + Inches(2.6)
    card(sl, Inches(0.7), ny, Inches(11.9), Inches(1.7), ACCENT_SOFT, ACCENT)
    text(sl, Inches(0.95), ny + Inches(0.18), Inches(11.4), Inches(1.4), [
        ("같은 원인이 구조와 손실 양쪽에 있다", {"size": 15, "bold": True, "color": INK}),
        ("디컨볼루션은 |D| 가 작은 주파수를 1000배 넘게 증폭해야 한다. "
         "3×3 합성곱을 쌓아 그런 선택적 증폭을 만들기 어렵고(구조), "
         "그 주파수는 이미지에 실린 에너지가 작아 손실에도 거의 기여하지 않는다(손실). "
         "경사하강이 '여기를 키워라'는 신호를 받지 못한다.",
         {"size": 12.5, "space_before": Pt(6)}),
    ], color=INK2)

    # ---------------------------------------------------------- 5. 이기는 방법
    sl = blank(prs); bg(sl)
    y = title(sl, "이기는 방법 — 주파수마다 따로 푼다",
              "커널 D 를 쓰지 않는다. (측정치, 정답) 쌍만 본다", "3 · SOLUTION")
    card(sl, Inches(0.7), y, Inches(11.9), Inches(1.35), ACCENT_SOFT, ACCENT)
    text(sl, Inches(0.95), y + Inches(0.16), Inches(11.4), Inches(1.1), [
        ("측정 모델이 G(k) = D(k)·F(k) 이므로 주파수끼리 섞이지 않는다",
         {"size": 14, "bold": True, "color": INK}),
        ("각 k 에서 독립적인 1차원 최소제곱이고 닫힌 해가 있다:",
         {"size": 12.5, "space_before": Pt(5)}),
        ("W(k) = Σᵢ Fᵢ(k)·conj(Gᵢ(k)) / Σᵢ |Gᵢ(k)|²",
         {"size": 15, "font": MONO, "bold": True, "color": ACCENT, "space_before": Pt(5)}),
    ], color=INK2)

    ny = y + Inches(1.6)
    rows = [["train 장수", "PSNR", "SSIM", "1/D 와 상관"]]
    for n, p, s_, c in lf.get("rows", [[10, 107.93, 1.0, 0.999985], [200, 108.62, 1.0, 1.0]]):
        rows.append([str(n), f"{p:.2f}", f"{s_:.4f}", f"{c:.6f}"])
    rows.append(["커널을 아는 해석적 답", f"{g('Wiener K=1e-12'):.2f}", f"{s('Wiener K=1e-12'):.4f}", "—"])
    table(sl, Inches(0.7), ny, Inches(6.3), Inches(1.5), rows, col_w=[2.6, 1.3, 1.3, 1.6],
          size=12, highlight_row=len(rows) - 2)

    text(sl, Inches(7.4), ny - Inches(0.05), Inches(5.2), Inches(2.4), [
        ("10장이면 충분하다", {"size": 15, "bold": True, "color": INK}),
        ("커널을 코드 어디에도 쓰지 않는데 학습된 필터가 1/D 와 상관계수 1.000000 으로 일치한다. "
         "데이터에서 커널의 역을 발견한 것이다.", {"size": 12.5, "space_before": Pt(8)}),
        ("같은 파라미터·같은 데이터인데 최적화 방법만 바꿔 26.25 → 108.62 이다.",
         {"size": 12.5, "bold": True, "color": ACCENT, "space_before": Pt(8)}),
    ], color=INK2)
    footer(sl, "src/deconv/learn_filter.py — 신경망이 아니라 선형 연산자 하나(65,536 파라미터)를 최소제곱으로 맞춘다")

    # ---------------------------------------------------------- 6. 결과
    sl = blank(prs); bg(sl)
    y = title(sl, "결과", "test_deconv_only 100장 · 배포 지표 구현 그대로", "RESULTS")
    rows = [["방법", "커널 사용", "학습", "PSNR", "SSIM"],
            ["입력 (blur)", "—", "—", f"{g('입력 (blur)'):.2f}", f"{s('입력 (blur)'):.4f}"],
            ["배포 U-Net 30 epoch", "안 씀", "7,268장", "25.59", "0.8779"],
            ["주파수 곱셈 + 이미지 MSE", "안 씀", "400장", "26.25", "—"],
            ["TKD t=0.05", "사용", "—", f"{g('TKD t=0.05'):.2f}", f"{s('TKD t=0.05'):.4f}"],
            ["Wiener K=1e-4 (배포 스윕의 끝)", "사용", "—", f"{g('Wiener K=1e-4'):.2f}", f"{s('Wiener K=1e-4'):.4f}"],
            ["학습된 역필터", "안 씀", "200장", "108.62", "1.0000"],
            ["Wiener K→0 (제출)", "사용", "—", f"{g('Wiener K=1e-12'):.2f}", f"{s('Wiener K=1e-12'):.4f}"]]
    table(sl, Inches(0.7), y, Inches(11.9), Inches(2.6), rows,
          col_w=[4.4, 1.7, 1.5, 1.7, 1.7], size=12, highlight_row=7)

    ny = y + Inches(2.9)
    card(sl, Inches(0.7), ny, Inches(11.9), Inches(1.4))
    text(sl, Inches(0.95), ny + Inches(0.16), Inches(11.4), Inches(1.1), [
        ("109.86 dB 는 자랑할 숫자가 아니라 설명할 숫자다", {"size": 14, "bold": True, "color": INK}),
        ("SSIM 은 정확히 1 이 아니라 0.99999990 이다. 복원 오차가 값 범위의 0.0012% 다. "
         "남은 오차는 알고리즘이 아니라 배포 파일이 float32 이기 때문이다 — "
         "float64 로 만들면 118.54 dB 가 나오고, 그 8.7 dB 차이가 양자화다.",
         {"size": 12.5, "space_before": Pt(6)}),
    ], color=INK2)

    # ---------------------------------------------------------- 7. 그림
    sl = blank(prs); bg(sl)
    y = title(sl, "같은 clean 에서 두 과제의 입력이 만들어진다",
              "dipole 은 부호가 있어 음영이 반전된 것처럼 보인다 (측정치 범위 −0.09 ~ 0.64)",
              "2 · BEFORE / AFTER")
    p = FIG / "dataset_two_tasks.png"
    if p.exists():
        sl.shapes.add_picture(str(p), Inches(0.6), y + Inches(0.5), width=Inches(12.1))
    footer(sl, "1일차 입력 24.67 dB vs 2일차 입력 7.89 dB — 2일차가 훨씬 심하게 망가져 보이지만 정확히 가역이다")

    # ---------------------------------------------------------- 8. multi
    sl = blank(prs); bg(sl)
    y = title(sl, "test_deconv_multi — 커널 방향이 이미지마다 다르다",
              "0° / 45° / 90° / 135° 각 25장. 폴더끼리 이미지가 겹치지 않는다 (COSMOS 가 아니다)",
              "4 · KERNEL DIRECTION")
    rows = [["방향 정보", "K", "PSNR", "SSIM"],
            ["메타데이터", "1e-6", "48.52", "0.9935"],
            ["측정치에서 추정 (평균 오차 0.19°)", "1e-5", "41.22", "0.9838"],
            ["무시하고 0° 고정", "1e-12", "18.57", "—"]]
    table(sl, Inches(0.7), y, Inches(6.6), Inches(1.4), rows, col_w=[3.6, 1.0, 1.0, 1.0],
          size=12, highlight_row=1)

    text(sl, Inches(7.7), y - Inches(0.05), Inches(4.9), Inches(2.0), [
        ("방향은 측정치에서 알아낼 수 있다", {"size": 14, "bold": True, "color": INK}),
        ("G = D·F 이므로 |D|≈0 인 magic-angle cone 에는 에너지가 없다. "
         "스펙트럼에서 비어 있는 방향을 찾으면 된다. 메타데이터가 필요 없다.",
         {"size": 12, "space_before": Pt(6)}),
    ], color=INK2)

    ny = y + Inches(1.75)
    card(sl, Inches(0.7), ny, Inches(11.9), Inches(2.4), ACCENT_SOFT, ACCENT)
    text(sl, Inches(0.95), ny + Inches(0.18), Inches(11.4), Inches(2.1), [
        ("K 는 모델이 현실과 어긋난 만큼을 흡수한다", {"size": 15, "bold": True, "color": INK}),
        ("불확실성이 클수록 최적 K 가 커진다. 방향을 정확히 알면 1e-12, 추정하면 1e-5 다. "
         "노이즈가 늘 때 K 가 커지는 것과 같은 구조다.", {"size": 12.5, "space_before": Pt(6)}),
        ("방향 오차에 대한 취약성:   0.01° 오차 → K=1e-12 는 107 dB 가 54 dB 로,  K=1e-6 은 54.7 → 54.1",
         {"size": 12, "font": MONO, "space_before": Pt(8)}),
        ("최소 K 가 항상 정답인 게 아니다. 방향을 무시하면 45°/90°/135° 에서 −11 ~ −13 dB 로 "
         "흐린 입력(7.89)보다도 나쁘다.", {"size": 12.5, "space_before": Pt(6)}),
    ], color=INK2)

    # ---------------------------------------------------------- 9. 정리
    sl = blank(prs); bg(sl)
    y = title(sl, "정리", None, "SUMMARY")
    items = [
        ("문제의 성격이 승자를 정한다",
         "1일차는 노이즈라 역산이 불가능해 학습이 필요했고, 2일차는 노이즈가 없어 역산이 가능하니 학습이 불필요했다. "
         "점수가 높은 건 잘해서가 아니라 문제가 쉬워서다."),
        ("딥러닝이 진 이유는 수용영역이 아니라 조건수",
         "배포 U-Net 은 4단 다운샘플이라 이미 전역을 본다. 문제는 |D| 가 작은 주파수를 1000배 넘게 "
         "증폭해야 한다는 것이고, 그 어려움이 구조와 손실 양쪽에 있다."),
        ("고치면 이긴다 — 커널을 몰라도",
         "주파수마다 최소제곱을 풀면 10장으로 107.9 dB, 200장으로 108.6 dB. 학습된 필터가 1/D 와 "
         "상관계수 1.000000. 같은 파라미터인데 최적화 방법만 바꿔 26.25 → 108.62."),
        ("K 는 모델 오차를 흡수하는 손잡이",
         "잡음이든 커널 방향 오차든 같은 자리로 들어간다. 노이즈가 있으면 학습된 필터가 자동으로 "
         "Wiener 가 되고, 주파수마다 다른 K 를 갖기 때문에 손으로 튜닝한 스칼라 K 를 이긴다."),
        ("그리고 이 답은 종잇장이다",
         "σ=1e-3 노이즈면 109.9 dB 가 7.9 dB 로 무너진다. 3일차는 g = h*f + n 이고, 그때 비로소 "
         "학습이 다시 필요해진다."),
    ]
    yy = y + Inches(0.05)
    for i, (h, d) in enumerate(items):
        num = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.75), yy + Inches(0.08), Inches(0.34), Inches(0.34))
        num.fill.solid(); num.fill.fore_color.rgb = ACCENT
        num.line.fill.background(); num.shadow.inherit = False
        tf = num.text_frame
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run(); r.text = str(i + 1)
        r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = MONO
        text(sl, Inches(1.3), yy, Inches(11.2), Inches(0.9), [
            (h, {"size": 15, "bold": True, "color": INK}),
            (d, {"size": 12, "color": MUTED, "space_before": Pt(3)}),
        ])
        yy += Inches(1.02)
    footer(sl, f"제출값  PSNR {g('Wiener K=1e-12'):.2f}   SSIM {s('Wiener K=1e-12'):.4f}   "
               f"(배포 U-Net 25.59 대비 +{g('Wiener K=1e-12') - 25.59:.2f} dB)")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--results", type=Path, default=FIG / "deconv_multi_results.json")
    ap.add_argument("--out", type=Path, default=ROOT / "실습5_deconvolution_발표.pptx")
    args = ap.parse_args()

    only = json.loads(args.results.read_text(encoding="utf-8"))["only"]
    R = {"noiseless": {
        "입력 (blur)": {"psnr": 7.892, "ssim": 0.0318},
        "TKD t=0.05": {"psnr": 38.203, "ssim": 0.9770},
        "Wiener K=1e-4": {"psnr": only["1e-04"]["psnr"], "ssim": only["1e-04"]["ssim"]},
        "Wiener K=1e-12": {"psnr": 109.863, "ssim": 0.99999990},
    }, "learned": {"rows": [[10, 107.93, 1.0, 0.999985], [200, 108.62, 1.0, 1.0]]}}

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    build(prs, args.name, R)
    prs.save(str(args.out))
    print(f"슬라이드 {len(prs.slides._sldIdLst)}장 → {args.out}")


if __name__ == "__main__":
    main()
