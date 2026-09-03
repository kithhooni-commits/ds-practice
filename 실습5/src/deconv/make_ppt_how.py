"""'전처리 · 모델 · 학습' 한 장. 발표에서 반드시 물어보는 세 가지를 한 화면에.

본 발표 파일과 따로 만든다 — 손으로 고친 것을 건드리지 않기 위해서다.
문장은 짧게. 이 장은 말로 채우는 자리고, 슬라이드는 뼈대만 보인다.

## 배치

칸마다 **텍스트 상자 하나**에 문단을 쌓는다. 줄 높이를 손으로 계산해 상자를
여러 개 놓았더니 줄바꿈이 예상과 달라 글자가 겹쳤다. 파워포인트가 흐르게 두는
편이 맞다.

`make_ppt.text` 는 항목 하나를 문단 하나로 만들어 한 문단 안에서 굵게를 섞을 수
없다. 그래서 여기서는 `runs` 를 따로 쓴다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))

from make_ppt import (  # noqa: E402
    ACCENT, FONT, INK, INK2, MONO, MUTED, SURF, H, W, Inches, Pt, Presentation,
    MSO_ANCHOR, blank, bg, card, footer, text, title,
)

B = {"bold": True}                 # 굵게
C = {"bold": True, "color": ACCENT}  # 강조 (초록)
M = {"font": MONO}                 # 고정폭

COLS = [
    ("01", "데이터 전처리", "정규화·크롭 없음 — 쌍을 만든다", [
        [("깨끗한 이미지 ", {}), ("7,268장뿐.", B), (" 열화 쌍이 없다", {})],
        [("매 배치마다 ", {}), ("f → dipole → +noise", M), (" 로 직접 생성", {})],
        [],
        [("augmentation 은 뒤집기만, ", {}), ("90° 회전 제외", B)],
        [("dipole 은 B₀ 방향이 있는 비등방 연산 — 회전하면 test 에 "
          "없는 방향이 된다", {"color": MUTED})],
        [],
        [("크롭은 ", {}), ("흐리게 만든 뒤에", B)],
        [("dipole 은 FFT 전역 연산이라 먼저 자르면 경계가 거짓이 된다",
          {"color": MUTED})],
        [],
        [("노이즈는 ", {}), ("4종 시뮬레이터 그대로", B)],
        [("Gaussian 만 쓰면 Rician 의 편향을 못 배운다", {"color": MUTED})],
        [],
        [("val 은 파일명 seed 로 노이즈 고정 · ", {}), ("test 는 채점에만", C)],
    ]),
    ("02", "모델", "Unrolled network — 물리를 층으로", [
        [("x₀ = Wiener(g, λ₀)", M)],
        [("반복 N:  z = DRUNet(x, σ̂)", M)],
        [("             x = (D·G + λZ) / (D² + λ)", M)],
        [],
        [("데이터 정합 단계에 ", {}), ("dipole D 가 그대로", B), (" 들어간다", {})],
        [("→ 물리를 배울 필요가 없다", {"color": MUTED})],
        [],
        [("가중치 공유", B), (" — N 단계가 같은 망", {})],
        [("파라미터 N배 절약. 추론 때 반복 수를 바꿀 수 있다", {"color": MUTED})],
        [],
        [("σ 조건화", B), (" — 세기를 여분 채널로. 단 ", {}),
         ("σ 는 측정치에서 읽는다", C)],
        [("영널 원뿔엔 노이즈만 있다 → 라벨·메타데이터 불필요 (오차 1.9%)",
          {"color": MUTED})],
        [],
        [("제출 = 구조가 다른 두 모델 평균", {})],
        [("4단계 30.32  +  6단계 30.41  →  30.44", B)],
    ]),
    ("03", "학습", "수렴까지 돌리는 것이 가장 큰 지렛대", [
        [("Charbonnier √(e²+ε²) · AdamW 2e-4 · cosine", M)],
        [("bf16 · clip 1.0 · batch 8 · 256² 통째로", M)],
        [],
        [("λ 도 학습한다", B), (" (log 로 저장, 단계마다 따로)", {})],
        [("초기값은 Wiener 최적값을 재서 → 시작이 10.19 가 아닌 14.05 dB",
          {"color": MUTED})],
        [],
        [("best epoch 이 세 번 연속 마지막", C)],
        [("아직 수렴 전이었다. 이어 돌린 것이 점수를 가장 많이 올렸다",
          {"color": MUTED})],
        [],
        [("SSIM loss 는 처음부터 쓰면 나빠진다", B), (" (18.73 → 17.53)", {})],
        [("정확한 모델에서 미세조정으로만", {"color": MUTED})],
        [],
        [("추론에 ", {}), ("16× self-ensemble", B), (" — 뒤집기·180°·대각 순환이동", {})],
        [("dipole 과 교환되는 변환만 (오차 7e-16)", {"color": MUTED})],
        [],
        [("모델은 ", {}), ("val 100장", C), (" 으로 골랐다", {})],
    ]),
]


def runs(slide, x, y, w, h, paras, size=10.5, color=INK2, gap=6):
    """문단 목록을 상자 하나에 쌓는다. 문단 하나가 run 여러 개일 수 있다.

    빈 문단은 문단 사이 간격으로 쓴다 — 글자 없이 space_before 만 준다.
    """
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    first = True
    for para in paras:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.line_spacing = 1.22
        if not para:                       # 숨 쉬는 자리
            p.space_before = Pt(0)
            for r in [p.add_run()]:
                r.text = ""
                r.font.size = Pt(gap)
            continue
        p.space_before = Pt(2)
        for item in para:
            s, opt = item if isinstance(item, tuple) else (item, {})
            r = p.add_run()
            r.text = s
            f = r.font
            f.name = opt.get("font", FONT)
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", False)
            f.color.rgb = opt.get("color", color)
    return tb


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=float, default=9.5, help="본문 글자 크기")
    ap.add_argument("--out", type=Path, default=ROOT / "day3_how.pptx")
    args = ap.parse_args()

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    sl = blank(prs); bg(sl)
    title(sl, "어떻게 만들었나 — 전처리 · 모델 · 학습",
          "깨끗한 이미지만 주어진 문제다. 데이터를 만들고, 물리를 층으로 넣고, 수렴까지 돌렸다",
          eyebrow="방법 요약")

    top, ch = Inches(2.26), Inches(4.55)
    x0, cw, gap = Inches(0.62), Inches(3.95), Inches(0.2)
    for i, (num, head, sub, paras) in enumerate(COLS):
        x = x0 + i * (cw + gap)
        card(sl, x, top, cw, ch, fill=SURF)
        text(sl, x + Inches(0.24), top + Inches(0.16), cw, Inches(0.26), num,
             size=10, color=ACCENT, bold=True, font=MONO)
        text(sl, x + Inches(0.24), top + Inches(0.38), cw - Inches(0.48), Inches(0.34),
             head, size=16, color=INK, bold=True)
        text(sl, x + Inches(0.24), top + Inches(0.72), cw - Inches(0.48), Inches(0.24),
             sub, size=10, color=MUTED)
        runs(sl, x + Inches(0.24), top + Inches(1.02), cw - Inches(0.48),
             ch - Inches(1.2), paras, size=args.size)

    footer(sl, "test_deconv_noise 100장 · 최종 30.44 dB / 0.8899 "
               "(배포 baseline 25.02 / 0.815 대비 +5.42 dB — 오차 에너지 3.5배 감소)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(args.out))
    print(f"저장 → {args.out}")


if __name__ == "__main__":
    main()
