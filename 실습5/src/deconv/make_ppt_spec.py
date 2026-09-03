"""'네트워크 · 학습 방법 · 학습 시간' 한 장. 제출 양식에 그대로 옮겨 적을 수 있게.

`make_ppt_how.py` 와 같은 3단 카드 배치를 쓴다 — 나란히 놓았을 때 한 벌로 보이게.
숫자는 실측이고, 로그가 없어 어림한 것은 '약' 을 붙여 구분한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))

from make_ppt import (  # noqa: E402
    ACCENT, INK, INK2, MONO, MUTED, SURF, H, W, Inches, Presentation,
    blank, bg, card, footer, text, title,
)
from make_ppt_how import runs  # noqa: E402

B = {"bold": True}
C = {"bold": True, "color": ACCENT}
M = {"font": MONO}
G = {"color": MUTED}

COLS = [
    ("01", "네트워크", "Unrolled network — 물리를 층으로", [
        [("x₀ = Wiener(g, λ₀)", M)],
        [("반복 N:  σ̂ = estimate_sigma(g)", M)],
        [("             z = DRUNet(x, σ̂)", M)],
        [("             x = (D·G + λZ)/(D² + λ)", M)],
        [],
        [("백본 ", {}), ("DRUNet", B), (" (U-Net + residual, σ 조건화 채널) · features 48", {})],
        [("파라미터 ", {}), ("18,359,572", B), (" — 그중 λ 는 4개뿐", {})],
        [("가중치 공유: 모든 단계가 같은 망", G)],
        [],
        [("pretrained 사용 안 함", C)],
        [("ImageNet·공개 DRUNet 등 외부 가중치 0. 주어진 데이터로만 만들었다", G)],
        [("초기화는 무작위, 또는 우리가 직접 학습한 체크포인트에서 이어 학습", G)],
        [],
        [("제출 = 4단계 + 6단계 두 모델의 0.5:0.5 평균", B)],
    ]),
    ("02", "학습 방법", "Supervised — 열화 쌍을 직접 만든다", [
        [("제공된 train ", {}), ("7,268장", B), (" 만 사용. ", {}), ("test 는 채점에만", C)],
        [("clean f 에서 매 배치 ", {}), ("f → dipole → +noise(4종)", M), (" 로 쌍 생성", {})],
        [],
        [("Loss  Charbonnier √(e²+ε²)", M)],
        [("Optim  AdamW · lr 2e-4 · wd 1e-5", M)],
        [("Sched  warmup 200 step → cosine", M)],
        [("AMP   bf16 · grad clip 1.0", M)],
        [("Batch  8 · 256² 통째로 (크롭 없음)", M)],
        [],
        [("augmentation 은 뒤집기만 — 90° 회전 제외", G)],
        [("λ 도 함께 학습 (log 로 저장, 단계마다 따로)", G)],
        [],
        [("모델 선택은 ", {}), ("val 100장", C), (" 의 best epoch", {})],
        [("추론에 16× self-ensemble (뒤집기·180°·대각 순환이동)", G)],
    ]),
    ("03", "학습 시간", "Colab A100 1장", [
        [("v1  60 epoch", M), ("        6,133 s  (1.7 h)", B)],
        [("long  4단계 이어 학습", M), ("   약 2.5 h", {})],
        [("iter6  6단계 이어 학습", M), ("  약 2.5 h", {})],
        [],
        [("제출 모델 합계  약 7 h", C)],
        [],
        [("실측으로 기록된 것은 v1 의 6,133초다. 이어 학습 두 건은 로그를 남기지 "
          "않아 에폭당 약 100초 기준으로 어림했다", G)],
        [],
        [("탐색·실패한 시도까지 포함한 3일차 총 GPU 시간", {})],
        [("약 15 h", B), ("  (DC-Net · 2단 분해 · label-free 등)", G)],
        [],
        [("추론 (test 100장)", {})],
        [("16× self-ensemble  약 10~20분 · 앙상블 없이 1분 이내", G)],
    ]),
]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=float, default=9.5)
    ap.add_argument("--out", type=Path, default=ROOT / "day3_spec.pptx")
    args = ap.parse_args()

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    sl = blank(prs); bg(sl)
    title(sl, "구성 — 네트워크 · 학습 방법 · 학습 시간",
          "외부 사전학습 가중치는 쓰지 않았다. 주어진 데이터만으로 A100 한 장에서 만들었다",
          eyebrow="제출 정보")

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
               "(배포 baseline 25.02 / 0.815 대비 +5.42 dB)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(args.out))
    print(f"저장 → {args.out}")


if __name__ == "__main__":
    main()
