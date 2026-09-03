"""'전처리 · 모델 · 학습' 한 장. 발표에서 반드시 물어보는 세 가지를 한 화면에.

본 발표 파일과 따로 만든다 — 손으로 고친 것을 건드리지 않기 위해서다.
문장은 짧게. 이 장은 말로 채우는 자리고, 슬라이드는 뼈대만 보인다.
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

B = {"bold": True, "color": INK}
C = {"color": ACCENT, "bold": True}
M = {"font": MONO, "color": INK}

COLS = [
    ("01", "데이터 전처리", "정규화·크롭 없음. 쌍을 만든다", [
        [("깨끗한 이미지 ", {}), ("7,268장", B), ("뿐. 열화 쌍이 없다", {})],
        [("매 배치마다 ", {}), ("f → dipole → +noise", M), (" 로 직접 생성", {})],
        [],
        [("augmentation: ", {}), ("뒤집기만", B), (". 90° 회전 제외", {})],
        [("→ dipole 은 B₀ 방향이 있는 ", {}), ("비등방", C),
         (" 연산. 회전하면 test 에 없는 방향이 된다", {})],
        [],
        [("크롭은 ", {}), ("흐리게 만든 뒤", B), ("에", {})],
        [("→ dipole 은 FFT 전역 연산. 먼저 자르면 경계가 거짓이 된다", {})],
        [],
        [("노이즈는 ", {}), ("4종 시뮬레이터", B), (" 그대로", {})],
        [("→ Gaussian 만 쓰면 Rician 의 편향을 못 배운다", {})],
        [],
        [("val 은 파일명 seed 로 노이즈 고정 · ", {}), ("test 는 채점에만", C)],
    ]),
    ("02", "모델", "Unrolled network — 물리를 층으로", [
        [("x₀ = Wiener(g, λ₀)", M)],
        [("반복 N: ", M), ("z = DRUNet(x, σ̂)", M)],
        [("          x = (D·G + λZ)/(D² + λ)", M)],
        [],
        [("데이터 정합 단계에 ", {}), ("dipole D 가 그대로", B),
         (" 들어간다 → 물리를 배울 필요가 없다", {})],
        [],
        [("가중치 공유", B), (" — N 단계가 같은 망. 파라미터 N배 절약, "
                          "추론 때 반복 수를 바꿀 수 있다", {})],
        [],
        [("σ 조건화", B), (" — 세기를 여분 채널로. 단 ", {}),
         ("σ 는 측정치에서 읽는다", C)],
        [("→ 영널 원뿔(|D|<0.02)엔 노이즈만 있다. 라벨·메타데이터 불필요 (오차 1.9%)", {})],
        [],
        [("제출 = 구조가 다른 두 모델 평균 ", {}),
         ("4단계 30.32 + 6단계 30.41 → 30.44", B)],
    ]),
    ("03", "학습", "수렴까지 돌리는 것이 가장 큰 지렛대", [
        [("Charbonnier √(e²+ε²)", M), (" · AdamW 2e-4 · cosine", {})],
        [("bf16 · clip 1.0 · batch 8 · 256² 통째로", {})],
        [],
        [("λ 도 학습한다", B), (" (log 로 저장, 단계마다 따로)", {})],
        [("초기값은 Wiener 최적값을 재서 → 시작이 ", {}), ("14.05 dB", B), (" (아니면 10.19)", {})],
        [],
        [("best epoch 이 세 번 연속 마지막", C), (" — 수렴 전이었다. 이어 돌린 것이 "
                                            "점수를 가장 많이 올렸다", {})],
        [],
        [("SSIM loss 는 처음부터 쓰면 나빠진다", B), (" (18.73→17.53). 미세조정으로만", {})],
        [],
        [("추론: ", {}), ("16× self-ensemble", B), (" — 뒤집기·180°·대각 순환이동. "
                                                 "dipole 과 교환되는 변환만 썼다", {})],
        [],
        [("모델은 ", {}), ("val 100장", C), (" 으로 골랐다 — test 로 고르면 test 를 쓴 것이다", {})],
    ]),
]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "day3_how.pptx")
    args = ap.parse_args()

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    sl = blank(prs); bg(sl)
    title(sl, "어떻게 만들었나 — 전처리 · 모델 · 학습",
          "깨끗한 이미지만 주어진 문제다. 데이터를 만들고, 물리를 층으로 넣고, 수렴까지 돌렸다",
          eyebrow="방법 요약")

    x0, cw, gap = Inches(0.62), Inches(3.95), Inches(0.2)
    for i, (num, head, sub, lines) in enumerate(COLS):
        x = x0 + i * (cw + gap)
        card(sl, x, Inches(1.72), cw, Inches(4.75), fill=SURF)
        text(sl, x + Inches(0.26), Inches(1.92), cw, Inches(0.3), num,
             size=11, color=ACCENT, bold=True, font=MONO)
        text(sl, x + Inches(0.26), Inches(2.16), cw - Inches(0.5), Inches(0.4), head,
             size=19, color=INK, bold=True)
        text(sl, x + Inches(0.26), Inches(2.58), cw - Inches(0.5), Inches(0.3), sub,
             size=11, color=MUTED)
        y = Inches(2.98)
        for ln in lines:
            if not ln:                      # 빈 줄 = 문단 사이 숨 쉬는 자리
                y += Inches(0.12)
                continue
            text(sl, x + Inches(0.26), y, cw - Inches(0.5), Inches(0.34), ln,
                 size=10.5, color=INK2, spacing=1.12)
            # 한 줄에 들어가는 글자 수로 몇 줄이 될지 어림해 다음 줄 위치를 잡는다
            n = sum(len(s if isinstance(s, str) else s[0]) for s in ln)
            y += Inches(0.185) * max(1, -(-n // 30))

    footer(sl, "test_deconv_noise 100장 · 최종 30.44 dB / 0.8899 "
               "(배포 baseline 25.02 / 0.815 대비 +5.42 dB — 오차 에너지 3.5배 감소)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(args.out))
    print(f"저장 → {args.out}")


if __name__ == "__main__":
    main()
