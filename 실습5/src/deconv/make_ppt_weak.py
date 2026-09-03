"""'어떤 노이즈에 약한가' 그래프 한 장.

`day3_weakness.png` 는 막대(노이즈 종류 × 방법)와 산점도(σ 대 PSNR)를 나란히
놓은 그림이다. 표보다 한눈에 들어와서 따로 한 장을 준다.

본 발표 파일은 건드리지 않는다 — 손으로 고친 것이 있기 때문이다. 이 파일을
열어 슬라이드를 복사해 넣으면 된다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))

from make_ppt import (  # noqa: E402
    ACCENT, INK, MUTED, SURF, H, W, Inches, Presentation,
    blank, bg, card, footer, text, title,
)
from make_ppt_how import runs  # noqa: E402
from make_ppt_zoom import fit  # noqa: E402

B = {"bold": True}
C = {"bold": True, "color": ACCENT}
G = {"color": MUTED}

READ = [
    [("왼쪽 — 방법별로 얼마나 벌어지나", C)],
    [("네 종류 모두 같은 순서다. 측정치 < Wiener < median→Wiener < 우리 모델", {})],
    [("특정 종류에 맞춘 게 아니라는 뜻", G)],
    [],
    [("Rician 만 막대가 낮다", B), (". 다른 셋은 30 부근인데 23.41 이다", {})],
    [("평균이 0 이 아니라서 생긴 편향이 DC 를 타고 3배가 된다", G)],
    [],
    [("Salt & Pepper 가 가장 높다", B), (" (38.19)", {})],
    [("임펄스는 주파수로 넓게 퍼져 영널 원뿔 밖에 대부분 놓인다", G)],
    [],
    [("오른쪽 — σ 가 커질수록 떨어진다", C)],
    [("σ 0.0007~0.1325, 200배 범위다", {})],
    [("중요한 건 ", {}), ("색깔이 섞이지 않고 층을 이룬다", B), ("는 점이다", {})],
    [("같은 σ 에서도 Rician(주황)은 아래, Salt & Pepper(빨강)는 위다. "
      "세기만이 아니라 종류가 따로 영향을 준다", G)],
]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=Path, default=ROOT / "figures" / "day3_weakness.png")
    ap.add_argument("--size", type=float, default=9.5)
    ap.add_argument("--out", type=Path, default=ROOT / "day3_weakness.pptx")
    args = ap.parse_args()

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    sl = blank(prs); bg(sl)
    title(sl, "취약점 — 어떤 노이즈에 약한가",
          "노이즈 종류별 방법 비교와, σ 가 커질 때의 성능 변화",
          eyebrow="결과 분석")

    fit(sl, args.fig, Inches(0.5), Inches(2.2), Inches(8.2), Inches(4.3))

    x = Inches(8.85)
    card(sl, x, Inches(2.2), Inches(3.92), Inches(4.3), fill=SURF)
    text(sl, x + Inches(0.24), Inches(2.36), Inches(3.0), Inches(0.28),
         "읽는 법", size=11, color=ACCENT, bold=True)
    runs(sl, x + Inches(0.24), Inches(2.66), Inches(3.45), Inches(3.7),
         READ, size=args.size)

    footer(sl, "test_deconv_noise 100장 · 대표 이미지가 아니라 100장 전부를 찍은 것이다")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(args.out))
    print(f"저장 → {args.out}")


if __name__ == "__main__":
    main()
