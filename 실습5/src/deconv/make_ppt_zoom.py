"""difference map + zoom 슬라이드만 따로 뽑는다 — 노이즈 종류마다 한 장씩.

본 발표에는 대표 한 장(rician)만 넣고, 질문이 나오면 이 묶음을 열어 나머지를
보여주는 쓰임을 생각했다. 그래서 본 파일과 별도로 만든다 — 손으로 고친 발표
파일을 건드리지 않는다.

먼저 그림을 만들어 두어야 한다:

    python figures_day3.py --ckpt <ckpt> --self-ensemble --shift-ensemble

종류마다 `day3_diff_zoom_<종류>.png` 가 나오고, 여기서 그것들을 한 장씩 담는다.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))

from make_ppt import (  # noqa: E402
    ACCENT, INK, INK2, MONO, MUTED, H, W, Inches, MSO_SHAPE, Presentation,
    blank, bg, footer, pic, text, title,
)

FIG = ROOT / "figures"

# 최종 제출(16x self-ensemble, 융합)에서 실측한 종류별 점수
SCORE = {
    "gaussian":       (30.37, 0.8891),
    "rician":         (23.41, 0.7689),
    "uniform":        (29.81, 0.9150),
    "salt_and_pepper": (38.19, 0.9864),
}
KO = {"gaussian": "Gaussian", "rician": "Rician",
      "uniform": "Uniform", "salt_and_pepper": "Salt & Pepper"}

# 종류마다 difference map 에서 **무엇을 봐야 하는지**. 그림만 넘기면 아무도 안 본다.
READ = {
    "gaussian": (
        "가장 무난한 경우다. 오차는 경계와 잔무늬에 몰리고 평평한 곳은 거의 비어 있다",
        [("평균이 0 인 백색 노이즈라 역산이 키워도 편향이 남지 않는다. ", {}),
         ("남는 오차는 노이즈가 아니라 영널 원뿔에서 잃은 정보다", {"bold": True, "color": INK})],
        "X자 무늬 = 매직앵글 영널 원뿔. 정보가 애초에 없는 자리라 어떤 방법으로도 남는다"),
    "rician": (
        "우리 약점이다. difference map 이 **고르게** 밝다 — 경계가 아니라 전면이 틀렸다",
        [("Rician 만 평균이 0 이 아니다(+0.04~0.06). ", {}),
         ("그 편향은 순수 DC 이고 dipole 은 DC 를 1/3 로 보존하므로 역산에서 3배가 된다", {"bold": True, "color": INK}),
         (". 이미지 std 가 0.222 인데 0.119 가 편향으로 남는다", {})],
        "다른 종류보다 7 dB 낮은 23.41. 원인은 노이즈 세기가 아니라 평균이다"),
    "uniform": (
        "Gaussian 과 거의 같게 보인다. 노이즈의 모양보다 세기가 결과를 정한다",
        [("균등 분포도 평균이 0 이라 편향이 없다. ", {}),
         ("모델이 분포 모양을 따로 배우지 않아도 되는 이유", {"bold": True, "color": INK}),
         (" — σ 하나만 알면 충분하다", {})],
        "29.81 dB. Gaussian(30.37) 과의 차이는 대표 이미지의 σ 차이 정도다"),
    "salt_and_pepper": (
        "가장 잘 나온다(38.19 dB). 오차는 점으로만 남고 면으로는 번지지 않는다",
        [("튀는 화소는 드물고 ", {}),
         ("주파수 영역에서 넓게 퍼져 영널 원뿔 밖에 대부분 놓인다", {"bold": True, "color": INK}),
         (". 그래서 데이터 일관성 단계가 대부분 걷어낸다", {})],
        "38.19 dB / 0.9864 — 네 종류 중 최고. 임펄스는 이 문제에서 가장 쉬운 노이즈다"),
}


def fit(slide, path: Path, x, y, box_w, box_h):
    """그림을 상자 안에 비율 그대로 넣고 가운데에 놓는다."""
    if not path.exists():
        return pic(slide, path, x, y, w=box_w)
    with path.open("rb") as fh:
        head = fh.read(33)
    iw, ih = struct.unpack(">II", head[16:24])   # PNG IHDR
    sc = min(box_w / iw, box_h / ih)
    w, h = int(iw * sc), int(ih * sc)
    return pic(slide, path, x + (box_w - w) // 2, y + (box_h - h) // 2, w=w)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=Path, default=FIG, help="그림이 있는 폴더")
    ap.add_argument("--types", nargs="*", default=list(SCORE), choices=list(SCORE))
    ap.add_argument("--out", type=Path, default=ROOT / "실습5_day3_zoom.pptx")
    args = ap.parse_args()

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ---- 표지 ----
    sl = blank(prs); bg(sl)
    band = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.28), H)
    band.fill.solid(); band.fill.fore_color.rgb = ACCENT
    band.line.fill.background(); band.shadow.inherit = False
    text(sl, Inches(1.0), Inches(2.0), Inches(11), Inches(0.4),
         "부록 · Day 3", size=13, color=ACCENT, bold=True, font=MONO)
    text(sl, Inches(1.0), Inches(2.5), Inches(11), Inches(1.1),
         "difference map 과 zoom — 노이즈 4종", size=36, color=INK, bold=True)
    text(sl, Inches(1.0), Inches(3.7), Inches(11), Inches(0.9),
         "같은 방법으로 복원했을 때 오차가 **어디에** 남는지는 노이즈마다 다르다. "
         "종류별로 한 장씩 본다.", size=15, color=MUTED)
    footer(sl, "대표 이미지는 그 종류의 σ 중앙값에 가장 가까운 것 — 잘 나온 것을 고르지 않았다")

    # ---- 종류별 ----
    for nz in args.types:
        p, s = SCORE[nz]
        sub, body, foot = READ[nz]
        sl = blank(prs); bg(sl)
        title(sl, f"{KO[nz]} — {p:.2f} dB / {s:.4f}", sub, eyebrow="difference map + zoom")
        f = args.fig / f"day3_diff_zoom_{nz}.png"
        # 열 수가 모델 개수에 따라 달라져 비율이 제각각이다. 상자에 맞춰 넣는다
        fit(sl, f, Inches(0.6), Inches(1.62), Inches(6.5), Inches(4.6))
        text(sl, Inches(7.4), Inches(2.1), Inches(5.3), Inches(3.0), body,
             size=13, color=INK2)
        footer(sl, foot)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(args.out))
    print(f"저장 → {args.out}  ({1 + len(args.types)}장)")


if __name__ == "__main__":
    main()
