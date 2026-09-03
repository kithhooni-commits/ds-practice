"""self-supervised 트랙 발표 슬라이드 (별도 파일).

팀원이 만든 `train_final_selfsupervised_patched_v2.ipynb` 결과를 정리한 것이다.
본 발표(`make_ppt3.py`)와 같은 헬퍼를 써서 style 이 맞고, 따로 뽑아 합칠 수 있다.

## 이 트랙의 성과

    classical 최고 (Wiener K=1e-2)   15.10 dB / 0.385
    self-supervised (N2N + EI)       24.55 dB / 0.781
    + TTA                            24.79 dB / 0.787     <- +9.7 dB

라벨을 한 번도 쓰지 않고 낸 값이다. 참고로 지도학습 최종은 30.37 / 0.8900 이다.

## 슬라이드

    1  개요        2단 구조와 결과 한 줄
    2  Stage 1     Noise2Noise — 노이즈만 지운다
    3  Stage 2     Equivariant Imaging — 원뿔을 메운다
    4  결과        classical 비교 · 노이즈별 · 데이터 효율성
    5  한계        rician 이 왜 어려운가 · 대칭의 두 얼굴

## 대칭의 두 얼굴 (4·5번의 논점)

self-ensemble 은 **연산자와 교환되는** 변환만 쓴다 (뒤집기). 교환되지 않으면 돌린
입력에 안 돌린 커널을 적용하는 꼴이라 손해다.

Equivariant Imaging 은 정반대로 **교환되지 않는** 변환을 써야 한다 (90도 회전).
교환되면 A(Tx) 가 새로운 측정 방향을 주지 못해 원뿔에 대한 정보가 늘지 않는다.

같은 사실(dipole 은 90도 회전과 교환되지 않는다)이 한쪽에서는 금지, 다른 쪽에서는
필수가 된다. 발표에서 짚기 좋은 대비다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))

from make_ppt import (  # noqa: E402
    ACCENT, ACCENT_SOFT, INK, INK2, MONO, MUTED, RULE, SURF, WARN,
    H, W, Inches, MSO_SHAPE, PP_ALIGN, Presentation,
    arrow, blank, bg, card, footer, table, text, title,
)

B = {"bold": True, "color": INK}
N: dict = {}

SUP = 30.37          # 지도학습 최종 (비교용)
WIENER = 15.10       # classical 최고


def build(prs, name: str) -> None:
    # ============================================================ 1. 개요
    sl = blank(prs); bg(sl)
    title(sl, "보너스 — self-supervised. 라벨을 한 번도 쓰지 않고",
          "노이즈 제거와 역산을 나눠, 각각 다른 자기지도 기법을 건다",
          eyebrow="보너스")

    xs = Inches(0.75)
    boxes = [("측정치 g", "blur + noise", SURF),
             ("Stage 1\n디노이저", "U-Net 0.48M\nNoise2Noise", ACCENT_SOFT),
             ("denoised blur", "노이즈만 지운 상태\n(흐림은 그대로)", SURF),
             ("Stage 2\n역산", "U-Net 5.7M\nEquivariant Imaging", ACCENT_SOFT),
             ("복원", "24.79 dB", ACCENT_SOFT)]
    for i, (h1, h2, fill) in enumerate(boxes):
        card(sl, xs, Inches(1.95), Inches(2.05), Inches(1.6), fill,
             ACCENT if fill == ACCENT_SOFT else RULE)
        text(sl, xs + Inches(0.08), Inches(2.08), Inches(1.9), Inches(0.5),
             h1, size=12, color=INK, bold=True, align=PP_ALIGN.CENTER)
        text(sl, xs + Inches(0.08), Inches(2.6), Inches(1.9), Inches(0.85),
             h2, size=10, color=MUTED, align=PP_ALIGN.CENTER, font=MONO)
        xs += Inches(2.4)
        if i < len(boxes) - 1:
            arrow(sl, xs - Inches(0.31), Inches(2.62), Inches(0.24), Inches(0.26))

    rows = [("", "Stage 1 — 디노이저", "Stage 2 — 역산"),
            ("구조", "U-Net (chans 24, pool 3)", "U-Net + Dilated Bottleneck (32, pool 4)"),
            ("입력", "measure + impulse mask (2채널)",
             "Wiener·TKD·median·불일치맵·mask (6채널)"),
            ("역할", "노이즈만 제거. 흐림은 그대로", "denoised blur → 최종 복원"),
            ("파라미터", "479,617", "5,724,801"),
            ("학습", "Noise2Noise", "Equivariant Imaging")]
    table(sl, Inches(0.9), Inches(3.85), Inches(11.5), Inches(2.2), rows,
          col_w=[1.4, 4.6, 5.5], size=11)
    footer(sl, "학습 loss 어디에도 label 이 들어가지 않는다 — 검증과 최종 채점에만 쓴다")
    if name:
        footer(sl, name)

    # ============================================================ 2. Stage 1
    sl = blank(prs); bg(sl)
    title(sl, "Stage 1 — Noise2Noise. 깨끗한 정답 대신 다른 노이즈를",
          "같은 흐림에서 노이즈만 다르게 두 장을 만들어, 하나로 다른 하나를 맞힌다",
          eyebrow="자기지도 ①")
    card(sl, Inches(2.2), Inches(1.85), Inches(8.8), Inches(0.95), ACCENT_SOFT, ACCENT)
    text(sl, Inches(2.35), Inches(2.0), Inches(8.5), Inches(0.7),
         [("y₁, y₂ = 같은 blur + 독립적인 노이즈 두 실현", {"font": MONO, "size": 13,
                                                    "color": INK, "bold": True}),
          ("loss = Charbonnier( f(y₁),  y₂ )", {"font": MONO, "size": 13,
                                                "color": ACCENT, "bold": True})],
         size=13, color=INK2, align=PP_ALIGN.CENTER)

    ys = Inches(3.05)
    for head, body in [
        ("왜 노이즈를 정답으로 써도 되나",
         "노이즈는 평균이 0 이고 두 실현이 독립이다. 그래서 y₂ 를 맞히려는 최적해가 "
         "곧 노이즈 없는 신호다 — E[y₂ | y₁] = 신호."),
        ("Charbonnier(L1)를 쓰는 이유",
         "salt & pepper 는 임펄스라 평균이 끌려간다. L1 은 조건부 중앙값으로 수렴해 "
         "그 편향을 피한다. 실제로 s&p 가 가장 잘 나온다 (30.09 dB)."),
        ("rician 만 따로 보정",
         "정류 때문에 평균이 0 이 아니다. 타깃을 √(max(y² − 2σ², 0)) 로 바꿔 "
         "moment 기준으로 편향을 뺀다."),
    ]:
        card(sl, Inches(0.9), ys, Inches(11.5), Inches(0.95))
        text(sl, Inches(1.15), ys + Inches(0.1), Inches(11), Inches(0.3),
             head, size=13, color=ACCENT, bold=True)
        text(sl, Inches(1.15), ys + Inches(0.42), Inches(11), Inches(0.45),
             body, size=11.5, color=MUTED)
        ys += Inches(1.05)
    footer(sl, "Noise2Noise — Lehtinen et al., ICML 2018.  깨끗한 정답 없이 노이즈 쌍만으로 학습한다")

    # ============================================================ 3. Stage 2
    sl = blank(prs); bg(sl)
    title(sl, "Stage 2 — Equivariant Imaging. 측정으로 못 본 곳을 채운다",
          "이미지를 돌려도 여전히 같은 종류의 이미지여야 한다는 제약을 건다",
          eyebrow="자기지도 ②")
    card(sl, Inches(1.6), Inches(1.8), Inches(10.0), Inches(1.35), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.8), Inches(1.95), Inches(9.6), Inches(1.1),
         [("x₁   = g(hint(denoised))", {"font": MONO, "size": 12, "color": INK}),
          ("loss_mc = ‖A(x₁) − y₃‖                  측정치와 맞아야 한다",
           {"font": MONO, "size": 12, "color": INK}),
          ("x₁ᵀ  = rot90(x₁),  x₂ = g(hint(A(x₁ᵀ)))", {"font": MONO, "size": 12, "color": INK}),
          ("loss_eq = ‖x₂ − x₁ᵀ‖                    돌려도 같아야 한다",
           {"font": MONO, "size": 12, "color": ACCENT, "bold": True})],
         size=12, color=INK2)

    ys = Inches(3.4)
    for head, body in [
        ("왜 이것이 원뿔을 메우나",
         "측정만으로는 매직앵글 원뿔의 정보를 얻을 수 없다. 그런데 이미지를 돌리면 "
         "그 원뿔이 이미지의 다른 방향과 겹친다 — 돌린 것도 같은 이미지여야 한다는 "
         "제약이 원뿔 안쪽에 대한 정보를 만들어 준다."),
        ("왜 90도 회전만 쓰나",
         "뒤집기는 dipole 커널과 교환된다. 교환되면 A(Tx) 가 새로운 측정 방향을 주지 "
         "못해 얻는 것이 없다. 90도 회전은 교환되지 않으므로 쓸 수 있다."),
        ("λ_eq 스케줄",
         "0 → 1.0 (3 epoch warmup) → 0.2 (6 epoch부터 감소). 초반엔 등변성을 세게 걸어 "
         "원뿔을 채우고, 후반엔 낮춰 측정 충실도 위주로 안정화한다."),
    ]:
        card(sl, Inches(0.9), ys, Inches(11.5), Inches(1.0))
        text(sl, Inches(1.15), ys + Inches(0.1), Inches(11), Inches(0.3),
             head, size=13, color=ACCENT, bold=True)
        text(sl, Inches(1.15), ys + Inches(0.42), Inches(11), Inches(0.5),
             body, size=11.5, color=MUTED)
        ys += Inches(1.1)
    footer(sl, "Equivariant Imaging — Chen et al., ICCV 2021.  "
               "그 밖에 rician 가중치 4배 · EMA(0.999) · early stopping(patience 15)")

    # ============================================================ 4. 결과
    sl = blank(prs); bg(sl)
    title(sl, "결과 — classical 최고보다 +9.7 dB",
          "test_deconv_noise 100장 · TTA = 항등 + 좌우 + 상하 + 180°", eyebrow="결과")
    rows = [("방법", "PSNR", "SSIM"),
            ("측정치 (입력)", "8.16", "−0.014"),
            ("Mean / Median / Adaptive 필터", "8.3 안팎", "0.05~0.06"),
            ("TKD (clip=5)", "14.93", "0.361"),
            ("Wiener (K=1e-2)  ← classical 최고", "15.10", "0.385"),
            ("Self-supervised (N2N + EI)", "24.55", "0.781"),
            ("Self-supervised + TTA", "24.79", "0.787")]
    table(sl, Inches(0.9), Inches(1.8), Inches(6.5), Inches(2.4), rows,
          col_w=[4.1, 1.2, 1.2], highlight_row=len(rows) - 1)

    rows = [("노이즈", "PSNR", "SSIM"),
            ("salt & pepper", "30.09", "0.943"),
            ("gaussian", "26.06", "0.806"),
            ("uniform", "25.60", "0.845"),
            ("rician", "17.40", "0.554")]
    table(sl, Inches(7.9), Inches(1.8), Inches(4.5), Inches(1.7), rows,
          col_w=[2.1, 1.2, 1.2])
    text(sl, Inches(7.9), Inches(3.65), Inches(4.5), Inches(0.5),
         "L1 이 임펄스에 강하다는 이론이 s&p 30.09 로 확인된다. rician 은 8~13 dB 뒤진다.",
         size=10.5, color=MUTED)

    card(sl, Inches(0.9), Inches(4.45), Inches(11.5), Inches(1.9), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.15), Inches(4.62), Inches(11), Inches(1.6),
         [("데이터 효율성 — 1000장(전체의 13.8%)만 써도 23.55 dB.", B),
          ("전체 7268장(24.79)과 1.2 dB 밖에 차이나지 않는다. gradient step 수를 고정해 "
           "공정하게 비교한 값이다.", N),
          ("", N),
          ("라벨이 필요 없으니 데이터를 모으기도 쉽고, 적은 데이터로도 버틴다 — "
           "자기지도 방식의 값어치가 여기에 있다.", B)], size=12.5, color=INK2)
    footer(sl, f"참고: 지도학습 최종은 {SUP:.2f} dB / 0.8900.  "
               f"라벨을 쓰지 않은 값으로는 classical({WIENER:.2f}) 대비 +{24.79 - WIENER:.1f} dB 다")

    # ============================================================ 5. 한계
    sl = blank(prs); bg(sl)
    title(sl, "한계와 논점", "무엇이 남았고, 무엇을 배웠나", eyebrow="논의")
    ys = Inches(1.85)
    for head, body, val in [
        ("rician 이 구조적으로 어렵다",
         "편향 보정과 4배 가중치를 줬는데도 다른 셋보다 8~13 dB 낮다. 진폭 도메인의 "
         "비선형·비가우시안 노이즈라 '노이즈는 예측 불가' 라는 자기지도의 전제가 약해진다.",
         "17.40 dB\n다음 셋은 25~30"),
        ("TTA 는 모든 조건에서 +0.2~0.3 dB",
         "재학습 없이 얻는 개선. 전체 budget 에서도 1000장 budget 에서도 일관되게 나온다.",
         "24.55\n→ 24.79"),
        ("salt & pepper 가 가장 쉽다",
         "Charbonnier(L1)가 조건부 중앙값으로 수렴한다는 이론이 임펄스 노이즈에 정확히 "
         "들어맞는다는 것이 실측으로 확인됐다.",
         "30.09 dB\n최고"),
        ("지도학습과의 격차",
         f"지도학습 최종 {SUP:.2f} dB 와는 아직 {SUP - 24.79:.1f} dB 차이가 난다. 다만 "
         "라벨을 한 장도 쓰지 않았다는 것이 이 트랙의 목표였다.",
         f"24.79\nvs {SUP:.2f}"),
    ]:
        card(sl, Inches(0.9), ys, Inches(11.5), Inches(1.05))
        text(sl, Inches(1.15), ys + Inches(0.1), Inches(7.6), Inches(0.3),
             head, size=13, color=INK, bold=True)
        text(sl, Inches(1.15), ys + Inches(0.42), Inches(7.6), Inches(0.55),
             body, size=11, color=MUTED)
        text(sl, Inches(9.1), ys + Inches(0.2), Inches(3.1), Inches(0.7),
             val, size=12, color=ACCENT, bold=True, font=MONO, align=PP_ALIGN.CENTER)
        ys += Inches(1.15)

    card(sl, Inches(0.9), Inches(6.5), Inches(11.5), Inches(0.75), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.15), Inches(6.63), Inches(11), Inches(0.5),
         "같은 사실이 정반대로 쓰인다 — self-ensemble 은 커널과 교환되는 변환(뒤집기)만, "
         "Equivariant Imaging 은 교환되지 않는 변환(90° 회전)만 쓴다.",
         size=12.5, color=INK, bold=True)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--out", type=Path, default=ROOT / "실습5_day3_selfsupervised.pptx")
    args = ap.parse_args()

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    build(prs, args.name)
    prs.save(str(args.out))
    print(f"슬라이드 {len(prs.slides._sldIdLst)}장 → {args.out}")


if __name__ == "__main__":
    main()
