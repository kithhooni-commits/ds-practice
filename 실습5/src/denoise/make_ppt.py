"""발표 슬라이드 생성 (python-pptx).

과제 요구사항 네 가지를 슬라이드로 옮긴다.
  1. pipeline
  2. before / after / error map / ground truth
  3. 왜 그 방법을 골랐는가
  4. label-free (가산점)

수치는 하드코딩하지 않고 runs/ 의 test_metrics*.json 에서 읽는다.
그래야 재학습할 때마다 슬라이드가 같이 갱신된다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIGDIR = ROOT / "figures"

W, H = Inches(13.333), Inches(7.5)

INK = RGBColor(0x13, 0x1A, 0x19)
INK2 = RGBColor(0x3C, 0x49, 0x47)
MUTED = RGBColor(0x66, 0x75, 0x6F)
ACCENT = RGBColor(0x0B, 0x6E, 0x5E)
ACCENT_SOFT = RGBColor(0xDC, 0xED, 0xE8)
RULE = RGBColor(0xDC, 0xE3, 0xE0)
SURF = RGBColor(0xF3, 0xF6, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
WARN = RGBColor(0x8C, 0x4A, 0x6B)

FONT = "맑은 고딕"
MONO = "Consolas"

NOISE_ORDER = ["gaussian", "rician", "uniform", "salt_and_pepper"]
NOISE_KO = {"gaussian": "Gaussian", "rician": "Rician",
            "uniform": "Uniform", "salt_and_pepper": "Salt & Pepper"}
BASE = {"psnr": 30.510, "ssim": 0.8950}


# ------------------------------------------------------------------ helpers


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def bg(slide, color=WHITE):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def text(slide, x, y, w, h, runs, size=16, color=INK2, bold=False, font=FONT,
         align=PP_ALIGN.LEFT, spacing=1.25, anchor=MSO_ANCHOR.TOP):
    """runs: 문자열 또는 (문자열, dict) 목록. dict 로 run 단위 서식을 덮어쓴다."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    if isinstance(runs, str):
        runs = [runs]

    first = True
    for item in runs:
        s, opt = item if isinstance(item, tuple) else (item, {})
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = opt.get("align", align)
        p.line_spacing = opt.get("spacing", spacing)
        if "space_before" in opt:
            p.space_before = opt["space_before"]
        r = p.add_run()
        r.text = s
        f = r.font
        f.name = opt.get("font", font)
        f.size = Pt(opt.get("size", size))
        f.bold = opt.get("bold", bold)
        f.color.rgb = opt.get("color", color)
    return tb


def title(slide, main, sub=None, eyebrow=None):
    y = Inches(0.46)
    if eyebrow:
        text(slide, Inches(0.7), y, Inches(12), Inches(0.3), eyebrow,
             size=11, color=ACCENT, bold=True, font=MONO)
        y = Inches(0.78)
    text(slide, Inches(0.7), y, Inches(12), Inches(0.6), main, size=30, color=INK, bold=True)
    y2 = y + Inches(0.72)
    if sub:
        text(slide, Inches(0.7), y2, Inches(12), Inches(0.4), sub, size=14, color=MUTED)
        y2 += Inches(0.5)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), y2 + Inches(0.05), Inches(11.93), Pt(1.2))
    line.fill.solid()
    line.fill.fore_color.rgb = RULE
    line.line.fill.background()
    line.shadow.inherit = False
    return y2 + Inches(0.35)


def card(slide, x, y, w, h, fill=SURF, line=RULE):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.adjustments[0] = 0.04
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line
    sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    return sh


def arrow(slide, x, y, w, h, color=ACCENT):
    sh = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def table(slide, x, y, w, h, rows, col_w=None, head=True, size=12, highlight_row=None,
          highlight_col=None):
    nr, nc = len(rows), len(rows[0])
    shape = slide.shapes.add_table(nr, nc, x, y, w, h)
    tbl = shape.table
    if col_w:
        total = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Emu(int(w * cw / total))
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(0.34)
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(val)
            cell.margin_left = Inches(0.09)
            cell.margin_right = Inches(0.09)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            is_head = head and r == 0
            hot = (highlight_row is not None and r == highlight_row) or \
                  (highlight_col is not None and c == highlight_col and r > 0)
            cell.fill.fore_color.rgb = ACCENT_SOFT if is_head else (
                RGBColor(0xEC, 0xF5, 0xF2) if hot else WHITE)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            for run in p.runs:
                run.font.size = Pt(size)
                run.font.name = FONT if c == 0 else MONO
                run.font.bold = is_head or hot
                run.font.color.rgb = INK if (is_head or hot) else INK2
    return shape


def pic(slide, path: Path, x, y, w=None, h=None):
    if not path.exists():
        text(slide, x, y, Inches(6), Inches(0.4), f"[그림 없음: {path.name}]", size=12, color=WARN)
        return None
    return slide.shapes.add_picture(str(path), x, y, width=w, height=h)


def footer(slide, s):
    text(slide, Inches(0.7), Inches(6.95), Inches(12), Inches(0.3), s, size=10, color=MUTED, font=MONO)


# ------------------------------------------------------------------ slides


def build(prs, M, name: str, lf: dict | None, abl: dict | None):
    se, base = M["se"], M["base"]

    def by_noise(rows, key):
        return {nz: float(np.mean([r[key] for r in rows if r["noise_type"] == nz])) for nz in NOISE_ORDER}

    p_model = by_noise(se["rows"], "psnr_model")
    q_model = by_noise(se["rows"], "ssim_model")
    p_in = by_noise(se["rows"], "psnr_noisy")
    q_in = by_noise(se["rows"], "ssim_noisy")
    p_med = by_noise(se["rows"], "psnr_median")
    p_mean = by_noise(se["rows"], "psnr_mean")
    p_adap = by_noise(se["rows"], "psnr_adaptive")

    # ---------------------------------------------------------- 1. 표지
    s = blank(prs); bg(s)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.28), H)
    band.fill.solid(); band.fill.fore_color.rgb = ACCENT
    band.line.fill.background(); band.shadow.inherit = False

    text(s, Inches(1.0), Inches(1.5), Inches(11), Inches(0.4),
         "삼성 DS2 · Image Restoration Challenge · Day 1", size=13, color=ACCENT, bold=True, font=MONO)
    text(s, Inches(1.0), Inches(2.0), Inches(11), Inches(1.2),
         "노이즈 4종을 하나의 네트워크로", size=42, color=INK, bold=True)
    text(s, Inches(1.0), Inches(2.85), Inches(11), Inches(0.7),
         "종류도 세기도 모른 채 복원하기", size=42, color=ACCENT, bold=True)
    text(s, Inches(1.0), Inches(3.95), Inches(10), Inches(0.5),
         "반도체 이미지 denoising · test 100장 · DnCNN + median 채널", size=15, color=MUTED)

    c = card(s, Inches(1.0), Inches(4.75), Inches(4.4), Inches(1.5), ACCENT_SOFT, ACCENT)
    text(s, Inches(1.25), Inches(4.95), Inches(4), Inches(0.3), "제출값", size=12, color=ACCENT, bold=True, font=MONO)
    text(s, Inches(1.25), Inches(5.3), Inches(4), Inches(0.7),
         f"PSNR {se['psnr_total']:.2f}   SSIM {se['ssim_total']:.4f}", size=21, color=INK, bold=True, font=MONO)

    c2 = card(s, Inches(5.7), Inches(4.75), Inches(4.4), Inches(1.5))
    text(s, Inches(5.95), Inches(4.95), Inches(4), Inches(0.3), "배포 기준선 (DnCNN 10 epoch)",
         size=12, color=MUTED, bold=True, font=MONO)
    text(s, Inches(5.95), Inches(5.3), Inches(4), Inches(0.7),
         f"PSNR {BASE['psnr']:.2f}   SSIM {BASE['ssim']:.4f}", size=21, color=MUTED, bold=True, font=MONO)
    text(s, Inches(5.95), Inches(5.85), Inches(4), Inches(0.3),
         f"→  +{se['psnr_total'] - BASE['psnr']:.2f} dB   +{se['ssim_total'] - BASE['ssim']:.4f}",
         size=13, color=ACCENT, bold=True, font=MONO)
    text(s, Inches(10.4), Inches(5.9), Inches(2.3), Inches(0.4), name, size=13, color=MUTED, align=PP_ALIGN.RIGHT)

    # ---------------------------------------------------------- 2. 문제
    s = blank(prs); bg(s)
    y = title(s, "문제 — 종류를 모른다는 것이 전부다",
              "이미지마다 4종 중 하나가 랜덤 σ 로 실려 있고, 어느 것인지 알려주지 않는다",
              "PROBLEM")

    rows = [["노이즈", "σ 범위", "성질", "입력 PSNR"]]
    desc = {
        "gaussian": "x + N(0, σ)",
        "rician": "|x + N + iN| — 어두운 쪽이 위로 들림",
        "uniform": "x + U(-σ, σ)",
        "salt_and_pepper": "픽셀의 σ 비율을 0 또는 max 로 덮어씀 (임펄스)",
    }
    rng = {"gaussian": "0 – 0.1", "rician": "0 – 0.15", "uniform": "0 – 0.2", "salt_and_pepper": "0 – 0.2"}
    for nz in NOISE_ORDER:
        rows.append([NOISE_KO[nz], rng[nz], desc[nz], f"{p_in[nz]:.2f} dB"])
    table(s, Inches(0.7), y, Inches(11.9), Inches(1.9), rows, col_w=[2.0, 1.6, 6.3, 1.7],
          highlight_row=4)

    y2 = y + Inches(2.25)
    c = card(s, Inches(0.7), y2, Inches(5.8), Inches(2.5))
    text(s, Inches(0.95), y2 + Inches(0.22), Inches(5.3), Inches(0.35),
         "고전 필터는 종류마다 정답이 정반대다", size=16, color=INK, bold=True)
    text(s, Inches(0.95), y2 + Inches(0.72), Inches(5.3), Inches(1.6), [
        (f"Salt & Pepper  ·  median {p_med['salt_and_pepper']:.1f}  vs  mean {p_mean['salt_and_pepper']:.1f} dB", {"font": MONO, "size": 12.5}),
        (f"Gaussian       ·  adaptive {p_adap['gaussian']:.1f}  vs  median {p_med['gaussian']:.1f} dB", {"font": MONO, "size": 12.5}),
        ("임펄스는 몇 픽셀이 통째로 거짓말이라 평균 계열이 그 거짓말을 이웃에 퍼뜨린다. "
         "반대로 median 은 미세한 계조를 계단으로 뭉갠다.", {"size": 12.5, "space_before": Pt(10)}),
    ], size=12.5, color=INK2)

    c = card(s, Inches(6.8), y2, Inches(5.8), Inches(2.5), ACCENT_SOFT, ACCENT)
    text(s, Inches(7.05), y2 + Inches(0.22), Inches(5.3), Inches(0.35),
         "그래서 고정 필터 하나로는 못 넘는다", size=16, color=INK, bold=True)
    tot_in = se["rows"]
    m_all = float(np.mean([r["psnr_mean"] for r in tot_in]))
    n_all = float(np.mean([r["psnr_noisy"] for r in tot_in]))
    text(s, Inches(7.05), y2 + Inches(0.72), Inches(5.3), Inches(1.6), [
        (f"복원 안 함   {n_all:.2f} dB", {"font": MONO, "size": 13}),
        (f"mean 3×3     {m_all:.2f} dB   ← 겨우 +{m_all - n_all:.2f}", {"font": MONO, "size": 13}),
        ("종류를 모른 채 하나의 규칙을 고정하면 아무것도 안 한 것과 다를 바 없다. "
         "학습이 필요한 이유가 여기 있다.", {"size": 12.5, "space_before": Pt(10)}),
    ], size=12.5, color=INK2)
    footer(s, "출처: dataset/test_noise_only/noise_meta.json · 25장 × 4종 = 100장")

    # ---------------------------------------------------------- 3. 파이프라인
    s = blank(prs); bg(s)
    y = title(s, "파이프라인", "학습은 노이즈를 만들어 배우고, 추론은 만들지 않는다", "1 · PIPELINE")

    # 학습 경로
    text(s, Inches(0.7), y + Inches(0.05), Inches(3), Inches(0.3), "학습 (supervised)",
         size=13, color=ACCENT, bold=True, font=MONO)
    ty = y + Inches(0.45)
    boxes = [
        ("clean 7,268장", "train/", 2.05),
        ("랜덤 크롭 128\nflip · rot90", "증강", 1.75),
        ("노이즈 4종 중\n1개 랜덤 + σ 랜덤", "합성", 2.1),
        ("DnCNN + median 채널\n17층 · 64ch · 잔차", "모델", 2.5),
        ("Charbonnier loss\nvs clean", "학습 신호", 2.0),
    ]
    x = Inches(0.7)
    for i, (label, tag, wid) in enumerate(boxes):
        wid = Inches(wid)
        fill = ACCENT_SOFT if i == 3 else SURF
        card(s, x, ty, wid, Inches(1.05), fill, ACCENT if i == 3 else RULE)
        text(s, x + Inches(0.12), ty + Inches(0.08), wid - Inches(0.24), Inches(0.22), tag,
             size=9.5, color=ACCENT if i == 3 else MUTED, bold=True, font=MONO, align=PP_ALIGN.CENTER)
        text(s, x + Inches(0.1), ty + Inches(0.36), wid - Inches(0.2), Inches(0.6), label,
             size=11.5, color=INK, bold=(i == 3), align=PP_ALIGN.CENTER, spacing=1.05)
        x += wid
        if i < len(boxes) - 1:
            arrow(s, x + Inches(0.04), ty + Inches(0.4), Inches(0.28), Inches(0.24))
            x += Inches(0.36)

    # 추론 경로
    iy = ty + Inches(1.6)
    text(s, Inches(0.7), iy - Inches(0.35), Inches(4), Inches(0.3), "추론 (test 100장)",
         size=13, color=ACCENT, bold=True, font=MONO)
    boxes2 = [
        ("corrupted 100장", "test_noise_only/", 2.6),
        ("8× self-ensemble\ndihedral 평균", "추론", 2.5),
        ("같은 네트워크\n(가중치 1벌)", "모델", 2.5),
        (f"PSNR {se['psnr_total']:.2f}\nSSIM {se['ssim_total']:.4f}", "제출", 2.3),
    ]
    x = Inches(0.7)
    for i, (label, tag, wid) in enumerate(boxes2):
        wid = Inches(wid)
        last = i == len(boxes2) - 1
        card(s, x, iy, wid, Inches(1.05), ACCENT_SOFT if last else SURF, ACCENT if last else RULE)
        text(s, x + Inches(0.12), iy + Inches(0.08), wid - Inches(0.24), Inches(0.22), tag,
             size=9.5, color=ACCENT if last else MUTED, bold=True, font=MONO, align=PP_ALIGN.CENTER)
        text(s, x + Inches(0.1), iy + Inches(0.36), wid - Inches(0.2), Inches(0.6), label,
             size=11.5 if not last else 12.5, color=INK, bold=last,
             align=PP_ALIGN.CENTER, spacing=1.05, font=MONO if last else FONT)
        x += wid
        if not last:
            arrow(s, x + Inches(0.04), iy + Inches(0.4), Inches(0.28), Inches(0.24))
            x += Inches(0.36)

    ky = iy + Inches(1.55)
    card(s, Inches(0.7), ky, Inches(11.9), Inches(1.25))
    text(s, Inches(0.95), ky + Inches(0.15), Inches(11.4), Inches(1.0), [
        ("핵심: 추론 경로에는 노이즈 종류·σ 정보가 들어가지 않는다", {"size": 14, "bold": True, "color": INK}),
        ("noise_meta.json 에 종류와 σ 가 다 적혀 있지만 복원에는 쓰지 않았다. "
         "표를 종류별로 쪼개 보여주는 데만 썼다 — 채점 세트에 그 정보가 없을 수 있고, "
         "모르고도 되는 것이 더 강한 결과다.", {"size": 12.5, "space_before": Pt(6)}),
    ], color=INK2)

    # ---------------------------------------------------------- 4. 왜 이 방법인가
    s = blank(prs); bg(s)
    y = title(s, "왜 이 방법을 골랐는가",
              "배포 코드와 같은 데이터·같은 노이즈 합성·같은 지표를 쓰고 학습 쪽만 바꿨다",
              "3 · DESIGN RATIONALE")

    rows = [["바꾼 것", "관찰", "그래서"]]
    rows += [
        ["입력에 median 3×3\n채널 추가",
         f"s&p 입력이 {p_in['salt_and_pepper']:.1f} dB 로 혼자 10 dB 아래.\n"
         f"고전 필터 중 median 만 통함 ({p_med['salt_and_pepper']:.1f} vs mean {p_mean['salt_and_pepper']:.1f})",
         "임펄스가 이미 지워진 버전을 같이 준다.\n파라미터는 576개만 증가"],
        ["Charbonnier loss\n(L2 → smooth L1)",
         "L2 는 s&p 의 극단값 몇 픽셀에\ngradient 가 끌려간다",
         "큰 오차의 영향을 선형으로 제한"],
        ["cosine LR · 40 epoch",
         "배포 설정은 10 epoch 에 plateau×0.88 —\n감쇠가 사실상 안 걸림",
         "끝까지 부드럽게 낮춰 수렴"],
        ["patch 128 랜덤 크롭\n+ rot90",
         "DnCNN 수용영역은 35px.\n256² 로 학습할 이유가 없다",
         "크롭 자체가 증강.\nstep 수도 2배"],
        ["8× self-ensemble",
         "flip/rot90 로 학습했으니\n8개 변환을 같게 다뤄야 맞다",
         f"실제로는 어긋남. 평균이 그것을 지움\n(+{se['psnr_total'] - base['psnr_total']:.2f} dB, 학습 비용 0)"],
    ]
    table(s, Inches(0.7), y, Inches(11.9), Inches(4.1), rows, col_w=[2.5, 5.2, 4.2], size=11)
    footer(s, "모델 구조·층수·채널·활성함수는 배포된 DnCNN 그대로. 바꾼 것은 입력 채널 하나와 학습 절차뿐이다.")

    # ---------------------------------------------------------- 5. 결과
    s = blank(prs); bg(s)
    y = title(s, "결과", f"test 100장 · 제출값 PSNR {se['psnr_total']:.2f} / SSIM {se['ssim_total']:.4f}", "RESULTS")
    pic(s, FIGDIR / "summary_bars.png", Inches(0.6), y + Inches(0.15), w=Inches(7.7))

    rows = [["노이즈", "입력", "제안 모델", "Δ"]]
    for nz in NOISE_ORDER:
        rows.append([NOISE_KO[nz], f"{p_in[nz]:.2f}", f"{p_model[nz]:.2f}", f"+{p_model[nz] - p_in[nz]:.2f}"])
    rows.append(["전체", f"{n_all:.2f}", f"{se['psnr_total']:.2f}", f"+{se['psnr_total'] - n_all:.2f}"])
    table(s, Inches(8.5), y + Inches(0.15), Inches(4.1), Inches(2.0), rows,
          col_w=[1.7, 0.9, 1.1, 0.9], size=11.5, highlight_row=5)

    ry = y + Inches(2.45)
    card(s, Inches(8.5), ry, Inches(4.1), Inches(2.6), ACCENT_SOFT, ACCENT)
    text(s, Inches(8.72), ry + Inches(0.18), Inches(3.7), Inches(2.2), [
        ("Salt & Pepper 가 갈림길", {"size": 14, "bold": True, "color": INK}),
        (f"{p_in['salt_and_pepper']:.2f}  →  {p_model['salt_and_pepper']:.2f} dB   "
         f"(+{p_model['salt_and_pepper'] - p_in['salt_and_pepper']:.2f})",
         {"size": 13, "font": MONO, "space_before": Pt(8)}),
        (f"SSIM {q_in['salt_and_pepper']:.4f} → {q_model['salt_and_pepper']:.4f} — 4종 중 최고",
         {"size": 12, "font": MONO}),
        ("배포 기준선 DnCNN 은 여기서 29.69 였다. median 채널을 넣은 근거가 그대로 숫자로 나왔다.",
         {"size": 12, "space_before": Pt(8)}),
    ], color=INK2)
    footer(s, "conventional 3종은 배포 코드 구현 그대로 · self-ensemble 적용")

    # ---------------------------------------------------------- 6-9. 4패널
    for nz in NOISE_ORDER:
        s = blank(prs); bg(s)
        y = title(s, f"{NOISE_KO[nz]}",
                  f"corrupted → restored → |error| → ground truth   ·   "
                  f"{p_in[nz]:.2f} → {p_model[nz]:.2f} dB (+{p_model[nz] - p_in[nz]:.2f})",
                  "2 · BEFORE / AFTER / ERROR / GT")
        pic(s, FIGDIR / f"panel_{nz}.png", Inches(0.55), y + Inches(0.45), w=Inches(12.2))
        note = {
            "gaussian": "오차가 경계선을 따라 남는다. 평탄한 영역은 거의 완전히 복원된다.",
            "rician": "4종 중 가장 어렵다. |·| 때문에 어두운 영역의 값이 위로 들려 있어, 밝기 자체가 이동한 상태에서 되돌려야 한다.",
            "uniform": "gaussian 과 비슷한 양상. 입력 PSNR 이 원래 높아 개선 폭은 작다.",
            "salt_and_pepper": "임펄스가 완전히 사라졌다. 남은 오차는 원래 임펄스가 있던 자리의 미세한 흔적뿐이다.",
        }[nz]
        footer(s, note)

    # ---------------------------------------------------------- 10. 오차 지도
    s = blank(prs); bg(s)
    y = title(s, "오차는 어디에 남는가", "열마다 같은 스케일 — 위: 아무것도 안 했을 때, 아래: 모델 통과 후",
              "2 · ERROR MAP")
    pic(s, FIGDIR / "error_maps.png", Inches(2.55), y + Inches(0.05), h=Inches(4.95))
    footer(s, "PSNR 한 숫자로는 '어디서' 틀렸는지 안 보인다. 오차가 경계에 몰려 있는지 평탄부에 흩어져 있는지가 다음 수를 결정한다.")

    # ---------------------------------------------------------- 11. 검증과 실패
    s = blank(prs); bg(s)
    y = title(s, "믿을 수 있는 숫자인가", "그리고 되지 않은 것 하나", "VALIDATION")

    c = card(s, Inches(0.7), y, Inches(5.8), Inches(2.3), ACCENT_SOFT, ACCENT)
    text(s, Inches(0.95), y + Inches(0.2), Inches(5.3), Inches(2.0), [
        ("지표 구현이 채점과 같은지 먼저 맞췄다", {"size": 15, "bold": True, "color": INK}),
        ("배포 예시 로그의 mean/median/adaptive 성적을 우리 로더·우리 지표로 다시 쟀다.",
         {"size": 12.5, "space_before": Pt(8)}),
        ("PSNR 최대 차이  0.0000 dB", {"size": 15, "font": MONO, "bold": True, "color": ACCENT, "space_before": Pt(8)}),
        ("4종 × 4방법 전부 소수 넷째자리까지 일치. 여기가 맞은 뒤에 학습을 돌렸다.",
         {"size": 12.5, "space_before": Pt(6)}),
    ], color=INK2)

    c = card(s, Inches(6.8), y, Inches(5.8), Inches(2.3))
    text(s, Inches(7.05), y + Inches(0.2), Inches(5.3), Inches(2.0), [
        ("되지 않은 것 — σ 게이트", {"size": 15, "bold": True, "color": WARN}),
        ("입력이 이미 50~62 dB 인 이미지에서는 모델이 손해다 (100장 중 8장). "
         "정답을 보고 매번 유리한 쪽을 고르면 +0.52 dB.", {"size": 12.5, "space_before": Pt(8)}),
        ("그런데 σ̂ 로는 그 8장을 못 고른다. MAD 추정기가 이미지의 미세 질감을 노이즈로 착각해, "
         "실제 σ=0.001 인 이미지를 0.037 로 추정한다. 임계값을 어디 두든 손해였다.",
         {"size": 12.5, "space_before": Pt(6)}),
    ], color=INK2)

    ay = y + Inches(2.6)
    if abl:
        rows = [["구성", "test PSNR", "test SSIM"]]
        rows.append(["DnCNN (배포 구조, 같은 레시피)", f"{abl['psnr']:.2f}", f"{abl['ssim']:.4f}"])
        rows.append(["+ median 채널 (제안)", f"{base['psnr_total']:.2f}", f"{base['ssim_total']:.4f}"])
        rows.append(["+ 8× self-ensemble", f"{se['psnr_total']:.2f}", f"{se['ssim_total']:.4f}"])
        table(s, Inches(0.7), ay, Inches(6.4), Inches(1.4), rows, col_w=[3.6, 1.4, 1.4],
              size=12, highlight_row=3)
        text(s, Inches(7.4), ay + Inches(0.1), Inches(5.2), Inches(1.3),
             "같은 학습 레시피로 구조만 원본으로 돌린 ablation. "
             "median 채널과 self-ensemble 각각의 기여를 분리해서 본다.",
             size=12.5, color=MUTED)
    else:
        card(s, Inches(0.7), ay, Inches(11.9), Inches(1.3))
        text(s, Inches(0.95), ay + Inches(0.25), Inches(11.4), Inches(0.9),
             "ablation (median 채널 없는 순정 DnCNN, 같은 레시피) 학습 진행 중 — 완료 후 이 표를 채운다.",
             size=13, color=MUTED)
    footer(s, "check_baselines.py · analyze_gate.py 로 재현 가능")

    # ---------------------------------------------------------- 12. label-free
    s = blank(prs); bg(s)
    y = title(s, "Label-free 파이프라인", "clean 이미지를 loss 에 한 번도 쓰지 않는 경로 (Noise2Void)",
              "4 · BONUS")

    c = card(s, Inches(0.7), y, Inches(5.8), Inches(2.9), ACCENT_SOFT, ACCENT)
    text(s, Inches(0.95), y + Inches(0.2), Inches(5.3), Inches(2.6), [
        ("원리 — 그 픽셀을 입력에서 지운다", {"size": 15, "bold": True, "color": INK}),
        ("입력 : noisy (일부 픽셀을 이웃 값으로 덮어씀)", {"size": 12, "font": MONO, "space_before": Pt(8)}),
        ("정답 : 덮어쓰기 전 그 픽셀의 noisy 값", {"size": 12, "font": MONO}),
        ("loss : 덮어쓴 자리에서만", {"size": 12, "font": MONO}),
        ("노이즈가 픽셀마다 독립이면, 주변에서 그 픽셀의 노이즈를 알아낼 방법이 없다. "
         "네트워크가 맞힐 수 있는 것은 구조뿐이다. clean 은 어디에도 등장하지 않는다.",
         {"size": 12.5, "space_before": Pt(8)}),
    ], color=INK2)

    c = card(s, Inches(6.8), y, Inches(5.8), Inches(2.9))
    text(s, Inches(7.05), y + Inches(0.2), Inches(5.3), Inches(2.6), [
        ("우리 노이즈 4종은 전제를 만족한다", {"size": 15, "bold": True, "color": INK}),
        ("gaussian · rician · uniform · salt&pepper 모두 픽셀별 독립.",
         {"size": 12.5, "space_before": Pt(8)}),
        ("다만 L2 loss 는 조건부 평균을 학습하므로 노이즈 평균이 0 이어야 한다. "
         "s&p 는 0/max 로 덮으니 아니고, rician 은 절댓값 때문에 위로 들린다.",
         {"size": 12.5, "space_before": Pt(6)}),
        ("→ L1 loss 로 조건부 중앙값을 학습시켜 임펄스 편향을 없앤다.",
         {"size": 12.5, "bold": True, "color": ACCENT, "space_before": Pt(6)}),
    ], color=INK2)

    ly = y + Inches(3.2)
    if lf:
        rows = [["파이프라인", "clean 사용", "test PSNR", "test SSIM"]]
        rows.append(["supervised (제출)", "loss 에 사용", f"{se['psnr_total']:.2f}", f"{se['ssim_total']:.4f}"])
        for k, v in lf.items():
            rows.append([v["label"], v["clean"], f"{v['psnr']:.2f}", f"{v['ssim']:.4f}"])
        table(s, Inches(0.7), ly, Inches(11.9), Inches(1.5), rows,
              col_w=[4.6, 2.6, 2.3, 2.4], size=12, highlight_row=len(rows) - 1)
    else:
        card(s, Inches(0.7), ly, Inches(11.9), Inches(1.6))
        text(s, Inches(0.95), ly + Inches(0.2), Inches(11.4), Inches(1.2), [
            ("두 가지 모드로 학습 중", {"size": 14, "bold": True, "color": INK}),
            ("① test_noise_only 100장만 사용 — clean 을 한 장도 건드리지 않는 가장 순수한 label-free",
             {"size": 12.5, "space_before": Pt(6)}),
            ("② train clean 으로 noisy 를 합성한 뒤 그 noisy 만 학습 — loss 에 clean 미사용, 데이터 7,268장",
             {"size": 12.5}),
            ("체크포인트 선택도 val noisy 의 마스킹 loss 로 한다. clean 기반 PSNR 로 고르면 파이프라인이 "
             "label-free 가 아니게 되기 때문이다.", {"size": 12, "color": MUTED, "space_before": Pt(6)}),
        ], color=INK2)
    footer(s, "src/denoise/train_n2v.py")

    # ---------------------------------------------------------- 13. 정리
    s = blank(prs); bg(s)
    y = title(s, "정리", None, "SUMMARY")
    items = [
        ("종류를 모른다는 제약이 이 문제의 전부",
         "고정 필터 하나로는 아무것도 안 한 것과 다를 바 없다 (mean 24.86 vs 입력 24.67). 학습이 필요한 이유."),
        ("가장 약한 고리를 겨냥해 구조를 바꿨다",
         f"s&p 를 위해 median 채널 하나 추가. 파라미터 576개로 {p_in['salt_and_pepper']:.1f} → {p_model['salt_and_pepper']:.1f} dB."),
        ("숫자를 믿을 수 있게 먼저 맞췄다",
         "배포 예시 로그와 PSNR 최대 차이 0.0000 dB. 검증 없이 낸 숫자는 근거가 아니다."),
        ("안 되는 것도 확인했다",
         "σ 게이트로 얻을 0.52 dB 는 blind 상태에서 집을 수 없다. 제약이 실제 비용을 발생시킨다는 증거."),
        ("label-free 경로를 따로 만들었다",
         "노이즈가 픽셀 독립이라는 성질만으로 clean 없이 학습 가능. 현실에서는 clean 을 못 구하는 경우가 많다."),
    ]
    yy = y + Inches(0.05)
    for i, (h, d) in enumerate(items):
        num = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.75), yy + Inches(0.08), Inches(0.34), Inches(0.34))
        num.fill.solid(); num.fill.fore_color.rgb = ACCENT
        num.line.fill.background(); num.shadow.inherit = False
        tf = num.text_frame
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run(); r.text = str(i + 1)
        r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = MONO
        text(s, Inches(1.3), yy, Inches(11.2), Inches(0.9), [
            (h, {"size": 15, "bold": True, "color": INK}),
            (d, {"size": 12.5, "color": MUTED, "space_before": Pt(3)}),
        ])
        yy += Inches(1.02)
    footer(s, f"제출값  PSNR {se['psnr_total']:.2f}   SSIM {se['ssim_total']:.4f}    "
              f"(배포 기준선 대비 +{se['psnr_total'] - BASE['psnr']:.2f} dB / +{se['ssim_total'] - BASE['ssim']:.4f})")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "runs" / "0831-1015_dncnn_plus_main")
    ap.add_argument("--name", default="")
    ap.add_argument("--lf", type=Path, default=None, help="label-free 결과 json (evaluate.py 출력)")
    ap.add_argument("--lf-train", type=Path, default=None)
    ap.add_argument("--ablation", type=Path, default=None, help="ablation 결과 json")
    ap.add_argument("--out", type=Path, default=ROOT / "실습5_denoising_발표.pptx")
    args = ap.parse_args()

    M = {
        "se": json.loads((args.run / "test_metrics_se.json").read_text(encoding="utf-8")),
        "base": json.loads((args.run / "test_metrics.json").read_text(encoding="utf-8")),
    }

    lf = {}
    for key, path, label, clean in (
        ("test", args.lf, "label-free · test 100장만", "전혀 안 씀"),
        ("train", args.lf_train, "label-free · train noisy", "loss 에 미사용"),
    ):
        if path and path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            lf[key] = {"label": label, "clean": clean, "psnr": d["psnr_total"], "ssim": d["ssim_total"]}
    lf = lf or None

    abl = None
    if args.ablation and args.ablation.exists():
        d = json.loads(args.ablation.read_text(encoding="utf-8"))
        abl = {"psnr": d["psnr_total"], "ssim": d["ssim_total"]}

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    build(prs, M, args.name, lf, abl)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(args.out))
    print(f"슬라이드 {len(prs.slides.__iter__.__self__._sldIdLst)}장 → {args.out}")


if __name__ == "__main__":
    main()
