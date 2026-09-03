"""3일차 발표 슬라이드 생성.

1일차 `make_ppt.py` 의 헬퍼(색·타이포·표·카드)를 그대로 재사용한다.

## 구성 — 시도를 축으로, 말하기 좋은 순서로

앞서는 "문제 → 설계 → 결과" 순이었는데 무엇을 해봤고 무엇이 늘었는지 따라가기
어려웠다. 이번에는 **여정** 을 3번에 파이프라인으로 먼저 깔고, 뒤 슬라이드가 그
단계를 하나씩 펼친다. 그림도 이야기 흐름에 맞춰 흩어 놓았다 — 열화 사슬은 문제를
설명하는 자리에, 복원 결과는 점수를 말한 직후에.

     1  표지                제출값
     2  목차
     3  여정                무엇을 바꿔 몇 dB 올랐나 — 이 발표의 지도
     4  문제                g = h*f + n · 기호 넷
     5  문제 (그림)          clean → blur → +noise
     6  시도 ①              1일차 + 2일차 조합. 왜 지는가
     7  시도 ① 한계          선형의 천장 21.95
     8  시도 ②              DC-Net. 2일차 최고 구조가 죽은 이유
     9  시도 ③              2단 분해. 전달 곡선과 실패
    10  우리 방법            전개형 파이프라인
    11  시도 ④              σ 조건화
    12  시도 ④ 근거          σ ablation
    13  시도 ⑤              안 통한 것들, 그리고 고쳐서 통한 것
    14  최종 결과            노이즈 종류별 · 비교
    15  복원 결과 (그림)      노이즈 4종 × 방법별
    16  복원 결과 (그림)      difference map + zoom
    17  취약점               어떤 노이즈·어떤 이미지
    18  보너스               label-free
    19  검증 규칙            test 는 채점에만

숫자는 실측이다. 다시 재면 `R` 과 `--psnr/--ssim/--lf-*` 만 바꾸면 된다.
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
    arrow, blank, bg, card, footer, pic, table, text, title,
)

FIG = ROOT / "figures"

# 조교가 공지한 배포 baseline (End2End U-Net, test 100장)
BASELINE, BASELINE_S = 25.02, 0.815
BAR_P, BAR_S = 26.0, 0.83            # 통과 기준

B = {"bold": True, "color": INK}     # 강조 문단
N: dict = {}                         # 보통 문단


def build(prs, name: str, M: dict, R: dict) -> None:
    p, s = M["psnr"], M["ssim"]
    gain = p - BASELINE

    def image_slide(fname, ttl, sub, foot, mode="wide", eyebrow="복원 결과"):
        sl = blank(prs); bg(sl)
        title(sl, ttl, sub, eyebrow=eyebrow)
        f = FIG / f"{fname}.png"
        if f.exists():
            if mode == "tall":
                pic(sl, f, Inches(1.8), Inches(1.55), h=Inches(5.3))
            else:
                pic(sl, f, Inches(0.8), Inches(2.1), w=Inches(11.7))
        footer(sl, foot)

    # ============================================================ 1. 표지
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
         "g = dipole(f) + n   ·   test_deconv_noise 100장", size=15, color=MUTED)

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

    # ============================================================ 2. 목차
    sl = blank(prs); bg(sl)
    title(sl, "목차", None, eyebrow="Contents")
    items = [("여정", "무엇을 바꿔 몇 dB 올랐나", "3"),
             ("문제", "흐림과 노이즈가 겹쳤다", "4–5"),
             ("시도 ①", "1일차 + 2일차 조합 — 왜 지는가 · 선형의 천장", "6–7"),
             ("시도 ②", "DC-Net — 2일차 최고 구조가 죽은 이유", "8"),
             ("시도 ③", "2단 분해 — 전달 곡선과 실패", "9"),
             ("우리 방법", "전개형 파이프라인", "10"),
             ("시도 ④", "σ 조건화 — 성능의 절반", "11–12"),
             ("시도 ⑤", "안 통한 것들, 그리고 고쳐서 통한 것", "13"),
             ("결과", "최종 점수 · 복원 이미지 · 취약점", "14–17"),
             ("보너스", "label-free — 정답을 한 장도 쓰지 않고", "18"),
             ("규칙", "test 는 채점에만 썼다", "19")]
    ys = Inches(1.62)
    for head, body, pg in items:
        card(sl, Inches(0.9), ys, Inches(11.5), Inches(0.44))
        text(sl, Inches(1.15), ys + Inches(0.06), Inches(1.8), Inches(0.34),
             head, size=12.5, color=ACCENT, bold=True)
        text(sl, Inches(3.1), ys + Inches(0.08), Inches(7.6), Inches(0.34),
             body, size=11.5, color=INK2)
        text(sl, Inches(11.0), ys + Inches(0.08), Inches(1.2), Inches(0.34),
             pg, size=11, color=MUTED, font=MONO, align=PP_ALIGN.RIGHT)
        ys += Inches(0.49)

    # ============================================================ 3. 여정
    sl = blank(prs); bg(sl)
    title(sl, "여정 — 무엇을 바꿔 무엇이 늘었나", "각 칸이 뒤에서 한 장씩 펼쳐진다",
          eyebrow="한눈에")
    steps = [("출발", "Wiener\n단독", "13.79"),
             ("① 조합", "1일차 디노이저\n→ Wiener", "21.06"),
             ("② 구조", "전개형\n(물리+학습)", "25.91"),
             ("③ 조건화", "σ 를 측정치\n에서 읽는다", "29.25"),
             ("④ 수렴", "끝까지\n더 학습", "30.32"),
             ("⑤ 융합", "다른 구조\n둘을 평균", f"{p:.2f}")]
    xs = Inches(0.55)
    for i, (tag, what, sc) in enumerate(steps):
        last = i == len(steps) - 1
        card(sl, xs, Inches(1.85), Inches(1.72), Inches(1.75),
             ACCENT_SOFT if last else SURF, ACCENT if last else RULE)
        text(sl, xs + Inches(0.06), Inches(1.96), Inches(1.6), Inches(0.3),
             tag, size=10.5, color=ACCENT, bold=True, font=MONO, align=PP_ALIGN.CENTER)
        text(sl, xs + Inches(0.06), Inches(2.26), Inches(1.6), Inches(0.72),
             what, size=10.5, color=INK, bold=True, align=PP_ALIGN.CENTER)
        text(sl, xs + Inches(0.06), Inches(3.02), Inches(1.6), Inches(0.4),
             f"{sc} dB", size=13.5, color=ACCENT if last else INK2,
             bold=True, font=MONO, align=PP_ALIGN.CENTER)
        xs += Inches(2.02)
        if not last:
            arrow(sl, xs - Inches(0.27), Inches(2.62), Inches(0.2), Inches(0.24))

    rows = [("단계", "무엇을 바꿨나", "왜 올랐나", "이득"),
            ("① 조합", "역산 전에 노이즈를 먼저 지운다",
             "노이즈가 흐림 뒤에 붙어 측정치 위에서는 백색이다", "+7.3"),
            ("② 구조", "물리 제약과 학습을 번갈아 4번",
             "역산은 닫힌 해로 끝내고 네트워크는 모르는 주파수만 채운다", "+4.9"),
            ("③ 조건화", "σ 를 널 원뿔에서 읽어 넣는다",
             "σ 가 장마다 200배 차이난다. 하나로 뭉뚱그리면 평균에 타협한다", "+3.3"),
            ("④ 수렴", "세 번에 걸쳐 230 에폭",
             "늘릴 때마다 마지막 에폭이 best 였다. 세 번 다 수렴 전이었다", "+1.07"),
            ("⑤ 융합", "4단계 · 6단계 모델을 평균",
             "구조가 다르니 틀리는 방식도 달라 오차가 상쇄된다 (무게는 val 에서)",
             f"+{p - 30.32:.2f}")]
    table(sl, Inches(0.7), Inches(3.95), Inches(11.9), Inches(2.1), rows,
          col_w=[1.5, 3.0, 6.0, 1.0], highlight_row=len(rows) - 1, size=11)
    footer(sl, f"출발 13.79 → 최종 {p:.2f}   ·   통과 기준 {BAR_P:.0f} / {BAR_S:.2f}   ·   "
               f"배포 baseline {BASELINE:.2f} / {BASELINE_S:.3f}")

    # ============================================================ 4. 문제
    sl = blank(prs); bg(sl)
    title(sl, "문제 — 두 열화가 겹쳤다", "g = h * f + n", eyebrow="Day 3")
    rows = [("기호", "뜻", "아는가"),
            ("f", "원본 이미지", "모름 — 찾는 것"),
            ("h", "dipole 커널 (한 점이 퍼지는 모양)", "앎 — 조교 지침 4"),
            ("n", "노이즈 (4종, σ 가 장마다 다름)", "모름 — 조교 지침 4"),
            ("g", "측정치", "앎 — 주어진 것")]
    table(sl, Inches(0.9), Inches(1.8), Inches(6.3), Inches(1.7), rows,
          col_w=[0.9, 3.4, 2.0])
    card(sl, Inches(7.6), Inches(1.8), Inches(4.8), Inches(1.7), ACCENT_SOFT, ACCENT)
    text(sl, Inches(7.85), Inches(2.0), Inches(4.3), Inches(1.4),
         [("h 를 안다는 것이 핵심 자산이다.", B),
          ("역필터를 해석적으로 만들 수 있고,", N),
          ("σ 를 공짜로 잴 수 있다.", N)], size=12.5, color=INK2)

    rows = [("", "1일차", "2일차", "3일차"),
            ("문제", "g = f + n", "g = h * f", "g = h * f + n"),
            ("답", "DRUNet", "Wiener K→0", "?"),
            ("점수", "37.42 dB", "109.86 dB", "—"),
            ("입력 SNR", "+8.6 dB", "무한대 (노이즈 없음)", "+0.9 dB")]
    table(sl, Inches(0.9), Inches(3.7), Inches(11.5), Inches(1.7), rows,
          col_w=[1.4, 2.0, 2.6, 2.4], highlight_col=3)
    card(sl, Inches(0.9), Inches(5.65), Inches(11.5), Inches(0.85), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.15), Inches(5.82), Inches(11), Inches(0.5),
         "3일차 측정치는 신호와 노이즈가 거의 같은 크기다 (SNR +0.9 dB). "
         "흐림이 대비를 0.41배로 죽였기 때문이다.", size=13, color=INK, bold=True)

    # ============================================================ 5. 문제 (그림)
    image_slide("day3_forward_chain",
                "문제가 어떻게 만들어지는가",
                "clean → dipole blur → + noise. 학습 쌍도 같은 방식으로 만든다",
                "노이즈가 흐림 **뒤에** 붙는다 — 오른쪽 끝이 노이즈만 뽑아낸 것이다. "
                "그래서 측정치 위에서는 백색이고, 역산 전에 지워야 한다",
                eyebrow="문제")

    # ============================================================ 6. 시도 ①
    sl = blank(prs); bg(sl)
    title(sl, "시도 ① — 1일차 + 2일차를 이어 붙인다", "각자 최고인 도구인데 합치면 진다",
          eyebrow="시도 ①")
    rows = [("순서", "방법", "PSNR"),
            ("디노이징 먼저", "1일차 디노이저 → Wiener", "21.06"),
            ("", "median 3×3 → Wiener", "19.20"),
            ("디컨볼루션 먼저", "Wiener → 1일차 디노이저", "16.88"),
            ("번갈아", "plug-and-play (디노이저 고정)", "16.02"),
            ("참고", "Wiener 단독 · 2일차 답(K→0)", "13.79 · −24")]
    table(sl, Inches(0.9), Inches(1.75), Inches(6.4), Inches(2.1), rows,
          col_w=[1.8, 3.2, 1.4], highlight_row=1)
    card(sl, Inches(7.7), Inches(1.75), Inches(4.7), Inches(2.1), ACCENT_SOFT, ACCENT)
    text(sl, Inches(7.95), Inches(1.95), Inches(4.2), Inches(1.8),
         [("순서가 4.2 dB 를 가른다.", B), ("", N),
          ("Wiener 를 먼저 걸면 노이즈가 1/D 로 증폭되고 X자로 상관돼, "
           "디노이저가 본 적 없는 종류가 된다.", N)], size=12, color=INK2)

    ys = Inches(4.1)
    for head, body, val in [
        ("2일차 답은 노이즈가 0이라서 통했다",
         "1/D 로 역산하면 노이즈 에너지가 폭발한다. n=0 이면 무한대 × 0 = 0 이었다.",
         "+51.5 dB\n노이즈 증폭"),
        ("흐림이 대비를 죽여 같은 σ 가 훨씬 어려워진다",
         "std(f)=0.222 → std(h*f)=0.092. 노이즈는 1일차와 똑같은데 신호만 줄었다.",
         "SNR +8.6\n→ +0.9 dB"),
        ("주파수의 16% 는 진짜로 복원 불가다",
         "|D|<0.1 은 역산 후 SNR 이 −20 dB 이하. 2일차엔 이것도 되살렸다 (노이즈가 0이라).",
         "16%\n버려진다"),
    ]:
        card(sl, Inches(0.9), ys, Inches(11.5), Inches(0.78))
        text(sl, Inches(1.15), ys + Inches(0.07), Inches(7.6), Inches(0.32),
             head, size=13, color=INK, bold=True)
        text(sl, Inches(1.15), ys + Inches(0.38), Inches(7.6), Inches(0.34),
             body, size=11, color=MUTED)
        text(sl, Inches(9.1), ys + Inches(0.12), Inches(3.1), Inches(0.6),
             val, size=13, color=ACCENT, bold=True, font=MONO, align=PP_ALIGN.CENTER)
        ys += Inches(0.88)

    # ============================================================ 7. 선형의 천장
    sl = blank(prs); bg(sl)
    title(sl, "시도 ① 의 한계 — 선형은 21.95 dB 에서 끝난다",
          "정답의 파워 스펙트럼을 알려줘도 그렇다", eyebrow="한계")
    rows = [("필터 / 방법", "PSNR", "성격"),
            ("Wiener 상수 K (val 에서 고름)", "13.72", "실제로 쓸 수 있는 것"),
            ("1일차 디노이저 → Wiener", "21.06", "학습 없는 조합 중 최고"),
            ("오라클 위너 (장마다 참 스펙트럼)", "21.95", "선형의 천장. 만들 수 없다"),
            ("원뿔을 포기하는 방법의 상한", "25.08", "비선형이어도 넘을 수 없다"),
            ("배포 baseline (End2End U-Net)", f"{BASELINE:.2f}", "그 상한에 붙어 있다"),
            ("우리 최종", f"{p:.2f}", "원뿔을 실제로 메우고 있다")]
    table(sl, Inches(0.9), Inches(1.8), Inches(6.6), Inches(2.4), rows,
          col_w=[3.6, 1.3, 3.0], highlight_row=len(rows) - 1)
    card(sl, Inches(7.9), Inches(1.8), Inches(4.5), Inches(2.4), ACCENT_SOFT, ACCENT)
    text(sl, Inches(8.15), Inches(2.0), Inches(4.0), Inches(2.1),
         [("천장이 두 개다.", B), ("", N),
          ("선형은 21.95 에서 끝난다.", N),
          ("원뿔을 포기하면 비선형이라도", N), ("25.08 이 한계다 —", B),
          ("우리 오차(정답의 0.77%)가", N),
          ("원뿔에 든 정답(2.24%)보다 작다.", N), ("", N),
          (f"{p:.2f} 는 원뿔의 최소 65% 를", B), ("메우고 있다는 뜻이다.", B)],
         size=12, color=INK2)

    text(sl, Inches(0.9), Inches(4.45), Inches(11.5), Inches(0.4),
         "왜 정답을 봐도 완벽하지 못한가", size=14, color=ACCENT, bold=True)
    rows = [("이유", "설명"),
            ("위상은 모른다", "파워 스펙트럼은 '어느 굵기 무늬가 얼마나' 만 안다. "
                            "'어디에 있는지' 는 위상이 담고 있다"),
            ("곱하기밖에 못 한다", "F̂(k) = W(k)·G(k). 측정치에 없는 것은 어떤 수를 곱해도 안 나온다"),
            ("오차의 32% 는 버린 주파수", "주파수의 89.2% 가 노이즈가 더 세다. 거기선 포기가 최선이고 "
                                    "버린 만큼이 오차로 남는다"),
            ("오차의 68% 는 새어든 노이즈", "살린 주파수에서는 신호를 완벽히 살리는 것과 "
                                     "노이즈를 완전히 막는 것을 동시에 못 한다")]
    table(sl, Inches(0.9), Inches(4.9), Inches(11.5), Inches(1.5), rows,
          col_w=[2.6, 8.9], size=10.5)

    # ============================================================ 8. 시도 ② DC-Net
    sl = blank(prs); bg(sl)
    title(sl, "시도 ② — DC-Net. 2일차 최고 구조를 그대로",
          "2일차 42.93 dB → 3일차 14.80 dB", eyebrow="시도 ②")
    text(sl, Inches(0.9), Inches(1.75), Inches(11.5), Inches(0.4),
         "hard DC 는 |D|>τ 인 주파수를 G/D 로 못박는다. 그런데 G = D·F + N 이므로",
         size=13.5, color=INK2)
    card(sl, Inches(2.6), Inches(2.25), Inches(8.1), Inches(0.7), ACCENT_SOFT, ACCENT)
    text(sl, Inches(2.7), Inches(2.4), Inches(7.9), Inches(0.45),
         "G/D  =  F  +  N/D        ← 증폭된 노이즈를 못박는다",
         size=15, color=INK, bold=True, font=MONO, align=PP_ALIGN.CENTER)
    text(sl, Inches(0.9), Inches(3.05), Inches(11.5), Inches(0.35),
         "네트워크는 그 주파수를 고칠 권한이 없다. 2일차엔 N=0 이라 완벽했다.",
         size=13, color=MUTED)

    rows = [("τ", "못박는 주파수", "경계 노이즈 증폭 1/τ", "결과"),
            ("0.005 (2일차 최적 방향)", "99.2%", "200배", "폭발"),
            ("0.05", "92.0%", "20배", "14.80 dB"),
            ("0.2", "66.4%", "5배", "DC 가 무의미")]
    table(sl, Inches(0.9), Inches(3.6), Inches(11.5), Inches(1.4), rows,
          col_w=[3.4, 2.4, 3.0, 2.7])
    card(sl, Inches(0.9), Inches(5.25), Inches(11.5), Inches(1.2), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.15), Inches(5.45), Inches(11), Inches(0.9),
         [("τ 를 키우면 DC 가 무의미해지고 줄이면 폭발한다 — 3일차엔 좋은 τ 가 없다.", B),
          ("고칠 방법은 못박지 말고 무게를 두는 것. 그것이 soft DC 이고,", N),
          ("전개형이 곧 DC-Net 의 노이즈 대응 일반형이다 (λ→0 이면 DC-Net).", N)],
         size=13, color=INK2)

    # ============================================================ 9. 시도 ③ 2단 분해
    sl = blank(prs); bg(sl)
    title(sl, "시도 ③ — 2단 분해. 1일차 문제로 환원하기",
          "노이즈가 흐림 뒤에 붙으니 측정치 위에서는 1일차 문제다", eyebrow="시도 ③")
    xs = Inches(1.6)
    for h1, h2 in [("측정치 g", "h*f + n"), ("디노이저", "z ≈ h*f\n1일차 문제"),
                   ("역필터", "x = D·Z/(D²+λ)\n2일차 답"), ("복원", "15.6 dB\n실패")]:
        card(sl, xs, Inches(1.8), Inches(2.2), Inches(1.3), SURF)
        text(sl, xs + Inches(0.1), Inches(1.92), Inches(2.0), Inches(0.32),
             h1, size=12.5, color=INK, bold=True, align=PP_ALIGN.CENTER)
        text(sl, xs + Inches(0.1), Inches(2.3), Inches(2.0), Inches(0.7),
             h2, size=10.5, color=MUTED, font=MONO, align=PP_ALIGN.CENTER)
        xs += Inches(2.5)

    rows = [("측정치 영역에서 지운 정도", "→ 최종 PSNR", "→ 최종 SSIM"),
            ("25 dB", "20.24", "0.5572"),
            ("35 dB  ← 우리가 도달한 곳", "26.83", "0.7809"),
            ("40 dB", "29.86", "0.8626"),
            ("완벽 (오차 0)", "71.47", "0.9999")]
    table(sl, Inches(0.9), Inches(3.35), Inches(6.4), Inches(1.7), rows,
          col_w=[3.4, 1.6, 1.6], highlight_row=4)
    card(sl, Inches(7.7), Inches(3.35), Inches(4.7), Inches(1.7), ACCENT_SOFT, ACCENT)
    text(sl, Inches(7.95), Inches(3.55), Inches(4.2), Inches(1.4),
         [("전달 곡선은 맞았다.", B),
          ("측정치에서 노이즈만 완벽히 지우면", N),
          ("2일차 답이 71 dB 를 낸다 —", N),
          ("구조적 한계는 없다.", N)], size=12.5, color=INK2)

    card(sl, Inches(0.9), Inches(5.3), Inches(11.5), Inches(1.15), ACCENT_SOFT, WARN)
    text(sl, Inches(1.15), Inches(5.48), Inches(11), Inches(0.9),
         [("그런데 필요한 난이도를 계산하지 않았다. 1일차는 25.95 → 37.42 (+11.5 dB) 였는데 "
           "여기서는 13.50 → 40 (+26.5 dB) 를 요구한다.", B),
          ("역필터가 네트워크의 오차를 1/D 로 증폭하고, 뭉개진 h*f 는 선명한 f 보다 "
           "사전지식이 약하다. 1일차보다 훨씬 어려운 1일차 문제였다.", N)],
         size=12.5, color=INK2)

    # ============================================================ 10. 우리 방법
    sl = blank(prs); bg(sl)
    title(sl, "우리 방법 — 전개형", "데이터 정합과 사전지식을 번갈아 4번",
          eyebrow="방법")
    xs = Inches(0.75)
    boxes = [("측정치 g", "h*f + n", SURF),
             ("Wiener 초기화", "x₀ = D·G\n   /(D²+λ₀)", SURF),
             ("사전지식\nDRUNet", "z = net(x, σ)\nσ 는 측정치에서", ACCENT_SOFT),
             ("데이터 정합", "x = (D·G+λZ)\n     /(D²+λ)", SURF),
             ("복원 f̂", f"{p:.2f} dB", ACCENT_SOFT)]
    for i, (h1, h2, fill) in enumerate(boxes):
        card(sl, xs, Inches(2.0), Inches(2.05), Inches(1.6), fill,
             ACCENT if fill == ACCENT_SOFT else RULE)
        text(sl, xs + Inches(0.08), Inches(2.14), Inches(1.9), Inches(0.5),
             h1, size=12.5, color=INK, bold=True, align=PP_ALIGN.CENTER)
        text(sl, xs + Inches(0.08), Inches(2.64), Inches(1.9), Inches(0.8),
             h2, size=10.5, color=MUTED, align=PP_ALIGN.CENTER, font=MONO)
        xs += Inches(2.4)
        if i < len(boxes) - 1:
            arrow(sl, xs - Inches(0.31), Inches(2.68), Inches(0.24), Inches(0.26))
    text(sl, Inches(3.1), Inches(3.78), Inches(6.5), Inches(0.4),
         "↑____________ 4번 반복 ____________|", size=12, color=ACCENT,
         bold=True, align=PP_ALIGN.CENTER, font=MONO)

    rows = [("단계", "학습되나", "하는 일"),
            ("Wiener 초기화", "아니오", "닫힌 해. dipole 은 주파수 영역에서 대각 연산자다"),
            ("DRUNet", "예", "역산이 포기한 주파수를 이미지 사전지식으로 메운다"),
            ("데이터 정합", "λ 만", "매번 측정치로 되돌려 오차가 쌓이지 않게 한다")]
    table(sl, Inches(0.9), Inches(4.3), Inches(11.5), Inches(1.4), rows,
          col_w=[2.4, 1.4, 7.7])
    card(sl, Inches(0.9), Inches(5.95), Inches(11.5), Inches(0.9), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.15), Inches(6.12), Inches(11), Inches(0.6),
         [("네트워크는 '아는 주파수를 다시 맞히는' 일을 배울 필요가 없다. "
           "모르는 주파수를 채우는 데만 용량을 쓴다.", B),
          ("λ 는 단계마다 따로 학습해 '이번엔 측정치를 얼마나 믿을지' 를 스스로 정한다.", N)],
         size=12.5, color=INK2)

    # ============================================================ 11. σ 조건화
    sl = blank(prs); bg(sl)
    title(sl, "시도 ④ — σ 를 라벨 없이 읽어 모델에 알려준다",
          "3일차 σ 는 장마다 200배 차이난다", eyebrow="시도 ④")
    text(sl, Inches(0.9), Inches(1.75), Inches(11.5), Inches(0.8),
         [("|D| < 0.02 인 주파수엔 dipole 이 신호를 보내지 않는다. 거기 있는 것은 전부 노이즈다.", B),
          ("파세발로 E|G|² = N·σ².  정답도 noise_meta.json 도 쓰지 않는다 — 측정치 하나에서 나온다.", N)],
         size=13, color=INK2)
    rows = [("", "값"),
            ("σ 범위 (test 100장)", "0.0007 ~ 0.1325   (200배)"),
            ("추정 상대오차", "중앙값 1.9%"),
            ("σ<0.05 구간 성능", "23.91 dB"),
            ("σ≥0.10 구간 성능", "17.64 dB   (6.3 dB 차이)")]
    table(sl, Inches(0.9), Inches(2.85), Inches(6.3), Inches(1.7), rows, col_w=[2.9, 3.4])
    card(sl, Inches(7.6), Inches(2.85), Inches(4.8), Inches(1.7), ACCENT_SOFT, ACCENT)
    text(sl, Inches(7.85), Inches(3.05), Inches(4.3), Inches(1.4),
         [("blind 모델 하나로 200배 범위를", N), ("덮으려면 평균에 타협해야 한다.", B),
          ("", N), ("σ 를 알려주면 매 장에 맞는", N), ("세기로 지울 수 있다.", N)],
         size=12.5, color=INK2)
    card(sl, Inches(0.9), Inches(4.8), Inches(11.5), Inches(1.5), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.15), Inches(5.0), Inches(11), Inches(1.2),
         [("커널을 알기에 쓸 수 있는 추정기다.", B),
          ("MAD 같은 일반 추정기는 이미지의 고주파를 노이즈로 착각한다 — "
           "1일차에 σ 게이트가 실패한 원인이 그것이었다.", N),
          ("여기서는 '신호가 있을 수 없는 자리' 를 커널이 알려주므로 착각할 여지가 없다.", N)],
         size=12.5, color=INK2)

    # ============================================================ 12. σ ablation
    sl = blank(prs); bg(sl)
    title(sl, "시도 ④ 의 근거 — σ 가 성능의 절반을 책임진다",
          "가중치는 그대로 두고 σ 입력만 바꾼다. 학습이 필요 없는 ablation",
          eyebrow="근거")
    ab = R.get("ablation", [])
    base_p = ab[0][1][0] if ab else p
    rows = [("σ 를 어떻게 주는가", "PSNR", "SSIM", "손실")] + [
        (k, f"{v[0]:.2f}", f"{v[1]:.4f}",
         "—" if i == 0 else f"{v[0] - base_p:+.2f} dB") for i, (k, v) in enumerate(ab)]
    table(sl, Inches(0.9), Inches(1.85), Inches(11.5), Inches(2.2), rows,
          col_w=[5.0, 1.8, 1.8, 2.4], highlight_row=1)
    card(sl, Inches(0.9), Inches(4.4), Inches(11.5), Inches(1.9), ACCENT_SOFT, ACCENT)
    text(sl, Inches(1.15), Inches(4.6), Inches(11), Inches(1.6),
         [("장마다 다른 σ 를 주는 것이, 하나의 값으로 뭉뚱그리는 것보다 4.22 dB 낫다.", B),
          ("", N),
          ("σ 를 0 이라고 알려주면 29.59 에서 15.93 으로 무너진다. "
           "2배·절반으로 틀리게 알려줘도 7 dB 가까이 잃는다.", N),
          ("모델이 σ 를 실제로 쓰고 있고, 정확히 알려주는 것이 중요하다는 직접적인 증거다.", N)],
         size=13, color=INK2)

    # ============================================================ 13. 안 통한 것들
    sl = blank(prs); bg(sl)
    title(sl, "시도 ⑤ — 안 통한 것들, 그리고 고쳐서 통한 것",
          "왜 안 되는지 알면 고칠 수 있다", eyebrow="실패 기록")
    rows = [("시도", "결과", "왜 안 됐나"),
            ("SSIM 을 처음부터 손실에", "17.5",
             "덜 학습된 모델을 '정답과 맞든 아니든 국소 대비를 키우는' 쪽으로 민다"),
            ("언샤프 후처리로 SSIM 보정", "무효",
             "잔차까지 키워 σx 만 커진다. 잃은 국소 대비는 사후에 만들 수 없다"),
            ("노이즈 통계 조건화 (왜도·첨도)", "29.59",
             "첨도가 rician 을 가르지만(10.4 vs 2.9~3.8) 모델이 쓰지 않았다 (−0.03 dB)"),
            ("모델 융합 1차 — 형제 셋", "29.66",
             "실패. 같은 체크포인트에서 이어받은 형제라 틀리는 방식이 같다"),
            ("추론 때 반복 횟수 늘리기", "20.99",
             "4단계로 학습한 모델은 5단계에서 분포를 벗어난다. val 이 4를 골랐다"),
            ("end-to-end DRUNet (대조군)", "24.6",
             "전개형보다 5 dB 아래. 물리 구조가 실제로 이득이라는 증거"),
            ("모델 융합 2차 — 4단계 + 6단계", f"{p:.2f}",
             "성공. 구조가 다르니 오차 상관이 낮다. 1차 실패의 원인을 고친 것이다")]
    table(sl, Inches(0.7), Inches(1.85), Inches(11.9), Inches(3.0), rows,
          col_w=[3.2, 1.2, 7.5], size=11)
    card(sl, Inches(0.7), Inches(5.2), Inches(11.9), Inches(1.15), ACCENT_SOFT, ACCENT)
    text(sl, Inches(0.95), Inches(5.4), Inches(11.4), Inches(0.9),
         [("점수를 올린 것은 하나뿐이었다 — 수렴할 때까지 더 학습하기.", B),
          ("60 에폭으로 끝냈는데 매번 마지막 에폭까지 오르고 있었다. "
           "새 구조를 찾기 전에 있는 것을 끝까지 돌리는 편이 나았다.", N)],
         size=12.5, color=INK2)

    # ============================================================ 14. 최종 결과
    sl = blank(prs); bg(sl)
    title(sl, "최종 결과", "test_deconv_noise 100장 · 4× self-ensemble", eyebrow="제출값")
    rows = [("", "PSNR", "SSIM", "판정"),
            ("통과 기준", f"{BAR_P:.0f}", f"{BAR_S:.2f}", "—"),
            ("입력 (blur + noise)", "8.02", "−0.0187", "출발점"),
            ("배포 baseline (End2End U-Net)", f"{BASELINE:.2f}", f"{BASELINE_S:.3f}", "미달"),
            ("우리 (전개형 + σ + 4× SE + 융합)", f"{p:.2f}", f"{s:.4f}", "통과")]
    table(sl, Inches(0.9), Inches(1.8), Inches(11.5), Inches(2.2), rows,
          col_w=[5.2, 2.0, 2.0, 2.3], highlight_row=len(rows) - 1)

    nz = R.get("per_noise", {})
    rows = [("노이즈", "PSNR", "SSIM")] + [
        (k, f"{v[0]:.2f}", f"{v[1]:.4f}") for k, v in nz.items()]
    table(sl, Inches(0.9), Inches(4.3), Inches(5.4), Inches(1.7), rows,
          col_w=[2.8, 1.3, 1.3])
    card(sl, Inches(6.7), Inches(4.3), Inches(5.7), Inches(1.7), ACCENT_SOFT, ACCENT)
    text(sl, Inches(6.95), Inches(4.5), Inches(5.2), Inches(1.4),
         [("rician 만 23.39 로 7 dB 뒤진다.", B),
          ("정류가 밝기를 +0.0396 밀어 올리는데", N),
          ("dipole 은 DC 를 1/3 로 보존하므로", N),
          ("역산에서 3배가 되어 0.119 로 남는다", N),
          ("— 이미지 std 가 0.222 인데 그렇다.", N)], size=12, color=INK2)
    footer(sl, f"배포 baseline 대비 {gain:+.2f} dB · {s - BASELINE_S:+.4f}   ·   "
               f"통과 기준 대비 {p - BAR_P:+.2f} dB · {s - BAR_S:+.4f}")

    # ============================================================ 15~16. 복원 결과
    image_slide("day3_methods_grid",
                "복원 결과 — 노이즈 4종 × 방법별",
                "각 종류의 σ 중앙값에 가까운 대표 이미지. 잘 나온 것을 고르지 않았다",
                "왼쪽에서 오른쪽으로 좋아진다: 정답 · 측정치 8.02 · Wiener 13.79 · "
                "median→Wiener 19.20 · 1일차 디노이저→Wiener 21.06 · 우리 모델",
                mode="tall")
    image_slide("day3_diff_zoom",
                "복원 결과 — difference map 과 zoom-in",
                "위: 복원 · 가운데: |복원−정답| (열 공통 스케일) · 아래: 72×72 확대",
                "X자 무늬는 매직앵글 영널 원뿔이다. 정보가 애초에 없는 자리라 "
                "어떤 방법으로도 남는다",
                mode="tall")

    # ============================================================ 17. 취약점
    sl = blank(prs); bg(sl)
    title(sl, "취약점 분석 — 어떤 노이즈, 어떤 이미지에 약한가",
          "노이즈는 meta 가 알려주지만 이미지 종류는 라벨이 없다. 이미지 자체 특징으로 가른다",
          eyebrow="결과 분석")
    f = FIG / "day3_image_types.png"
    if f.exists():
        pic(sl, f, Inches(0.7), Inches(1.6), w=Inches(11.9))
    rows = [("특징", "하위 1/3 → 상위 1/3", "상관계수", "해석"),
            ("에지 밀도", "−7.63 dB", "−0.458", "구조가 복잡할수록 나쁘다. 가장 강한 예측 변수"),
            ("대비 (std)", "−4.33 dB", "−0.299", "σ 는 이미지와 무관하게 뽑히므로 대비가 클수록 불리"),
            ("동적범위", "−3.55 dB", "−0.248", "위와 같은 이유"),
            ("무늬 조밀도", "−2.77 dB", "−0.170", "고주파일수록 |D| 가 작아 역산이 어렵다")]
    table(sl, Inches(0.9), Inches(4.6), Inches(11.5), Inches(1.6), rows,
          col_w=[2.0, 2.2, 1.5, 5.8], size=10.5)
    footer(sl, "가장 못 살린 10장 — rician 8 · uniform 2 · gaussian 0 · s&p 0   |   "
               "σ 0.128 (전체 0.074) · 에지 0.082 (0.050).  약점은 '강한 rician + 복잡한 구조' 다")

    # ============================================================ 18. label-free
    sl = blank(prs); bg(sl)
    title(sl, "보너스 — label-free. 정답을 한 장도 쓰지 않고",
          "노이즈가 측정치 위에서 백색이므로 Noise2Void 가 그대로 통한다",
          eyebrow="보너스")
    text(sl, Inches(0.9), Inches(1.7), Inches(11.5), Inches(0.75),
         [("g 의 일부 화소를 이웃으로 덮고, 덮은 자리에서 g 를 맞히게 한다.", B),
          ("가린 화소를 못 보니 이웃에서 신호를 추정할 수밖에 없고, 잡음은 화소마다 독립이라 "
           "예측할 수 없으므로 최적해가 곧 노이즈 없는 h*f 다.", N)], size=12.5, color=INK2)
    xs = Inches(0.9)
    for h1, h2 in [("측정치 g", "가진 것은\n이것뿐"),
                   ("N2V 디노이저", "가린 자리에서\ng 를 맞힌다"),
                   ("σ 추정", "널 원뿔에서\n라벨 불필요"),
                   ("역필터", "λ 도 측정치에서\n학습 없음"),
                   ("복원", (f"{R['lf_psnr']:.2f} dB" if "lf_psnr" in R else "—"))]:
        card(sl, xs, Inches(2.6), Inches(2.15), Inches(1.4), ACCENT_SOFT, ACCENT)
        text(sl, xs + Inches(0.08), Inches(2.72), Inches(2.0), Inches(0.35),
             h1, size=12.5, color=INK, bold=True, align=PP_ALIGN.CENTER)
        text(sl, xs + Inches(0.08), Inches(3.1), Inches(2.0), Inches(0.8),
             h2, size=10, color=MUTED, align=PP_ALIGN.CENTER, font=MONO)
        xs += Inches(2.35)

    lfp, lfs = R.get("lf_psnr", 0.0), R.get("lf_ssim", 0.0)
    rows = [("방법", "정답 사용", "PSNR", "SSIM"),
            ("Wiener 단독", "없음", "13.79", "0.4498"),
            ("label-free (3일차 자기지도)", "없음", f"{lfp:.2f}", f"{lfs:.4f}"),
            ("우리 최종 (supervised)", "학습에 사용", f"{p:.2f}", f"{s:.4f}")]
    table(sl, Inches(0.9), Inches(4.3), Inches(6.4), Inches(1.4), rows,
          col_w=[3.2, 1.4, 1.0, 1.0], highlight_row=2)
    card(sl, Inches(7.7), Inches(4.3), Inches(4.7), Inches(1.4), ACCENT_SOFT, ACCENT)
    text(sl, Inches(7.95), Inches(4.48), Inches(4.2), Inches(1.1),
         [("λ 도 라벨 없이 정한다.", B),
          ("고전 Wiener 의 K = 잡음/신호 를", N),
          ("측정치에서 계산한다. val 에서 고른", N),
          ("고정 K(17.77)보다 낫다 (17.90).", B)], size=11.5, color=INK2)
    footer(sl, "학습 측정치는 train 의 clean 에서 합성하되 clean 은 손실에 한 번도 들어가지 않는다 · "
               "test 는 건드리지 않는다")

    # ============================================================ 19. 검증 규칙
    sl = blank(prs); bg(sl)
    title(sl, "test set 은 채점에만 썼다",
          "학습이 아니어도 하이퍼파라미터를 test 로 고르면 test 를 쓴 것이다",
          eyebrow="검증 규칙")
    ys = Inches(1.9)
    for head, body in [
        ("튜닝은 val 에서만",
         "val clean 에 forward + 파일명 seed 노이즈를 걸어 (측정치, 정답) 쌍을 만든다. "
         "K · λ · 반복 횟수 · 융합 무게를 전부 여기서 고른다.  day3_common.load_val()"),
        ("test 는 report() 한 번",
         "배포된 test_deconv_noise 는 최종 점수를 낼 때만 읽는다.  day3_common.load_test()"),
        ("noise_meta.json 은 결과 분석에만",
         "배포 안내 그대로. 표를 노이즈 종류별로 쪼개고 그림의 대표 이미지를 고르는 데만 쓴다. "
         "복원 함수가 받는 것은 측정치와 b0 뿐이다."),
        ("metric 은 배포 구현 그대로",
         "src/deconv 에 초기 단계의 skimage 기반 metrics 가 남아 있었다. 이름을 바꿔 가리지 "
         "못하게 하고, 다른 구현이 잡히면 즉시 멈추는 가드를 넣었다."),
        ("직접 찾아 고친 위반",
         "초기 스크립트가 K 를 test 에서 스윕하고 있었다. combine_day3.py 는 폐기하고 "
         "나머지 스윕을 val 로 옮겼다.  check_rules.py 가 지침 4개를 코드로 검증한다."),
    ]:
        card(sl, Inches(0.9), ys, Inches(11.5), Inches(0.88))
        text(sl, Inches(1.15), ys + Inches(0.08), Inches(11), Inches(0.3),
             head, size=13, color=ACCENT, bold=True)
        text(sl, Inches(1.15), ys + Inches(0.38), Inches(11), Inches(0.44),
             body, size=11, color=MUTED)
        ys += Inches(0.98)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--psnr", type=float, required=True, help="최종 모델 test PSNR")
    ap.add_argument("--ssim", type=float, required=True)
    ap.add_argument("--lf-psnr", type=float, default=None, help="label-free test PSNR")
    ap.add_argument("--lf-ssim", type=float, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "실습5_day3_발표.pptx")
    args = ap.parse_args()

    # eval_day3.py 가 test 100장에서 실제로 낸 값. 다시 재면 여기만 고친다.
    R = {
        "per_noise": {"gaussian": (30.45, 0.8895), "rician": (23.39, 0.7692),
                      "uniform": (29.87, 0.9150), "salt_and_pepper": (37.76, 0.9863)},
        "ablation": [("추정 σ (장마다 다르게)", (29.59, 0.8817)),
                     ("전체 평균 하나로 고정", (25.37, 0.7751)),
                     ("σ 2배 (과대평가)", (22.52, 0.7890)),
                     ("σ 절반 (과소평가)", (22.92, 0.6484)),
                     ("전부 0 (노이즈가 없다고)", (15.93, 0.4040))],
    }
    if args.lf_psnr is not None:
        R["lf_psnr"], R["lf_ssim"] = args.lf_psnr, args.lf_ssim or 0.0

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    build(prs, args.name, {"psnr": args.psnr, "ssim": args.ssim}, R)
    prs.save(str(args.out))
    print(f"슬라이드 {len(prs.slides._sldIdLst)}장 → {args.out}")


if __name__ == "__main__":
    main()
