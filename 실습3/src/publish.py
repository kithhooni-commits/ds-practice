"""Assemble the publishable bundle for one 실습 folder inside the ds-practice repo.

The repo holds one folder per 실습 and GitHub Pages serves the repo root, so a project
lives at /<slug>/ and the root index lists whatever folders exist. Re-running this for
실습4 later needs only a different --slug and a different MANIFEST.

person_3 is excluded by default. Its reference set is selfies of an identifiable real
person, and Pages makes whatever lands here publicly fetchable and indexable; the
concept stays in every table and CSV, only its images are withheld. --include-person3
overrides that.

    python publish.py --repo ../ds-practice --slug 실습3
"""

import argparse
import os
import re
import shutil
import sys

import markdown

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TITLE = "이미지 생성 모델 개인화 — 파이프라인과 지표 검증"
BLURB = ("SD3.5-medium에 Textual Inversion · LoRA · DoRA · prior-preservation을 적용하고, "
         "CLIP T2I/I2I와 DINO로 채점한 뒤 그 지표 자체를 검증한 실험.")

# (source, destination) — destination is relative to the project folder
FILES = [
    ("FINDINGS.md", "FINDINGS.md"),
    ("발표_파이프라인.pptx", "발표_파이프라인.pptx"),
    ("zwx10_review/zwx_table.md", "tables/zwx_table.md"),
    ("v2_review/breakdown_v2.csv", "data/breakdown_v2.csv"),
    ("zwx10_review/breakdown_zwx10.csv", "data/breakdown_zwx10.csv"),
    ("r512_review/breakdown_r512.csv", "data/breakdown_r512.csv"),
    ("baseline_review/results/breakdown_all.csv", "data/breakdown_baseline.csv"),
]

SRC = ["run_all.py", "train_dreambooth_lora.py", "train_textual_inversion.py",
       "generate.py", "evaluation.py", "eval_breakdown.py", "make_viewer.py",
       "compare_res.py", "zwx_table.py", "make_deck3.py", "publish.py"]

FIGURES = ["deck3_img/trigger.jpg", "deck3_img/tank.jpg", "deck3_img/ti_fail.jpg",
           "deck3_img/prior.jpg", "deck3_img/entangle.jpg", "deck3_img/refs_af.jpg",
           "compare/sks_scan_p0.jpg"]

CONCEPTS = ["actionfigure_2", "decoritems_woodenpot", "furniture_sofa2",
            "instrument_music2", "luggage_backpack1", "person_3", "pet_cat5",
            "scene_waterfall", "transport_tank", "wearable_jacket1"]

CSS = """
:root{
  --paper:#FAF9F5; --band:#F5F2EA; --card:#FFFFFF; --ink:#1F1E1D; --ink2:#55524C;
  --ink3:#8A867E; --line:#E3DED2; --coral:#C15F3C; --coral-soft:#F3E4DC; --ochre:#8A6A2F;
}
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
  --paper:#171614; --band:#1E1D1A; --card:#211F1C; --ink:#F2EFE8; --ink2:#BDB8AE;
  --ink3:#8A867E; --line:#33302A; --coral:#E08256; --coral-soft:#3A2620; --ochre:#C9A54F;
}}
:root[data-theme="dark"]{
  --paper:#171614; --band:#1E1D1A; --card:#211F1C; --ink:#F2EFE8; --ink2:#BDB8AE;
  --ink3:#8A867E; --line:#33302A; --coral:#E08256; --coral-soft:#3A2620; --ochre:#C9A54F;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:15px/1.72 -apple-system,"Segoe UI","Malgun Gothic","맑은 고딕",system-ui,sans-serif;}
.wrap{max-width:980px;margin:0 auto;padding:0 24px 80px}
header.top{background:var(--band);border-bottom:1px solid var(--line);margin-bottom:34px}
header.top .wrap{padding-top:38px;padding-bottom:30px}
.eyebrow{font:600 12px/1 ui-monospace,Consolas,monospace;letter-spacing:.09em;
  color:var(--coral);text-transform:uppercase}
h1{font-size:30px;line-height:1.28;margin:14px 0 12px;letter-spacing:-.01em}
h2{font-size:21px;margin:38px 0 12px;padding-top:16px;border-top:1px solid var(--line)}
h3{font-size:16.5px;margin:26px 0 8px}
h4{font-size:14.5px;margin:20px 0 6px;color:var(--ink2)}
p{margin:0 0 13px;color:var(--ink2)}
h1+p,.lede{color:var(--ink2);font-size:16px}
a{color:var(--coral)}
code{font:13px ui-monospace,Consolas,monospace;background:var(--band);
  border:1px solid var(--line);border-radius:3px;padding:1px 5px}
pre{background:var(--band);border:1px solid var(--line);border-radius:4px;
  padding:13px 15px;overflow-x:auto}
pre code{background:none;border:0;padding:0}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px;display:block;
  overflow-x:auto;white-space:nowrap}
th,td{border:1px solid var(--line);padding:7px 11px;text-align:left}
th{background:var(--coral-soft);color:var(--ink);font-weight:600}
td:not(:first-child){font-family:ui-monospace,Consolas,monospace}
blockquote{border-left:2px solid var(--coral);margin:14px 0;padding:2px 0 2px 15px;color:var(--ink2)}
hr{border:0;border-top:1px solid var(--line);margin:30px 0}
ul,ol{color:var(--ink2);padding-left:22px}
li{margin:4px 0}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:3px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));gap:13px;margin:16px 0}
.card{display:block;background:var(--card);border:1px solid var(--line);border-radius:4px;
  padding:14px 15px;text-decoration:none;color:inherit;transition:border-color .12s}
.card:hover{border-color:var(--coral)}
.card .k{font:600 14.5px/1.3 ui-monospace,Consolas,monospace;color:var(--ink)}
.card .d{font-size:12px;color:var(--ink3);margin-top:5px;line-height:1.5}
.note{border-left:2px solid var(--coral);background:var(--card);padding:12px 16px;
  margin:18px 0;font-size:13.5px;color:var(--ink2)}
.note.warn{border-left-color:var(--ochre);background:var(--band)}
.note b{color:var(--ink)}
nav.crumb{font-size:12.5px;color:var(--ink3);margin-bottom:6px}
nav.crumb a{color:var(--ink3)}
footer{margin-top:52px;padding-top:18px;border-top:1px solid var(--line);
  font-size:12.5px;color:var(--ink3)}
"""


PAGE_TMPL = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%(title)s</title>
<style>%(css)s</style>
%(head)s
<div class="wrap">%(crumb)s%(body)s</div>
"""


def page(title, body, crumb=""):
    """A leading <header class="top"> is hoisted out of .wrap so its band runs full-bleed."""
    head = ""
    body = body.lstrip()
    if body.startswith('<header class="top">'):
        cut = body.index("</header>") + len("</header>")
        head, body = body[:cut], body[cut:]
    return PAGE_TMPL % {"title": title, "css": CSS, "head": head,
                        "crumb": crumb, "body": body}


def render_md(path, title, crumb):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    html = markdown.markdown(src, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    # the docs use ~~strike~~; python-markdown core does not
    html = re.sub(r"~~(.+?)~~", r"<del>\1</del>", html, flags=re.S)
    return page(title, html, crumb)


def copy(src_root, dst_root, pairs):
    n = 0
    for src, dst in pairs:
        s = os.path.join(src_root, src)
        if not os.path.exists(s):
            print(f"  [skip missing] {src}")
            continue
        d = os.path.join(dst_root, dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(s, d)
        n += 1
    return n


def project_index(slug, viewers, figures, excluded):
    v_cards = "\n".join(
        f'<a class="card" href="viewer/viewer_{c}.html"><div class="k">{c}</div>'
        f'<div class="d">4개 런 · 최대 11개 방법 · 프롬프트 10개</div></a>'
        for c in viewers)
    ex = ""
    if excluded:
        ex = ('<div class="note warn"><b>공개본에서 제외된 컨셉:</b> '
              + ", ".join(f"<code>{c}</code>" for c in excluded) +
              " — 레퍼런스가 실존 인물의 셀피라 이미지를 공개 저장소에 올리지 않았다. "
              "수치는 모든 표와 CSV에 그대로 들어 있다.</div>")
    figs = "\n".join(
        f'<figure style="margin:0 0 18px"><img src="figures/{os.path.basename(f)}" alt="">'
        f'</figure>' for f in figures[:1])
    return page(TITLE, f"""
<header class="top"><div class="wrap">
  <div class="eyebrow">{slug} · CustomConcept101 · SD3.5-medium</div>
  <h1>{TITLE}</h1>
  <p class="lede">{BLURB}</p>
</div></header>

<h2>핵심 발견</h2>
<p>가장 큰 결함은 하이퍼파라미터가 아니라 <b>단어 하나</b>였다. DreamBooth 관행대로 트리거 토큰을
<code>sks</code>로 잡았는데, SD3.5의 텍스트 인코더는 이것을 <b>SKS 소총</b>으로 읽는다. 요청하지 않은
소총과 군복이 이미지에 들어왔다.</p>
{figs}
<p>더 흥미로운 것은 지표의 반응이다. 이 결함을 고쳤더니(<code>zwx</code>로 교체) <b>DINO가 10컨셉 중
8개에서 오히려 하락</b>했다. CLIP T2I·I2I·DINO 세 지표를 다 써도 "폭포 앞에 소총이 떠 있다"를
잡아내지 못한다.</p>
<div class="note">이 실험의 결론은 "어떤 방법이 좋은가"보다 <b>"무엇으로 재는가"</b>에 있다.
지표는 게이밍당하고(best-of-N), 보지 못하고(학습 해상도 512), 때로는 반대로 채점한다(트리거 토큰).</div>

<h2>문서</h2>
<div class="grid">
  <a class="card" href="findings.html"><div class="k">FINDINGS</div>
    <div class="d">전체 분석 — 지표 검증 · ablation 8건 · 정성 관찰 · 한계</div></a>
  <a class="card" href="tables/zwx_table.html"><div class="k">트리거 토큰 평가 표</div>
    <div class="d">sks → zwx, 10컨셉 × LoRA/DoRA, 컨셉 단위 검정</div></a>
  <a class="card" href="발표_파이프라인.pptx"><div class="k">발표 자료 (.pptx)</div>
    <div class="d">파이프라인과 결과 13슬라이드</div></a>
</div>

<h2>생성 이미지 뷰어</h2>
<p>컨셉을 고르면 프롬프트별로 모든 런·방법의 생성물이 나란히 놓인다. 두 지표 모두 프롬프트별 교란을
가지므로 <b>같은 프롬프트 안에서 방법끼리</b> 비교하는 것만 유효하고, 레이아웃이 정확히 그 형태다.</p>
{ex}
<div class="grid">{v_cards}</div>

<h2>데이터와 코드</h2>
<div class="grid">
  <a class="card" href="data/"><div class="k">breakdown_*.csv</div>
    <div class="d">프롬프트 단위 원점수 — T2I · I2I · DINO</div></a>
  <a class="card" href="src/"><div class="k">src/</div>
    <div class="d">학습 · 생성 · 평가 · 시각화 스크립트</div></a>
  <a class="card" href="figures/"><div class="k">figures/</div>
    <div class="d">발표에 쓴 비교 그림</div></a>
</div>

<footer>CustomConcept101 · Stable Diffusion 3.5 Medium · CLIP ViT-B/32 · DINO ViT-S/16</footer>
""")


def root_index(repo, slugs):
    cards = []
    for s in slugs:
        cards.append(f'<a class="card" href="{s}/"><div class="k">{s}</div>'
                     f'<div class="d">{TITLE if s == "실습3" else ""}</div></a>')
    return page("데이터 사이언스 실습", f"""
<header class="top"><div class="wrap">
  <div class="eyebrow">ds-practice</div>
  <h1>데이터 사이언스 실습</h1>
  <p class="lede">로컬 · Colab · Claude Code on the web에서 이어서 작업하는 저장소.
  실습 하나당 폴더 하나.</p>
</div></header>
<div class="grid">{''.join(cards)}</div>
<footer>github.com/kithhooni-commits/ds-practice</footer>
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--slug", default="실습3")
    ap.add_argument("--viewers", default="view")
    ap.add_argument("--include-person3", action="store_true",
                    help="publish the person_3 viewer too — it embeds selfies of an "
                         "identifiable real person, so this is opt-in")
    args = ap.parse_args()

    proj = os.path.join(args.repo, args.slug)
    os.makedirs(proj, exist_ok=True)

    excluded = [] if args.include_person3 else ["person_3"]
    viewers = [c for c in CONCEPTS if c not in excluded]

    n = copy(".", proj, FILES)
    n += copy(".", proj, [(p, f"src/{os.path.basename(p)}") for p in SRC])
    n += copy(".", proj, [(p, f"figures/{os.path.basename(p)}") for p in FIGURES])
    n += copy(".", proj, [(f"{args.viewers}/viewer_{c}.html", f"viewer/viewer_{c}.html")
                          for c in viewers])
    print(f"  copied {n} files -> {proj}")

    crumb = f'<nav class="crumb"><a href="../">← {args.slug}</a></nav>'
    with open(os.path.join(proj, "findings.html"), "w", encoding="utf-8") as f:
        f.write(render_md("FINDINGS.md", "FINDINGS — " + TITLE, crumb))
    with open(os.path.join(proj, "tables", "zwx_table.html"), "w", encoding="utf-8") as f:
        f.write(render_md("zwx10_review/zwx_table.md", "트리거 토큰 평가 표",
                          '<nav class="crumb"><a href="../">← 프로젝트 홈</a></nav>'))
    with open(os.path.join(proj, "index.html"), "w", encoding="utf-8") as f:
        f.write(project_index(args.slug, viewers, FIGURES, excluded))

    slugs = sorted(d for d in os.listdir(args.repo)
                   if d.startswith("실습") and os.path.isdir(os.path.join(args.repo, d)))
    with open(os.path.join(args.repo, "index.html"), "w", encoding="utf-8") as f:
        f.write(root_index(args.repo, slugs))
    open(os.path.join(args.repo, ".nojekyll"), "w").close()

    total = sum(os.path.getsize(os.path.join(r, x))
                for r, _, fs in os.walk(proj) for x in fs)
    print(f"  pages: {args.slug}/index.html, findings.html, tables/zwx_table.html")
    print(f"  root index lists: {', '.join(slugs)}")
    print(f"  bundle size: {total / 1e6:.1f} MB")
    if excluded:
        print(f"  EXCLUDED from publish: {', '.join(excluded)} (use --include-person3 to add)")


if __name__ == "__main__":
    main()
