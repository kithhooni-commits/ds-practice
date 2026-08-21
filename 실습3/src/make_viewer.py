"""Build a self-contained HTML viewer for the generated images.

The bare 0.png..9.png filenames carry no hint of which prompt produced them, and
comparing runs means opening a dozen folders side by side. This collapses all of it
into one page: pick a concept, see its reference images, then every prompt as its own
block with one card per run/method and the CLIP (and optionally DINO) scores on it.

Prompt-major on purpose. Both metrics are confounded per prompt -- text similarity
rises with prompt length, image similarity rises when the prompt's implied framing
matches the reference set -- so the only sound comparison is between methods *within*
one prompt. The layout puts exactly those cards next to each other.

Images are embedded as data URIs, so the file works anywhere with no server.

    # one run
    python make_viewer.py --source baseline=baseline_review/generated

    # compare runs, with per-image scores
    python make_viewer.py \
        --source baseline=baseline_review/generated --scores baseline=baseline_review/results/breakdown_all.csv \
        --source v2=v2_review/generated --scores v2=v2_review/results/breakdown_v2.csv \
        --out viewer.html

--scores takes the tidy CSV from eval_breakdown.py (concept,method,prompt_idx,t2i,i2i
and optionally dino). Without it the grid still renders, just without numbers.
"""

import argparse
import base64
import csv
import glob
import html
import io
import json
import os

from PIL import Image

CLASS_PROMPT = {  # kept in sync with evaluation.py's CLASS_PROMPT
    "actionfigure_2": "action figure",
    "decoritems_woodenpot": "wooden pot",
    "furniture_sofa2": "sofa",
    "instrument_music2": "guitar",
    "luggage_backpack1": "backpack",
    "person_3": "person",
    "pet_cat5": "cat",
    "scene_waterfall": "waterfall",
    "transport_tank": "tank",
    "wearable_jacket1": "jacket",
}

# display order; anything found outside this list is appended alphabetically
METHOD_ORDER = ["ti", "lora", "dora", "lora_prior", "dora_prior",
                "lora_bo1", "dora_bo1", "lora_prior_p03", "lora_r512",
                "lora_zwx", "dora_zwx"]

METRICS = [("t2i", "T2I"), ("i2i", "I2I"), ("dino", "DINO")]


def thumb_data_uri(path, size, quality):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def parse_pairs(values, what):
    out = []
    for v in values or []:
        if "=" not in v:
            raise SystemExit(f"--{what} needs LABEL=VALUE form, got {v!r}")
        label, _, rest = v.partition("=")
        out.append((label.strip(), rest.strip()))
    return out


def load_scores(score_pairs):
    """{(label, concept, method, idx): {metric: value}}"""
    scores = {}
    for label, path in score_pairs:
        if not os.path.exists(path):
            print(f"  [warn] scores for {label!r} not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vals = {}
                for key, _ in METRICS:
                    raw = (r.get(key) or "").strip()
                    if raw:
                        vals[key] = float(raw)
                scores[(label, r["concept"], r["method"], int(r["prompt_idx"]))] = vals
    return scores


def sorted_methods(names):
    known = [m for m in METHOD_ORDER if m in names]
    return known + sorted(n for n in names if n not in METHOD_ORDER)


def collect(sources, dataset_dir, prompts_dir, wanted_methods, size, quality, max_refs):
    concepts = []
    for concept in sorted(CLASS_PROMPT):
        prompt_path = os.path.join(prompts_dir, f"{concept}.txt")
        if not os.path.exists(prompt_path):
            continue
        with open(prompt_path, encoding="utf-8") as f:
            prompts = [l.strip().replace("{}", CLASS_PROMPT[concept]) for l in f if l.strip()]

        rows = []
        for label, root in sources:
            found = {m for m in wanted_methods
                     if glob.glob(os.path.join(root, f"{concept}_{m}", "*.png"))}
            for method in sorted_methods(found):
                cells = []
                for i in range(len(prompts)):
                    p = os.path.join(root, f"{concept}_{method}", f"{i}.png")
                    cells.append(thumb_data_uri(p, size, quality) if os.path.exists(p) else None)
                rows.append({"label": label, "method": method, "cells": cells})
        if not rows:
            continue

        all_refs = sorted(glob.glob(os.path.join(dataset_dir, concept, "*")))
        concepts.append({
            "name": concept,
            "noun": CLASS_PROMPT[concept],
            "prompts": prompts,
            "rows": rows,
            "refs": [thumb_data_uri(p, size, quality) for p in all_refs[:max_refs]],
            "n_refs": len(all_refs),
        })
        print(f"  [ok] {concept}: {len(rows)} run/method x {len(prompts)} prompts")
    return concepts


CSS = """
:root {
  --paper:#f6f7f9; --surface:#ffffff; --surface-2:#eceff4; --sunken:#f0f2f6;
  --ink:#12161c; --ink-2:#495364; --ink-3:#7c8797;
  --line:#d7dce4; --line-2:#e7ebf0;
  --accent:#0f6f7a; --accent-ink:#ffffff; --accent-soft:#e2f0f2;
  --win:#0f6f7a; --warn:#9a5b12;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:#0d1015; --surface:#161a21; --surface-2:#1d232b; --sunken:#11151b;
    --ink:#e9edf2; --ink-2:#9ba6b4; --ink-3:#6c7785;
    --line:#2a313b; --line-2:#212730;
    --accent:#4fc3cf; --accent-ink:#0d1015; --accent-soft:#153037;
    --win:#4fc3cf; --warn:#d9a24a;
  }
}
:root[data-theme="dark"] {
  --paper:#0d1015; --surface:#161a21; --surface-2:#1d232b; --sunken:#11151b;
  --ink:#e9edf2; --ink-2:#9ba6b4; --ink-3:#6c7785;
  --line:#2a313b; --line-2:#212730;
  --accent:#4fc3cf; --accent-ink:#0d1015; --accent-soft:#153037;
  --win:#4fc3cf; --warn:#d9a24a;
}

* { box-sizing:border-box; }
body {
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, sans-serif;
  font-size:15px; line-height:1.5;
}
.mono { font-family:"IBM Plex Mono", ui-monospace, Menlo, monospace; }
.num { font-family:"IBM Plex Mono", ui-monospace, Menlo, monospace; font-variant-numeric:tabular-nums; }
.eyebrow {
  font-size:10.5px; font-weight:600; letter-spacing:.09em;
  text-transform:uppercase; color:var(--ink-3);
}

.wrap { display:grid; grid-template-columns:212px minmax(0,1fr); }

/* ---- concept index ---- */
.rail {
  position:sticky; top:0; align-self:start; height:100vh; overflow-y:auto;
  border-right:1px solid var(--line); background:var(--surface); padding:20px 0 40px;
}
.rail h1 { font-size:14.5px; font-weight:600; margin:0 16px 3px; letter-spacing:-.01em; }
.rail .sub { margin:0 16px 18px; font-size:11.5px; color:var(--ink-3); }
.rail .eyebrow { margin:0 16px 7px; }
.rail button, .rail a {
  display:flex; justify-content:space-between; gap:8px; align-items:baseline; width:100%;
  padding:7px 16px; background:none; border:0; border-left:2px solid transparent;
  color:var(--ink-2); font:inherit; font-size:12.5px; text-align:left; cursor:pointer;
  text-decoration:none;
}
.rail button:hover, .rail a:hover { background:var(--surface-2); color:var(--ink); }
.rail button:focus-visible, .rail a:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
.rail button[aria-current="true"] {
  background:var(--accent-soft); border-left-color:var(--accent);
  color:var(--ink); font-weight:600;
}
.rail .n { font-size:10.5px; color:var(--ink-3); }

/* ---- header + filters ---- */
.bar {
  position:sticky; top:0; z-index:30; background:var(--surface);
  border-bottom:1px solid var(--line); padding:13px 26px;
}
.bar .top { display:flex; align-items:baseline; gap:11px; flex-wrap:wrap; margin-bottom:11px; }
.bar h2 { font-size:20px; font-weight:600; margin:0; letter-spacing:-.015em; }
.bar .cls { font-size:12.5px; color:var(--ink-2); }
.bar .cls code {
  font:400 11.5px/1 "IBM Plex Mono", monospace; background:var(--surface-2);
  padding:2px 6px; border-radius:2px;
}
.filters { display:flex; flex-wrap:wrap; gap:16px; align-items:center; }
.group { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
.chip {
  font:500 11.5px/1 "IBM Plex Mono", ui-monospace, monospace;
  padding:6px 10px; border:1px solid var(--line); border-radius:2px;
  background:var(--surface); color:var(--ink-2); cursor:pointer;
}
.chip:hover { border-color:var(--ink-3); color:var(--ink); }
.chip:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
.chip[aria-pressed="true"] { background:var(--accent); border-color:var(--accent); color:var(--accent-ink); }
.chip.all { border-style:dashed; }

main { padding:0 26px 100px; }
.panel { display:none; }
.panel.on { display:block; }

.block {
  background:var(--surface); border:1px solid var(--line); border-radius:3px;
  margin-top:18px; padding:16px 18px;
}
.block > .eyebrow { display:block; margin-bottom:11px; }

/* ---- summary table ---- */
.tscroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:12.5px; }
th, td { padding:7px 12px; text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
thead th { border-bottom:1px solid var(--line); color:var(--ink-3); font-weight:600; font-size:11px;
           letter-spacing:.04em; text-transform:uppercase; }
tbody tr + tr td { border-top:1px solid var(--line-2); }
tbody tr:hover td { background:var(--sunken); }
td .run { color:var(--ink-3); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
td.best { color:var(--win); font-weight:600; }

/* ---- reference strip ---- */
.strip { display:flex; gap:6px; overflow-x:auto; padding-bottom:3px; }
.strip img {
  height:76px; width:76px; object-fit:cover; border-radius:2px; flex:0 0 auto;
  border:1px solid var(--line); background:var(--surface-2); cursor:zoom-in;
}

/* ---- prompt blocks ---- */
.phead { display:flex; align-items:baseline; gap:10px; margin-bottom:12px; }
.pidx {
  font:600 11px/1 "IBM Plex Mono", monospace; color:var(--accent-ink);
  background:var(--accent); padding:4px 7px; border-radius:2px; flex:0 0 auto;
}
.ptext { font-size:13.5px; color:var(--ink); text-wrap:balance; }
.cards { display:grid; grid-template-columns:repeat(auto-fill, minmax(158px,1fr)); gap:11px; }
.card {
  border:1px solid var(--line); border-radius:2px; overflow:hidden;
  background:var(--sunken); display:flex; flex-direction:column;
}
.card .chead {
  display:flex; justify-content:space-between; gap:6px; align-items:baseline;
  padding:6px 8px; border-bottom:1px solid var(--line-2); background:var(--surface);
}
.card .m { font:600 11.5px/1.2 "IBM Plex Mono", monospace; }
.card .r { font-size:9.5px; letter-spacing:.05em; text-transform:uppercase; color:var(--ink-3); }
.card img { width:100%; aspect-ratio:1; object-fit:cover; display:block; cursor:zoom-in; }
.card .miss {
  aspect-ratio:1; display:grid; place-items:center; font-size:11px; color:var(--ink-3);
  background:var(--surface-2);
}
.sc { display:flex; gap:9px; padding:6px 8px; background:var(--surface); flex-wrap:wrap; }
.sc span { font-size:10.5px; }
.sc .k { color:var(--ink-3); letter-spacing:.03em; }
.sc .v { color:var(--ink-2); }
.sc .v.best { color:var(--win); font-weight:600; }

.note {
  margin-top:18px; padding:12px 15px; border-left:2px solid var(--accent);
  background:var(--surface); font-size:12.5px; color:var(--ink-2); max-width:78ch;
}
.note b { color:var(--ink); font-weight:600; }
.empty { padding:40px 0; color:var(--ink-3); font-size:13px; }

/* ---- lightbox ---- */
#lb { position:fixed; inset:0; z-index:100; display:none; background:rgba(8,10,14,.88); padding:28px; }
#lb.on { display:grid; place-items:center; grid-template-rows:1fr auto; gap:14px; cursor:zoom-out; }
#lb img { max-width:min(92vw,720px); max-height:78vh; border-radius:3px; }
#lb .cap { font:400 12.5px/1.4 "IBM Plex Mono", monospace; color:#c9d2dd; text-align:center; }

@media (max-width:900px) {
  .wrap { grid-template-columns:1fr; }
  .rail { position:static; height:auto; border-right:0; border-bottom:1px solid var(--line); }
  .rail button { border-left:0; border-bottom:2px solid transparent; }
}
@media (prefers-reduced-motion:reduce) { * { animation:none !important; transition:none !important; } }
"""

JS = """
var METRICS = window.__METRICS__;

function pressed(kind) {
  var s = new Set();
  document.querySelectorAll('.chip[data-kind="' + kind + '"]').forEach(function (c) {
    if (c.getAttribute('aria-pressed') === 'true') s.add(c.dataset.value);
  });
  return s;
}

function apply() {
  var runs = pressed('run'), methods = pressed('method');
  document.querySelectorAll('[data-run]').forEach(function (el) {
    el.hidden = !(runs.has(el.dataset.run) && methods.has(el.dataset.method));
  });
  // re-mark the winner per prompt and per summary column over visible rows only
  document.querySelectorAll('.cards').forEach(function (grid) {
    METRICS.forEach(function (m) {
      var best = null, cells = [];
      grid.querySelectorAll('.card:not([hidden]) .v[data-m="' + m + '"]').forEach(function (v) {
        v.classList.remove('best');
        var n = parseFloat(v.dataset.val);
        if (!isNaN(n)) { cells.push([n, v]); if (best === null || n > best) best = n; }
      });
      cells.forEach(function (p) { if (p[0] === best) p[1].classList.add('best'); });
    });
  });
  document.querySelectorAll('table.summary').forEach(function (t) {
    METRICS.forEach(function (m) {
      var best = null, cells = [];
      t.querySelectorAll('tbody tr:not([hidden]) td[data-m="' + m + '"]').forEach(function (td) {
        td.classList.remove('best');
        var n = parseFloat(td.dataset.val);
        if (!isNaN(n)) { cells.push([n, td]); if (best === null || n > best) best = n; }
      });
      cells.forEach(function (p) { if (p[0] === best) p[1].classList.add('best'); });
    });
  });
  document.querySelectorAll('.panel.on').forEach(function (p) {
    var any = p.querySelectorAll('.card:not([hidden])').length;
    var e = p.querySelector('.empty');
    if (e) e.hidden = any > 0;
  });
}

function show(name) {
  document.querySelectorAll('.panel').forEach(function (p) {
    p.classList.toggle('on', p.dataset.concept === name);
  });
  document.querySelectorAll('.rail button').forEach(function (b) {
    b.setAttribute('aria-current', String(b.dataset.concept === name));
  });
  document.getElementById('ctitle').textContent = name;
  document.getElementById('cnoun').innerHTML =
    'class prompt <code>' + window.__NOUNS__[name] + '</code>';
  window.scrollTo(0, 0);
  apply();
}

document.querySelectorAll('.rail button').forEach(function (b) {
  b.addEventListener('click', function () { show(b.dataset.concept); });
});
document.querySelectorAll('.chip').forEach(function (c) {
  c.addEventListener('click', function () {
    if (c.classList.contains('all')) {
      var on = c.dataset.state !== 'on';
      c.dataset.state = on ? 'on' : 'off';
      document.querySelectorAll('.chip[data-kind="' + c.dataset.kind + '"]').forEach(function (o) {
        o.setAttribute('aria-pressed', String(on));
      });
    } else {
      c.setAttribute('aria-pressed', c.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
    }
    apply();
  });
});

var lb = document.getElementById('lb');
document.addEventListener('click', function (e) {
  if (e.target.tagName === 'IMG' && e.target.closest('.card, .strip')) {
    lb.querySelector('img').src = e.target.src;
    lb.querySelector('.cap').textContent = e.target.alt;
    lb.classList.add('on');
  } else if (e.target.closest('#lb')) {
    lb.classList.remove('on');
  }
});
document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape') lb.classList.remove('on');
});

show(document.querySelector('.rail button').dataset.concept);
"""


def render(concepts, sources, scores, title, links=None):
    """links: optional [(concept, href_or_None)] -- href None means "this page", which
    is rendered as the button the JS drives; the rest become plain links to sibling
    files. Used by --split, where one 18MB document is replaced by one file per concept."""
    e = html.escape
    runs = [lab for lab, _ in sources]
    methods = sorted_methods({r["method"] for c in concepts for r in c["rows"]})
    used = [(k, lab) for k, lab in METRICS
            if any(k in v for v in scores.values())] or [METRICS[0], METRICS[1]]

    # full document skeleton: this file is opened straight off disk, so without a
    # doctype the browser drops into quirks mode and the sticky/full-height layout
    # collapses, and without a charset the Korean prompt text mojibakes.
    o = ['<!doctype html>', '<html lang="en">', "<head>",
         '<meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width, initial-scale=1">',
         f"<title>{e(title)}</title>",
         '<link rel="preconnect" href="https://fonts.googleapis.com">',
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
         f"<style>{CSS}</style>", "</head>", "<body>", '<div class="wrap">']

    n_concepts = len(links) if links else len(concepts)
    o.append('<nav class="rail"><h1>Concept personalization</h1>'
             f'<p class="sub">{n_concepts} concepts &middot; {len(runs)} run'
             f'{"s" if len(runs) != 1 else ""} &middot; SD3.5-medium</p>'
             '<p class="eyebrow">Concepts</p>')
    if links:
        rows = {c["name"]: len(c["rows"]) for c in concepts}
        for name, href in links:
            n = f'<span class="n mono">{rows[name]}</span>' if name in rows else ""
            if href is None:
                o.append(f'<button data-concept="{e(name)}"><span>{e(name)}</span>{n}</button>')
            else:
                o.append(f'<a href="{e(href)}"><span>{e(name)}</span>{n}</a>')
    else:
        for c in concepts:
            o.append(f'<button data-concept="{e(c["name"])}"><span>{e(c["name"])}</span>'
                     f'<span class="n mono">{len(c["rows"])}</span></button>')
    o.append("</nav><div>")

    # ---- filter bar (shared across concepts) ----
    o.append('<div class="bar"><div class="top">'
             f'<h2 id="ctitle">&mdash;</h2><span class="cls" id="cnoun"></span></div>'
             '<div class="filters"><div class="group"><span class="eyebrow">Run</span>')
    for r in runs:
        o.append(f'<button class="chip" aria-pressed="true" data-kind="run" data-value="{e(r)}">{e(r)}</button>')
    o.append('</div><div class="group"><span class="eyebrow">Method</span>')
    for m in methods:
        o.append(f'<button class="chip" aria-pressed="true" data-kind="method" data-value="{e(m)}">{e(m)}</button>')
    o.append('<button class="chip all" data-kind="method" data-state="on">all</button>')
    o.append("</div></div></div><main>")

    for c in concepts:
        o.append(f'<section class="panel" data-concept="{e(c["name"])}">')

        # summary first: per run/method means over the 10 prompts
        o.append('<div class="block"><span class="eyebrow">Mean over 10 prompts</span>'
                 '<div class="tscroll"><table class="summary"><thead><tr><th>Run / method</th>')
        for _, lab in used:
            o.append(f"<th>{lab}</th>")
        o.append("</tr></thead><tbody>")
        for row in c["rows"]:
            got = [scores.get((row["label"], c["name"], row["method"], i), {})
                   for i in range(len(c["prompts"]))]
            o.append(f'<tr data-run="{e(row["label"])}" data-method="{e(row["method"])}">'
                     f'<td><span class="run">{e(row["label"])}</span> '
                     f'<b class="mono">{e(row["method"])}</b></td>')
            for k, _ in used:
                vals = [g[k] for g in got if k in g]
                if vals:
                    mean = sum(vals) / len(vals)
                    o.append(f'<td class="num" data-m="{k}" data-val="{mean:.6f}">{mean:.4f}</td>')
                else:
                    o.append(f'<td class="num" data-m="{k}">&mdash;</td>')
            o.append("</tr>")
        o.append("</tbody></table></div></div>")

        if c["refs"]:
            shown = "" if len(c["refs"]) == c["n_refs"] else f", showing {len(c['refs'])}"
            o.append('<div class="block"><span class="eyebrow">Reference &mdash; '
                     f'{c["n_refs"]} image{"s" if c["n_refs"] != 1 else ""}{shown}'
                     '</span><div class="strip">')
            for j, r in enumerate(c["refs"]):
                o.append(f'<img src="{r}" alt="{e(c["name"])} reference {j}" loading="lazy">')
            o.append("</div></div>")

        for i, prompt in enumerate(c["prompts"]):
            o.append('<div class="block"><div class="phead">'
                     f'<span class="pidx">{i:02d}</span><span class="ptext">{e(prompt)}</span></div>'
                     '<div class="cards">')
            for row in c["rows"]:
                uri = row["cells"][i] if i < len(row["cells"]) else None
                s = scores.get((row["label"], c["name"], row["method"], i), {})
                o.append(f'<div class="card" data-run="{e(row["label"])}" data-method="{e(row["method"])}">'
                         f'<div class="chead"><span class="m">{e(row["method"])}</span>'
                         f'<span class="r">{e(row["label"])}</span></div>')
                if uri:
                    alt = f'{c["name"]} / {row["label"]} / {row["method"]} / prompt {i}: {prompt}'
                    o.append(f'<img src="{uri}" alt="{e(alt)}" loading="lazy">')
                else:
                    o.append('<div class="miss">no image</div>')
                if s:
                    o.append('<div class="sc num">')
                    for k, lab in used:
                        if k in s:
                            o.append(f'<span><span class="k">{lab}</span> '
                                     f'<span class="v" data-m="{k}" data-val="{s[k]:.6f}">{s[k]:.3f}</span></span>')
                    o.append("</div>")
                o.append("</div>")
            o.append("</div></div>")

        o.append('<p class="empty" hidden>No run/method selected &mdash; turn one back on above.</p>')
        o.append("</section>")

    o.append("</main></div></div>")
    o.append('<div id="lb"><img src="" alt=""><p class="cap"></p></div>')
    names = json.dumps({c["name"]: c["noun"] for c in concepts})
    o.append(f"<script>window.__METRICS__={json.dumps([k for k, _ in used])};"
             f"window.__NOUNS__={names};</script>")
    o.append(f"<script>{JS}</script>")
    o.append("</body></html>")
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, metavar="LABEL=DIR",
                    help="a run to include, e.g. v2=generated (repeatable)")
    ap.add_argument("--scores", action="append", metavar="LABEL=CSV",
                    help="eval_breakdown.py CSV for that run (repeatable, optional)")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--prompts", default="prompts")
    ap.add_argument("--methods", default=",".join(METHOD_ORDER))
    ap.add_argument("--thumb", type=int, default=280, help="embedded thumbnail size in px")
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--max_refs", type=int, default=12)
    ap.add_argument("--title", default="Concept Personalization Runs")
    ap.add_argument("--out", default="viewer.html")
    ap.add_argument("--split", action="store_true",
                    help="write one file per concept next to --out instead of a single "
                         "document. Ten concepts of embedded thumbnails is ~18MB in one "
                         "file, which browsers open slowly; split files are ~1-2MB each")
    args = ap.parse_args()

    sources = parse_pairs(args.source, "source")
    for label, root in sources:
        if not os.path.isdir(root):
            raise SystemExit(f"source {label!r}: no such directory {root}")
    scores = load_scores(parse_pairs(args.scores, "scores"))
    methods = [m for m in args.methods.split(",") if m]

    print(f"Embedding thumbnails at {args.thumb}px q{args.quality}...")
    concepts = collect(sources, args.dataset, args.prompts, methods,
                       args.thumb, args.quality, args.max_refs)
    if not concepts:
        raise SystemExit("nothing to render -- check --source paths and --methods")

    n_img = sum(len([x for x in r["cells"] if x]) for c in concepts for r in c["rows"])
    n_img += sum(len(c["refs"]) for c in concepts)

    if not args.split:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(render(concepts, sources, scores, args.title))
        print(f"\nWrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB, {n_img} images)")
        return

    stem, ext = os.path.splitext(args.out)
    ext = ext or ".html"
    paths = {c["name"]: f"{os.path.basename(stem)}_{c['name']}{ext}" for c in concepts}
    total = 0
    print()
    for c in concepts:
        links = [(o["name"], None if o is c else paths[o["name"]]) for o in concepts]
        out = os.path.join(os.path.dirname(stem) or ".", paths[c["name"]])
        with open(out, "w", encoding="utf-8") as f:
            f.write(render([c], sources, scores, f'{args.title} — {c["name"]}', links))
        total += os.path.getsize(out)
        print(f"  {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    first = os.path.join(os.path.dirname(stem) or ".", paths[concepts[0]["name"]])
    print(f"\nWrote {len(concepts)} files ({total / 1e6:.1f} MB total, {n_img} images)."
          f"\nOpen {first} -- the left rail links to the rest.")


if __name__ == "__main__":
    main()
