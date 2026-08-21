"""Trigger-token comparison table: `sks` (v2) vs `zwx` (zwx10), 10 concepts x lora/dora.

FINDINGS 4.7 tested `zwx` on three concepts and predicted that the gain is confined to
the concepts where `sks` visibly contaminated the output -- that the clean six would
show no benefit. The full ten-concept run makes that prediction falsifiable, so the
table is grouped by contamination status rather than reported as one average.

    python zwx_table.py                       # print tables
    python zwx_table.py --md zwx_table.md     # also write markdown
"""

import argparse
import csv
import math
import sys
from collections import defaultdict

# the tables are Korean and use em dashes; a cp949 console would refuse them
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SKS_CSV = "v2_review/breakdown_v2.csv"
ZWX_CSV = "zwx10_review/breakdown_zwx10.csv"

# from the 0-prompt visual scan, compare/sks_scan_p0.jpg (FINDINGS 4.7)
CONTAMINATED = {
    "person_3": "군복+소총",
    "actionfigure_2": "전투복+소총",
    "scene_waterfall": "소총 부유",
    "luggage_backpack1": "밀리터리 재킷 (약)",
}
# transport_tank is excluded from both groups: its `tank` class noun carries its own
# military prior (FINDINGS 5.4), so `sks` and the class noun cannot be separated here.
SPECIAL = {"transport_tank": "class noun 자체가 군사 prior"}

METRICS = ["t2i", "i2i", "dino"]


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["concept"], r["method"], int(r["prompt_idx"]))
            rows[key] = {m: float(r[m]) for m in METRICS if r.get(m) not in (None, "")}
    return rows


def sign_test(k, n):
    """Two-sided exact binomial, p=0.5. Counts direction only -- it deliberately throws
    magnitude away, which is why 4.7's person_3 case reads as a loss."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(tail / 2 ** n * 2, 1.0)


def deltas(sks, zwx, concepts, base_method, zwx_method):
    """Per-prompt zwx - sks, pooled over the given concepts."""
    out = defaultdict(list)
    for c in concepts:
        for i in range(10):
            a = sks.get((c, base_method, i))
            b = zwx.get((c, zwx_method, i))
            if not a or not b:
                continue
            for m in METRICS:
                if m in a and m in b:
                    out[m].append(b[m] - a[m])
    return out


def concept_means(rows, concept, method):
    vals = defaultdict(list)
    for i in range(10):
        r = rows.get((concept, method, i))
        if r:
            for m, v in r.items():
                vals[m].append(v)
    return {m: sum(v) / len(v) for m, v in vals.items() if v}


def fmt_delta(d):
    n = len(d)
    if n == 0:
        return "—"
    up = sum(1 for x in d if x > 0)
    mean = sum(d) / n
    p = sign_test(up, n)
    star = "**" if p < 0.05 else ""
    return f"{star}{mean:+.4f}{star} ({up}/{n}, p={p:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", help="also write the tables as markdown to this path")
    args = ap.parse_args()

    sks, zwx = load(SKS_CSV), load(ZWX_CSV)
    concepts = sorted({c for (c, _, _) in zwx})
    groups = [
        ("오염 4컨셉", [c for c in concepts if c in CONTAMINATED]),
        ("깨끗한 5컨셉", [c for c in concepts if c not in CONTAMINATED and c not in SPECIAL]),
        ("transport_tank (별도)", [c for c in concepts if c in SPECIAL]),
        ("전체 10컨셉", concepts),
    ]

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# 트리거 토큰 `sks` → `zwx` — 10컨셉 평가")
    emit()
    emit(f"비교: `{SKS_CSV}` (sks) vs `{ZWX_CSV}` (zwx). rank 16 / 256 학습 / 512 생성 /")
    emit("best-of-4 동일, 트리거 토큰만 다름. Δ는 프롬프트 단위 짝지은 차이(zwx − sks)이고")
    emit("괄호는 상승 프롬프트 수와 양측 부호검정 p.")
    emit()

    for label, meth_s, meth_z in [("LoRA", "lora", "lora_zwx"), ("DoRA", "dora", "dora_zwx")]:
        emit(f"## {label}")
        emit()
        emit("| 그룹 | 컨셉 | ΔT2I | ΔCLIP I2I | ΔDINO |")
        emit("|---|---|---|---|---|")
        for gname, gcs in groups:
            if not gcs:
                continue
            d = deltas(sks, zwx, gcs, meth_s, meth_z)
            emit(f"| {gname} | {len(gcs)} | " +
                 " | ".join(fmt_delta(d[m]) for m in METRICS) + " |")
        emit()
        emit("컨셉 단위(프롬프트 평균을 컨셉당 1표로 집계). 한 컨셉의 프롬프트 10개는 같은")
        emit("가중치에서 나오므로 서로 독립이 아니다 — 위 표의 100개짜리 p는 유의성을")
        emit("과대평가하며, 아래가 올바른 검정 단위다:")
        emit()
        emit("| 그룹 | ΔT2I | ΔCLIP I2I | ΔDINO |")
        emit("|---|---|---|---|")
        for gname, gcs in groups:
            if len(gcs) < 2:
                continue
            per = {m: [] for m in METRICS}
            for c in gcs:
                d = deltas(sks, zwx, [c], meth_s, meth_z)
                for m in METRICS:
                    if d[m]:
                        per[m].append(sum(d[m]) / len(d[m]))
            emit(f"| {gname} | " + " | ".join(fmt_delta(per[m]) for m in METRICS) + " |")
        emit()

    emit("## 컨셉별 상세 (LoRA)")
    emit()
    emit("| 컨셉 | 오염 | sks T2I/I2I/DINO | zwx T2I/I2I/DINO | ΔI2I | ΔDINO |")
    emit("|---|---|---|---|---|---|")
    for c in concepts:
        a, b = concept_means(sks, c, "lora"), concept_means(zwx, c, "lora_zwx")
        if not a or not b:
            continue
        d = deltas(sks, zwx, [c], "lora", "lora_zwx")
        tag = CONTAMINATED.get(c, SPECIAL.get(c, "—"))
        emit(f"| {c} | {tag} | "
             f"{a['t2i']:.4f} / {a['i2i']:.4f} / {a.get('dino', float('nan')):.4f} | "
             f"{b['t2i']:.4f} / {b['i2i']:.4f} / {b.get('dino', float('nan')):.4f} | "
             f"{fmt_delta(d['i2i'])} | {fmt_delta(d['dino'])} |")
    emit()

    emit("## 방법별 절대값 (10컨셉 평균)")
    emit()
    emit("| 방법 | T2I | CLIP I2I | DINO |")
    emit("|---|---|---|---|")
    for src, rows, meths in [("sks", sks, ["lora", "dora"]), ("zwx", zwx, ["lora_zwx", "dora_zwx"])]:
        for m in meths:
            agg = defaultdict(list)
            for c in concepts:
                for k, v in concept_means(rows, c, m).items():
                    agg[k].append(v)
            if not agg:
                continue
            emit(f"| {m} | " + " | ".join(
                f"{sum(agg[k]) / len(agg[k]):.4f}" if agg.get(k) else "—" for k in METRICS) + " |")
    emit()

    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[wrote] {args.md}")


if __name__ == "__main__":
    main()
