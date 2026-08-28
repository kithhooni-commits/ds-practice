"""Run the punch detector over every dataset case and print a scoring table.

A case directory contains landmarks.jsonl (+ labels.json for graded cases).
Use this after any change to the detection pipeline to catch regressions.

Usage:
  python iter4/eval/run_suite.py [--datasets iter4/eval/datasets] [--out suite_report.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_video import PunchEvaluator
from scoring import iter_landmarks_jsonl, load_labels_file, score_predictions


def run_case(case_dir):
    jsonl = case_dir / "landmarks.jsonl"
    if not jsonl.exists():
        return None
    evaluator = PunchEvaluator(calib=True)
    for ts_ms, lm, wl in iter_landmarks_jsonl(jsonl):
        evaluator.process(lm, wl or [], ts_ms)
    result = {
        "case": case_dir.name,
        "predictions": evaluator.events,
        "windows_opened": sum(st.windows_opened for st in evaluator.arms.values()),
        "windows_expired": evaluator.windows_expired,
    }
    labels_path = case_dir / "labels.json"
    if labels_path.exists():
        result["scoring"] = score_predictions(evaluator.events, load_labels_file(labels_path))
    return result


def fmt(val, pattern="{:.2f}"):
    return "-" if val is None else pattern.format(val)


def main():
    ap = argparse.ArgumentParser(description="Evaluate all dataset cases")
    ap.add_argument("--datasets", default=str(Path(__file__).parent / "datasets"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "output" / "suite_report.json"))
    args = ap.parse_args()

    root = Path(args.datasets)
    case_dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []
    if not case_dirs:
        print(f"케이스 없음: {root}")
        raise SystemExit(1)

    results = []
    for case_dir in case_dirs:
        result = run_case(case_dir)
        if result is not None:
            results.append(result)

    header = f"{'case':<20}{'정답':>5}{'예측':>5}{'TP':>4}{'FP':>4}{'FN':>4}{'P':>7}{'R':>7}{'F1':>7}{'종류':>7}{'Δt(ms)':>9}"
    print(header)
    print("-" * len(header))
    totals = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
    all_pass = True
    for r in results:
        s = r.get("scoring")
        if s is None:
            print(f"{r['case']:<20}{len(r['predictions']):>5}  (라벨 없음 — 통계만)")
            continue
        ok = s["fp"] == 0 and s["fn"] == 0
        all_pass = all_pass and ok
        mark = "" if ok else "  ← 확인"
        print(
            f"{r['case']:<20}{s['ground_truth']:>5}{s['predicted']:>5}"
            f"{s['tp']:>4}{s['fp']:>4}{s['fn']:>4}"
            f"{fmt(s['precision'], '{:.3f}'):>7}{fmt(s['recall'], '{:.3f}'):>7}{fmt(s['f1'], '{:.3f}'):>7}"
            f"{fmt(s['kind_accuracy'], '{:.2f}'):>7}{fmt(s['timing_error_ms_mean'], '{:.0f}'):>9}{mark}"
        )
        totals["gt"] += s["ground_truth"]
        totals["pred"] += s["predicted"]
        totals["tp"] += s["tp"]
        totals["fp"] += s["fp"]
        totals["fn"] += s["fn"]

    gt, tp, fp, fn = totals["gt"], totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) else None
    print("-" * len(header))
    print(f"{'합계':<20}{gt:>5}{totals['pred']:>5}{tp:>4}{fp:>4}{fn:>4}"
          f"{fmt(precision, '{:.3f}'):>7}{fmt(recall, '{:.3f}'):>7}{fmt(f1, '{:.3f}'):>7}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n리포트 저장: {out_path}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
