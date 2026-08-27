#!/usr/bin/env python3
"""Benchmark Version Comparison & Diff Report Tool.

Reads runs_registry.json and outputs a comparative table of all evaluated algorithm versions.

Usage:
  python iter3/eval/compare_versions.py
  python iter3/eval/compare_versions.py v1_baseline v2_anti_sway
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REGISTRY_PATH = SCRIPT_DIR / "runs" / "runs_registry.json"


def load_registry():
    if not REGISTRY_PATH.exists():
        print(f"❌ 버전 레지스트리 없음: {REGISTRY_PATH}")
        print("   먼저 `python iter3/eval/run_pipeline.py --version v1_baseline`을 실행하세요.")
        sys.exit(1)
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ 레지스트리 읽기 실패: {e}")
        sys.exit(1)


def print_comparison_table(runs, baseline_ver=None, candidate_ver=None):
    if not runs:
        print("등록된 평가 버전이 없습니다.")
        return

    print("=" * 85)
    print("🥊 Boxing Action Recognition Benchmark — Multi-Version Leaderboard")
    print("=" * 85)

    headers = f"{'Version':<18} | {'F1-Score':<8} | {'Precision':<9} | {'Recall':<8} | {'TP/FP/FN':<10} | {'Rest/Foot FP':<12} | {'Latency':<7}"
    print(headers)
    print("-" * 85)

    for r in runs:
        ver = r.get("version", "unknown")
        f1 = f"{r.get('f1', 0.0):.4f}" if r.get('f1') is not None else "--"
        prec = f"{r.get('precision', 0.0):.4f}" if r.get('precision') is not None else "--"
        rec = f"{r.get('recall', 0.0):.4f}" if r.get('recall') is not None else "--"
        tp_fp_fn = f"{r.get('tp',0)}/{r.get('fp',0)}/{r.get('fn',0)}"
        non_act = f"{r.get('non_action_fp', 0)}회"
        lat = f"{r.get('timing_err_ms', 0.0):.1f}ms"

        row = f"{ver:<18} | {f1:<8} | {prec:<9} | {rec:<8} | {tp_fp_fn:<10} | {non_act:<12} | {lat:<7}"
        print(row)

    print("=" * 85)

    # 2개 버전 지정 시 A/B Delta 상세 비교
    if baseline_ver and candidate_ver:
        r_base = next((r for r in runs if r["version"] == baseline_ver), None)
        r_cand = next((r for r in runs if r["version"] == candidate_ver), None)
        if r_base and r_cand:
            print()
            print(f"🔍 [A/B Diff Analysis] Baseline ({baseline_ver}) ➔ Candidate ({candidate_ver})")
            print("-" * 65)
            d_f1 = (r_cand.get('f1', 0) or 0) - (r_base.get('f1', 0) or 0)
            d_prec = (r_cand.get('precision', 0) or 0) - (r_base.get('precision', 0) or 0)
            d_rec = (r_cand.get('recall', 0) or 0) - (r_base.get('recall', 0) or 0)
            d_fp = r_cand.get('fp', 0) - r_base.get('fp', 0)
            d_non_fp = r_cand.get('non_action_fp', 0) - r_base.get('non_action_fp', 0)

            print(f"• F1-Score       : {r_base.get('f1',0):.4f} ➔ {r_cand.get('f1',0):.4f} ({'+' if d_f1>=0 else ''}{d_f1:.4f}) {'🟢' if d_f1>0 else '🔴'}")
            print(f"• Precision      : {r_base.get('precision',0):.4f} ➔ {r_cand.get('precision',0):.4f} ({'+' if d_prec>=0 else ''}{d_prec:.4f}) {'🟢' if d_prec>0 else '🔴'}")
            print(f"• Recall         : {r_base.get('recall',0):.4f} ➔ {r_cand.get('recall',0):.4f} ({'+' if d_rec>=0 else ''}{d_rec:.4f})")
            print(f"• Total FP       : {r_base.get('fp',0)}회 ➔ {r_cand.get('fp',0)}회 ({'+' if d_fp>=0 else ''}{d_fp}회) {'🟢' if d_fp<0 else '🔴'}")
            print(f"• Non-Action FP  : {r_base.get('non_action_fp',0)}회 ➔ {r_cand.get('non_action_fp',0)}회 ({'+' if d_non_fp>=0 else ''}{d_non_fp}회) {'🟢' if d_non_fp<0 else '🔴'}")
            print("-" * 65)


def main():
    ap = argparse.ArgumentParser(description="Compare evaluated algorithm versions")
    ap.add_argument("baseline", nargs="?", default=None, help="Baseline version (optional)")
    ap.add_argument("candidate", nargs="?", default=None, help="Candidate version to compare (optional)")
    args = ap.parse_args()

    reg = load_registry()
    runs = reg.get("runs", [])
    print_comparison_table(runs, args.baseline, args.candidate)


if __name__ == "__main__":
    main()
