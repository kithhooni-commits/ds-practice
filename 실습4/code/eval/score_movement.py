"""
score_movement.py — 이동(FORWARD/BACK/LEFT/RIGHT) · 회전(ROT_LEFT/ROT_RIGHT) 룰베이스
채점기. 펀치의 scoring.py(이벤트 매칭)와 짝을 이루는, "상태(state) 구간" 채점기다.

입력:
  --report   evaluate_full_actions.py 가 만든 report.json (state_timeline 필요)
  --labels   RECORDING_GUIDE_MOVEMENT.md 스키마의 GT 라벨 json (move_segments/rot_segments/negative_windows)

계산하는 것:
  1. move 축 / rot 축 각각 프레임 단위 confusion matrix + per-class P/R/F1
     (한 시점의 "정답 상태"와 "예측 상태"를 프레임마다 비교 — 펀치처럼 tolerance_ms 안의
     이벤트 하나만 맞히면 되는 게 아니라, "그 순간 계속 맞는 상태를 유지하고 있는가"를 본다)
  2. 각 라벨 구간(segment)에 대한 onset/offset latency
     - onset: 구간 시작 이후 예측이 해당 state로 "확정"되기까지 걸린 시간
     - offset: 구간 종료 이후 예측이 NONE으로 돌아오기까지 걸린 시간
  3. negative_windows 안에서 move/rot 이 NONE이 아니었던 프레임 비율 (오검출률)

Usage:
  python eval/score_movement.py \
    --report eval/output/full_pipeline_report.json \
    --labels eval/video/benchmark_movement_labels.json \
    --out eval/output/movement_report.json
"""
import argparse
import io
import json
import sys
from pathlib import Path

# Windows 콘솔 기본 코드페이지(cp949)는 이모지/전각문자를 못 찍는다 — UTF-8로 강제.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MOVE_CLASSES = ["NONE", "FORWARD", "BACK", "LEFT", "RIGHT"]
ROT_CLASSES = ["NONE", "ROT_LEFT", "ROT_RIGHT"]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_gt_lookup(segments):
    """[{start_ms,end_ms,state}] 구간 리스트 → t_ms를 넣으면 정답 state를 돌려주는 함수.
    구간 밖은 전부 NONE."""
    segs = sorted(segments, key=lambda s: s["start_ms"])

    def gt_at(t_ms):
        for s in segs:
            if s["start_ms"] <= t_ms < s["end_ms"]:
                return s["state"]
        return "NONE"

    return gt_at


def confusion_and_prf(pairs, classes):
    """pairs: [(gt, pred), ...] → confusion matrix(dict) + per-class P/R/F1"""
    cm = {g: {p: 0 for p in classes} for g in classes}
    for gt, pred in pairs:
        if gt not in cm:
            continue
        if pred not in cm[gt]:
            pred = "NONE"  # 알 수 없는 라벨은 NONE으로 취급 (안전한 폴백)
        cm[gt][pred] += 1

    per_class = {}
    for c in classes:
        tp = cm[c][c]
        fn = sum(cm[c][p] for p in classes if p != c)
        fp = sum(cm[g][c] for g in classes if g != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and (precision + recall) > 0 else None)
        per_class[c] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    total = len(pairs)
    correct = sum(1 for gt, pred in pairs if gt == pred)
    accuracy = correct / total if total else None
    return cm, per_class, accuracy


def segment_latencies(segments, timeline, tolerance_ms, axis_key):
    """구간별 onset/offset latency(ms). 못 잡은 구간은 latency=None, missed=True."""
    results = []
    for seg in segments:
        state = seg["state"]
        start, end = seg["start_ms"], seg["end_ms"]

        onset = None
        for x in timeline:
            if x["t_ms"] >= start and x[axis_key] == state:
                onset = x["t_ms"] - start
                break
            if x["t_ms"] >= end:
                break

        offset = None
        if onset is not None:
            for x in timeline:
                if x["t_ms"] >= end and x[axis_key] != state:
                    offset = x["t_ms"] - end
                    break

        results.append({
            "desc": seg.get("desc", ""),
            "state": state,
            "start_ms": start,
            "end_ms": end,
            "onset_latency_ms": onset,
            "offset_latency_ms": offset,
            "missed": onset is None,
            "onset_within_tolerance": (onset is not None and onset <= tolerance_ms),
        })
    return results


def negative_window_fp_rate(windows, timeline, axis_key):
    total_frames = 0
    fp_frames = 0
    per_window = []
    for w in windows:
        frames = [x for x in timeline if w["start_ms"] <= x["t_ms"] < w["end_ms"]]
        bad = sum(1 for x in frames if x[axis_key] != "NONE")
        total_frames += len(frames)
        fp_frames += bad
        per_window.append({
            "desc": w.get("desc", ""),
            "start_ms": w["start_ms"], "end_ms": w["end_ms"],
            "frames": len(frames), "false_positive_frames": bad,
            "fp_rate": (bad / len(frames)) if frames else None,
        })
    overall = (fp_frames / total_frames) if total_frames else None
    return overall, per_window


def main():
    ap = argparse.ArgumentParser(description="Score rule-base movement/rotation recognition against segment-labeled GT")
    ap.add_argument("--report", required=True, help="evaluate_full_actions.py 의 report.json")
    ap.add_argument("--labels", required=True, help="benchmark_movement_labels.json")
    ap.add_argument("--out", default=None, help="채점 결과 저장 경로 (JSON)")
    args = ap.parse_args()

    report = load_json(args.report)
    labels = load_json(args.labels)

    timeline = report.get("state_timeline")
    if not timeline:
        raise SystemExit("report.json 에 state_timeline 이 없습니다 — evaluate_full_actions.py 를 최신 버전으로 다시 돌리세요.")

    tol = labels.get("onset_tolerance_ms", 250)

    move_gt = build_gt_lookup(labels.get("move_segments", []))
    rot_gt = build_gt_lookup(labels.get("rot_segments", []))

    move_pairs = [(move_gt(x["t_ms"]), x["move"]) for x in timeline]
    rot_pairs = [(rot_gt(x["t_ms"]), x["rot"]) for x in timeline]

    move_cm, move_prf, move_acc = confusion_and_prf(move_pairs, MOVE_CLASSES)
    rot_cm, rot_prf, rot_acc = confusion_and_prf(rot_pairs, ROT_CLASSES)

    move_latencies = segment_latencies(labels.get("move_segments", []), timeline, tol, "move")
    rot_latencies = segment_latencies(labels.get("rot_segments", []), timeline, tol, "rot")

    move_fp_overall, move_fp_windows = negative_window_fp_rate(labels.get("negative_windows", []), timeline, "move")
    rot_fp_overall, rot_fp_windows = negative_window_fp_rate(labels.get("negative_windows", []), timeline, "rot")

    result = {
        "source_report": args.report,
        "source_labels": args.labels,
        "move_axis": {
            "frame_accuracy": move_acc,
            "confusion_matrix": move_cm,
            "per_class": move_prf,
            "segment_latencies": move_latencies,
            "negative_window_fp_rate": move_fp_overall,
            "negative_windows": move_fp_windows,
        },
        "rot_axis": {
            "frame_accuracy": rot_acc,
            "confusion_matrix": rot_cm,
            "per_class": rot_prf,
            "segment_latencies": rot_latencies,
            "negative_window_fp_rate": rot_fp_overall,
            "negative_windows": rot_fp_windows,
        },
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 콘솔 요약 ---
    print("=" * 70)
    print("📊 이동/회전 룰베이스 채점 결과")
    print("=" * 70)
    for axis_name, prf, acc, latencies, fp in (
        ("MOVE (FORWARD/BACK/LEFT/RIGHT)", move_prf, move_acc, move_latencies, move_fp_overall),
        ("ROT (ROT_LEFT/ROT_RIGHT)", rot_prf, rot_acc, rot_latencies, rot_fp_overall),
    ):
        print(f"\n[{axis_name}]")
        print(f"  프레임 정확도: {acc:.3f}" if acc is not None else "  프레임 정확도: -")
        for cls, m in prf.items():
            if cls == "NONE":
                continue
            p = f"{m['precision']:.2f}" if m["precision"] is not None else "-"
            r = f"{m['recall']:.2f}" if m["recall"] is not None else "-"
            f1 = f"{m['f1']:.2f}" if m["f1"] is not None else "-"
            print(f"    {cls:<10} P={p} R={r} F1={f1} (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
        missed = sum(1 for x in latencies if x["missed"])
        onsets = [x["onset_latency_ms"] for x in latencies if x["onset_latency_ms"] is not None]
        avg_onset = sum(onsets) / len(onsets) if onsets else None
        print(f"  구간 {len(latencies)}개 중 미검출 {missed}개"
              + (f", 평균 onset latency {avg_onset:.0f}ms" if avg_onset is not None else ""))
        print(f"  오탐 구간(negative_windows) FP 프레임 비율: "
              + (f"{fp*100:.1f}%" if fp is not None else "-"))
    print("=" * 70)
    if args.out:
        print(f"저장: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
