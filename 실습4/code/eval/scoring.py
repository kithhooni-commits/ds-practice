"""Label matching and metric computation for punch evaluation.

labels.json schema:
  {"tolerance_ms": 350,
   "punches": [{"t_ms": 6250, "side": "L", "kind": "STRAIGHT"}, ...]}

Predictions use the same side/kind vocabulary as the runtime detector.
"""
import json
import statistics
from pathlib import Path


class ArrayPoint:
    def __init__(self, arr):
        self.x = arr[0]
        self.y = arr[1]
        self.z = arr[2]
        self.visibility = arr[3] if len(arr) > 3 else 1.0


def to_landmark_objects(rows):
    if rows is None:
        return None
    return [ArrayPoint(r) for r in rows]


def load_labels_file(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tolerance = int(data.get("tolerance_ms", 350))
    punches = [
        {
            "t_ms": int(p["t_ms"]),
            "side": p.get("side"),
            "kind": p.get("kind"),
        }
        for p in data.get("punches", [])
    ]
    punches.sort(key=lambda p: p["t_ms"])
    return {"tolerance_ms": tolerance, "punches": punches}


def score_predictions(predictions, labels):
    tolerance = labels["tolerance_ms"]
    truth = labels["punches"]
    used = set()
    matches = []
    false_positives = []
    for pred in sorted(predictions, key=lambda p: p["t_ms"]):
        best_i, best_dt = None, tolerance + 1
        for i, lab in enumerate(truth):
            if i in used:
                continue
            dt = abs(pred["t_ms"] - lab["t_ms"])
            if dt <= tolerance and dt < best_dt:
                best_i, best_dt = i, dt
        if best_i is None:
            false_positives.append(pred)
        else:
            used.add(best_i)
            matches.append((pred, truth[best_i], best_dt))

    missed = [truth[i] for i in range(len(truth)) if i not in used]
    tp = len(matches)
    fp = len(false_positives)
    fn = len(missed)
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision and recall and (precision + recall) else None
    )

    kind_correct = sum(
        1 for p, l, _ in matches if l["kind"] is None or p["kind"] == l["kind"]
    )
    side_correct = sum(
        1 for p, l, _ in matches if l["side"] is None or p["side"] == l["side"]
    )
    confusion = {}
    for p, l, _ in matches:
        key = f"{l['kind'] or '?'}->{p['kind']}"
        confusion[key] = confusion.get(key, 0) + 1

    dts = [dt for _, _, dt in matches]
    return {
        "tolerance_ms": tolerance,
        "ground_truth": len(truth),
        "predicted": len(predictions),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kind_accuracy": round(kind_correct / tp, 4) if tp else None,
        "side_accuracy": round(side_correct / tp, 4) if tp else None,
        "timing_error_ms_mean": round(statistics.mean(dts), 1) if dts else None,
        "timing_error_ms_max": max(dts) if dts else None,
        "confusion": confusion,
    }


def iter_landmarks_jsonl(path, start_ms=0.0, end_ms=float("inf")):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ts = int(rec["t_ms"])
            if ts < start_ms or ts > end_ms:
                continue
            lm = to_landmark_objects(rec.get("lm"))
            wl = to_landmark_objects(rec.get("wl") or [])
            yield ts, lm, wl
