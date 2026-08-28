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
    """예측과 정답 라벨을 dt 최소 우선 전역 매칭으로 채점한다.

    이전 그리디 구현은 예측을 시간순으로 훑으며 각 예측이 남은 라벨 중 최근접을
    선점했다. 그래서 콤보(예: t=76500 / 76933) 처럼 tolerance(400ms) 이내에
    다수 라벨이 몰릴 때, 이른 예측이 자기 것이 아닌 뒤쪽 라벨을 물어가면 이후
    올바른 예측이 FP 로 뒤집혔다.

    개선판은 (pred_i, label_j, dt) 후보 전체를 dt 오름차순으로 정렬한 뒤
    양쪽이 모두 미사용일 때만 매칭한다. 이는 tolerance 창 안에서 dt 총합을
    최소화하는 근사 최적 매칭이며, 콤보 구간의 뒤바뀜 문제를 제거한다.
    """
    # 이 예측 인덱스가 정답 인덱스 pi 에 매칭됐음을 나중에 phase 통계가
    # 뒤짚어 볼 수 있도록 pred_index 를 유지한다 (T P 인 예측을 non_action_fp
    # 에서 제외할 때 필요).
    tolerance = labels["tolerance_ms"]
    truth = labels["punches"]

    # 예측을 시간순으로 세우되 원래 순서 인덱스를 함께 기억한다.
    indexed_preds = sorted(
        enumerate(predictions), key=lambda kv: kv[1]["t_ms"]
    )

    candidates = []
    for pi, pred in indexed_preds:
        for li, lab in enumerate(truth):
            dt = abs(pred["t_ms"] - lab["t_ms"])
            if dt <= tolerance:
                # tie-break: dt 동률일 때 kind/side 일치를 우선한다.
                # 이래야 콤보 구간에서 같은 라벨을 두 예측이 dt=200ms 로 다툴 때
                # 종류·팔이 맞는 예측이 매칭되어 kind_accuracy 가 부풀지 않는다.
                kind_match = 0 if (lab["kind"] is None or pred["kind"] == lab["kind"]) else 1
                side_match = 0 if (lab["side"] is None or pred["side"] == lab["side"]) else 1
                candidates.append((dt, kind_match, side_match, pi, li))
    # 정렬 키: (dt asc, kind_match asc(=0 우선), side_match asc, pi, li).
    # 0 이 "일치" 이므로 오름차순이면 일치 우선.
    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

    used_pred = set()
    used_label = set()
    matches = []  # (pred_dict, label_dict, dt, pred_index)
    for dt, _km, _sm, pi, li in candidates:
        if pi in used_pred or li in used_label:
            continue
        used_pred.add(pi)
        used_label.add(li)
        matches.append((predictions[pi], truth[li], dt, pi))

    false_positives = [
        predictions[pi] for pi, _ in indexed_preds if pi not in used_pred
    ]
    missed = [truth[i] for i in range(len(truth)) if i not in used_label]
    tp = len(matches)
    fp = len(false_positives)
    fn = len(missed)
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None
    # F1: precision 또는 recall 이 정확히 0.0 이더라도 계산 가능한 상황이면
    # 관례상 0.0 을 반환한다. 예전 코드는 `if precision and recall` 이라 값이
    # 0.0 일 때 F1=None 이 나왔고, registry/대시보드 소비자가 None 을 float
    # 연산에 넣으면 크래시하는 잠재 문제가 있었다.
    if precision is None or recall is None:
        f1 = None
    elif (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = round(2 * precision * recall / (precision + recall), 4)

    kind_correct = sum(
        1 for p, l, _, _ in matches if l["kind"] is None or p["kind"] == l["kind"]
    )
    side_correct = sum(
        1 for p, l, _, _ in matches if l["side"] is None or p["side"] == l["side"]
    )
    confusion = {}
    for p, l, _, _ in matches:
        key = f"{l['kind'] or '?'}->{p['kind']}"
        confusion[key] = confusion.get(key, 0) + 1

    dts = [dt for _, _, dt, _ in matches]
    # phase 통계에서 TP 예측을 non_action_fp 에서 제외할 수 있도록 매칭된
    # 예측 인덱스와 pred_t_ms 를 함께 노출한다. run_pipeline.py 가 이걸
    # 소비하며, 이 필드가 없으면 예전처럼 "정상 검출"이 오검출로 잡히는
    # 이중 페널티가 다시 발생한다.
    matched_pred_info = [
        {"pred_index": pi, "pred_t_ms": p["t_ms"], "label_t_ms": l["t_ms"], "dt_ms": dt}
        for p, l, dt, pi in matches
    ]

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
        "matches": matched_pred_info,
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
