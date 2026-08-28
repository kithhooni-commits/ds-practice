"""evaluate_kind_classification.py — 타이밍(트리거)을 완전히 배제하고,
"이 펀치가 잽/훅/어퍼컷 중 뭔지"만 순수하게 평가한다.

이전 평가(run_pipeline.py)의 kind_accuracy는 "정답 시각 근처에서 rule-base 트리거가 실제로
발사됐는지"까지 섞여 있어(TP로 매칭된 11개만 채점), 타이밍 노이즈가 종류 판정 성능을 가린다.
여기서는 **29개 정답 펀치 전부**에 대해 "어느 팔인지는 이미 안다"는 전제로, 그 팔의 3가지
종류(STRAIGHT/HOOK/UPPERCUT) 중 뭔지만 판정하게 해서 confusion matrix와 F1을 낸다.

비교하는 3개 분류기:
  1) rule_at_gt   — classify_punch()를 정답 시각 ±250ms 구간의 "최고 속도 프레임"에 적용
                    (실제 punch_core가 창 안에서 peak를 추적하는 방식과 동일한 원리)
  2) tcn_deployed — 기존 4인 학습 배포 모델(motion_learning/boxing_tcn.pth), 정답 시각에
                    끝나는 60프레임 causal 윈도로 추론
  3) tcn_overfit  — 이번 실험에서 benchmark.mp4 자체로 overfitting한 모델
                    (motion_learning/overfit_hong/boxing_tcn.pth)

TCN 쪽은 "이미 어느 팔인지 안다"는 전제를 반영해, 10개 클래스 중 해당 팔(LEFT_/RIGHT_)의
JAB·HOOK·UPPERCUT 3개로만 softmax를 제한하고 그 중 argmax를 쓴다 (신뢰도 임계값 게이팅 없음 —
여기서는 트리거 시스템과의 결합이 아니라 분류기 자체의 순수 성능을 보고 싶은 것이므로).
"""
import os
import sys
import json
import math
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "..", "eval")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, EVAL_DIR)

from scoring import iter_landmarks_jsonl
from tcn_model import CausalMotionTCN
from train_tcn_benchmark_overfit import build_full_sequence, window_ending_at, CLASSES, DIM

BENCHMARK_LANDMARKS = os.path.join(EVAL_DIR, "video", "benchmark_landmarks.jsonl")
BENCHMARK_LABELS = os.path.join(EVAL_DIR, "video", "benchmark_labels.json")

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16
KIND_CLASSES = ["STRAIGHT", "HOOK", "UPPERCUT"]

# rule-base thresholds, synced with punch_core.js / evaluate_video.py
UPPERCUT_VY = 0.55
UPPERCUT_ELBOW = 150
HOOK_VX = 0.56
HOOK_ELBOW = 158

PEAK_SEARCH_WINDOW_MS = 250  # GT 시각 주변에서 "최고 속도 프레임"을 찾는 탐색 반경


def dist3(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])


def angle_deg(a, b, c):
    abx, aby, abz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    cbx, cby, cbz = c[0] - b[0], c[1] - b[1], c[2] - b[2]
    d = math.hypot(abx, aby, abz) * math.hypot(cbx, cby, cbz)
    if d < 1e-6:
        return 180.0
    cosv = max(-1.0, min(1.0, (abx * cbx + aby * cby + abz * cbz) / d))
    return math.degrees(math.acos(cosv))


def classify_punch_rule(speed, vx, vy, elbow):
    s = max(speed, 1e-3)
    up_ratio = -vy / s
    hook_ratio = abs(vx) / s
    if up_ratio > UPPERCUT_VY and elbow < UPPERCUT_ELBOW:
        return "UPPERCUT"
    elif hook_ratio > HOOK_VX and elbow < HOOK_ELBOW:
        return "HOOK"
    return "STRAIGHT"


def load_world_landmark_frames():
    """wl(월드 랜드마크, 미터 단위)을 프레임 순서대로 로드. rule-base 피크 탐색용."""
    frames = []
    for t_ms, _lm, wl in iter_landmarks_jsonl(BENCHMARK_LANDMARKS):
        if wl is None or len(wl) <= R_WR:
            frames.append((t_ms, None))
            continue
        frames.append((t_ms, [(p.x, p.y, p.z) for p in wl]))
    return frames


def rule_predict(wl_frames, side, t_ms):
    """t_ms ± PEAK_SEARCH_WINDOW_MS 구간에서 해당 팔 손목 속도가 최고인 프레임을 찾아
    그 순간의 속도벡터·팔꿈치 각도로 classify_punch_rule() 을 적용 (punch_core의 peak-tracking과 동일 원리)."""
    sh_id, el_id, wr_id = (L_SH, L_EL, L_WR) if side == "L" else (R_SH, R_EL, R_WR)
    lo, hi = t_ms - PEAK_SEARCH_WINDOW_MS, t_ms + PEAK_SEARCH_WINDOW_MS
    best = None  # (speed, vx, vy, elbow)
    prev = None
    for ts, wl in wl_frames:
        if wl is None:
            prev = None
            continue
        wr = wl[wr_id]
        if prev is not None and lo <= ts <= hi:
            p_ts, p_wr = prev
            dt = (ts - p_ts) / 1000.0
            if 0.008 < dt < 0.4:
                vx = (wr[0] - p_wr[0]) / dt
                vy = (wr[1] - p_wr[1]) / dt
                vz = (wr[2] - p_wr[2]) / dt
                speed = math.hypot(vx, vy, vz)
                if best is None or speed > best[0]:
                    elbow = angle_deg(wl[sh_id], wl[el_id], wr)
                    best = (speed, vx, vy, elbow)
        prev = (ts, wr)
    if best is None:
        return "STRAIGHT"  # 속도 신호 없음 -> 기본값(어차피 분류기 성능 비교가 목적)
    speed, vx, vy, elbow = best
    return classify_punch_rule(speed, vx, vy, elbow)


def load_tcn(model_dir):
    scaler = json.loads(open(os.path.join(model_dir, "boxing_tcn_scaler.json"), encoding="utf-8").read())
    model = CausalMotionTCN(input_dim=DIM, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(os.path.join(model_dir, "boxing_tcn.pth"), map_location="cpu"))
    model.eval()
    return model, scaler


def tcn_predict(model, scaler, window, side):
    median = np.array(scaler["median"], dtype=np.float32)
    scale = np.array(scaler["scale"], dtype=np.float32)
    clip = scaler.get("clip", 8.0)
    x = np.clip((window - median) / scale, -clip, clip)
    with torch.no_grad():
        logits = model(torch.tensor(x[None, :, :], dtype=torch.float32))[0]
        probs = torch.softmax(logits, dim=-1).numpy()

    prefix = "LEFT_" if side == "L" else "RIGHT_"
    # 이미 side는 GT로 알고 있다는 전제 -> 해당 팔의 JAB/HOOK/UPPERCUT 3개로만 argmax
    option_map = {"JAB": "STRAIGHT", "HOOK": "HOOK", "UPPERCUT": "UPPERCUT"}
    best_kind, best_p = "STRAIGHT", -1.0
    for suffix, kind in option_map.items():
        idx = CLASSES.index(prefix + suffix)
        if probs[idx] > best_p:
            best_p, best_kind = probs[idx], kind
    return best_kind, float(best_p)


def confusion_and_f1(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    per_class = {}
    for c in classes:
        yt = [1 if y == c else 0 for y in y_true]
        yp = [1 if y == c else 0 for y in y_pred]
        per_class[c] = {
            "precision": round(precision_score(yt, yp, zero_division=0), 4),
            "recall": round(recall_score(yt, yp, zero_division=0), 4),
            "f1": round(f1_score(yt, yp, zero_division=0), 4),
            "support": sum(yt),
        }
    macro_f1 = round(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0), 4)
    weighted_f1 = round(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0), 4)
    acc = round(sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true), 4)
    return {
        "confusion_matrix": cm.tolist(),
        "labels_order": classes,
        "per_class": per_class,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def print_confusion(name, result):
    print(f"\n=== {name} ===")
    print(f"accuracy={result['accuracy']}  macro_f1={result['macro_f1']}  weighted_f1={result['weighted_f1']}")
    classes = result["labels_order"]
    header = "true\\pred".ljust(10) + "".join(c.ljust(11) for c in classes)
    print(header)
    for i, c in enumerate(classes):
        row = c.ljust(10) + "".join(str(result["confusion_matrix"][i][j]).ljust(11) for j in range(len(classes)))
        print(row)
    for c, m in result["per_class"].items():
        print(f"  {c:10s} precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} (n={m['support']})")


def main():
    labels = json.loads(open(BENCHMARK_LABELS, encoding="utf-8").read())
    punches = labels["punches"]
    y_true = [p["kind"] for p in punches]

    print(f"[1] benchmark_landmarks.jsonl 재생 (17차원 피처 + 월드 랜드마크)")
    t_arr, f_arr = build_full_sequence()
    wl_frames = load_world_landmark_frames()

    print(f"[2] rule-base (classify_punch, GT시각 ±{PEAK_SEARCH_WINDOW_MS}ms 피크 탐색) 예측")
    y_pred_rule = [rule_predict(wl_frames, p["side"], p["t_ms"]) for p in punches]

    print(f"[3] TCN (배포 모델, 4인 학습) 예측")
    model_dep, scaler_dep = load_tcn(os.path.join(BASE_DIR))
    y_pred_dep, conf_dep = [], []
    for p in punches:
        w = window_ending_at(t_arr, f_arr, p["t_ms"])
        kind, conf = tcn_predict(model_dep, scaler_dep, w, p["side"])
        y_pred_dep.append(kind)
        conf_dep.append(conf)

    print(f"[4] TCN (overfit, hong/benchmark.mp4 전용) 예측")
    model_of, scaler_of = load_tcn(os.path.join(BASE_DIR, "overfit_hong"))
    y_pred_of, conf_of = [], []
    for p in punches:
        w = window_ending_at(t_arr, f_arr, p["t_ms"])
        kind, conf = tcn_predict(model_of, scaler_of, w, p["side"])
        y_pred_of.append(kind)
        conf_of.append(conf)

    results = {
        "n_samples": len(punches),
        "note": "타이밍/트리거 배제: 29개 GT 펀치 전부에 대해, side는 GT로 이미 알고 있다는 "
                "전제로 STRAIGHT/HOOK/UPPERCUT 3지 분류만 평가.",
        "rule_at_gt": confusion_and_f1(y_true, y_pred_rule, KIND_CLASSES),
        "tcn_deployed": confusion_and_f1(y_true, y_pred_dep, KIND_CLASSES),
        "tcn_overfit_hong": confusion_and_f1(y_true, y_pred_of, KIND_CLASSES),
    }

    print_confusion("rule_at_gt (classify_punch, 피크 탐색)", results["rule_at_gt"])
    print_confusion("tcn_deployed (4인 학습, 배포 모델)", results["tcn_deployed"])
    print_confusion("tcn_overfit_hong (benchmark.mp4 자체로 overfitting)", results["tcn_overfit_hong"])

    print("\n[세부 예측 — 틀린 것 위주로 대조]")
    print(f"{'t_ms':>8} {'side':>4} {'GT':>9} {'rule':>9} {'tcn_dep':>9}(conf) {'tcn_of':>9}(conf)")
    for i, p in enumerate(punches):
        print(f"{p['t_ms']:>8} {p['side']:>4} {y_true[i]:>9} {y_pred_rule[i]:>9} "
              f"{y_pred_dep[i]:>9}({conf_dep[i]:.2f}) {y_pred_of[i]:>9}({conf_of[i]:.2f})")

    out_path = os.path.join(BASE_DIR, "kind_classification_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 저장: {out_path}")


if __name__ == "__main__":
    main()
