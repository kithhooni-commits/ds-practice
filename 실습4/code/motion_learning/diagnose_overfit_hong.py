"""diagnose_overfit_hong.py — v6 overfit 모델이 실제 평가 중 왜 한 번도 rule-base를 덮어쓰지
못했는지 진단한다. 각 실제 트리거 시각(evaluate_video.py가 발사한 t_ms)에서 모델이 무엇을
예측했는지, 확신도가 얼마였는지, TCN_MIN_CONF(0.32) 임계·side 필터를 통과했는지를 출력한다.
"""
import os
import sys
import json
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "..", "eval")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, EVAL_DIR)

from train_tcn_benchmark_overfit import (
    build_full_sequence, window_ending_at, CLASSES, SEQ_LEN, DIM,
)
from tcn_model import CausalMotionTCN

OUT_DIR = os.path.join(BASE_DIR, "overfit_hong")
TCN_MIN_CONF = 0.32

PUNCH_CLASSES = {
    "LEFT_JAB", "RIGHT_JAB", "LEFT_HOOK", "RIGHT_HOOK", "LEFT_UPPERCUT", "RIGHT_UPPERCUT",
}


def load_model():
    scaler = json.loads(open(os.path.join(OUT_DIR, "boxing_tcn_scaler.json"), encoding="utf-8").read())
    model = CausalMotionTCN(input_dim=DIM, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "boxing_tcn.pth"), map_location="cpu"))
    model.eval()
    return model, scaler


def predict_window(model, scaler, window):
    median = np.array(scaler["median"], dtype=np.float32)
    scale = np.array(scaler["scale"], dtype=np.float32)
    clip = scaler.get("clip", 8.0)
    x = np.clip((window - median) / scale, -clip, clip)
    with torch.no_grad():
        logits = model(torch.tensor(x[None, :, :], dtype=torch.float32))[0]
        probs = torch.softmax(logits, dim=-1).numpy()
    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx])


def main():
    labels = json.loads(open(os.path.join(EVAL_DIR, "video", "benchmark_labels.json"), encoding="utf-8").read())
    punches_csv = os.path.join(EVAL_DIR, "runs", "v6_tcn_overfit_hong", "punches.csv")

    import csv
    rows = []
    with open(punches_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    model, scaler = load_model()
    t_arr, f_arr = build_full_sequence()

    print(f"{'trigger_t_ms':>12} {'side':>4} {'rule_kind':>9} | {'gt_nearest':>10} {'gt_dt_ms':>8} | "
          f"{'model_pred':>14} {'conf':>6} {'side_ok':>7} {'>=thr':>5}")
    n_would_override = 0
    for r in rows:
        t_ms = int(float(r["t_ms"]))
        side = r["side"]
        w = window_ending_at(t_arr, f_arr, t_ms)
        pred_label, conf = predict_window(model, scaler, w)
        want_side = "LEFT_" if side == "L" else "RIGHT_"
        side_ok = pred_label in PUNCH_CLASSES and pred_label.startswith(want_side)
        passes = side_ok and conf >= TCN_MIN_CONF
        if passes:
            n_would_override += 1

        # nearest GT punch on this side, for reference
        cand = [p for p in labels["punches"] if p["side"] == side]
        nearest = min(cand, key=lambda p: abs(p["t_ms"] - t_ms)) if cand else None
        gt_dt = (t_ms - nearest["t_ms"]) if nearest else None

        print(f"{t_ms:>12} {side:>4} {r['kind']:>9} | "
              f"{(nearest['kind'] if nearest else '-'):>10} {str(gt_dt):>8} | "
              f"{pred_label:>14} {conf:>6.3f} {str(side_ok):>7} {str(conf>=TCN_MIN_CONF):>5}")

    print(f"\n모델이 rule-base를 덮어쓸 수 있었던 트리거: {n_would_override}/{len(rows)}")


if __name__ == "__main__":
    main()
