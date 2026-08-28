"""evaluate_tcn_v5c.py — 하이브리드 트리거: "TCN 확신도" AND "룰베이스 물리 조건".

v5b(`evaluate_tcn_v6b.py`)의 문제: TCN 확신도 + edge + cooldown만으로 트리거를 걸었더니
90초 동안 60번을 쏴서 Precision이 0.20까지 무너졌다(F1 0.32, v5의 0.379보다 낮음). 프레임
단위로 클래스 예측이 흔들리는데(flicker) 그걸 막아줄 물리적 게이트가 없었기 때문이다
(`v5b_tcn_trigger/EXPERIMENT_REPORT.md` §3 참고).

이번 버전은 매 프레임 다음 두 조건을 **AND**로 결합해야만 발사한다:
  1) TCN이 어떤 펀치 클래스를 confidence >= TCN_MIN_CONF 로 예측 (edge + cooldown 유지)
  2) **그 순간(현재 프레임)의 실제 운동학 값이 룰베이스 물리 조건을 만족**한다 —
     해당 side의 손목 속도가 PUNCH_ARM_GATE 이상이고, reach_n(뻗음)이 PUNCH_EXTEND_GATE 이상.
     (이 두 상수는 `eval/evaluate_video.py`/`punch_core.js`가 "펀치 창을 여는(arm)" 데 쓰는
     것과 같은 계열의 하한선이다 — 트리거 전체를 룰베이스 상태기계로 대체하는 게 아니라,
     "몸이 실제로 그 방향으로 움직이고 있는가"라는 최소한의 물리적 증거만 요구한다.)

즉 TCN 혼자서는 못 열고, 룰베이스 혼자서도 못 연다 — 두 신호가 동시에 있어야 연다.

Leakage 방지 방식은 v5b와 동일: `train_tcn_v6b_trigger.py`가 만든, TEST_GT 15개를 학습에서
완전히 제외한 모델(`motion_learning/v6b_tcn_trigger/`)을 그대로 재사용한다(재학습 불필요 —
물리 게이트는 추론 후처리이지 모델 자체를 바꾸는 게 아니다).

실행:
  python motion_learning/evaluate_tcn_v5c.py
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

from scoring import iter_landmarks_jsonl, score_predictions
from tcn_model import CausalMotionTCN
from train_tcn_benchmark_overfit import extract_feat17
from train_tcn_v6b_trigger import CLASSES, SEQ_LEN, DIM, TEST_GT, TRAIN_GT, R_WR

MODEL_DIR = os.path.join(BASE_DIR, "v6b_tcn_trigger")  # v5b와 동일 모델 재사용
BENCHMARK_LANDMARKS = os.path.join(EVAL_DIR, "video", "benchmark_landmarks.jsonl")
RUN_NAME = "v5c_tcn_hybrid_gate"
OUT_DIR = os.path.join(EVAL_DIR, "runs", RUN_NAME)

DEVICE = torch.device("cpu")

TCN_MIN_CONF = 0.32
COOLDOWN_MS = 390

# 17차원 피처 중 물리 게이트에 쓰는 인덱스 (extract_feat17/scaler feature_names 순서와 동일)
IDX_LEFT_REACH, IDX_RIGHT_REACH = 2, 3
IDX_LEFT_SPEED, IDX_RIGHT_SPEED = 10, 11

# 룰베이스(eval/evaluate_video.py)의 "펀치 창을 여는" 하한선과 같은 계열의 값.
# PUNCH_SPEED(1.65)/PUNCH_REACH_N(0.88)는 "피크가 이 값을 넘어야 확정"이라는 상한 쪽 기준이라
# 이 순간(edge 프레임)에 그대로 요구하면 TCN이 edge를 잡는 시점과 물리적 피크 시점이 어긋날 때
# 정당한 펀치까지 게이트에서 막힌다. 그래서 "arm(창을 여는)" 단계의 하한 값을 게이트로 쓴다.
PUNCH_ARM_GATE = float(os.environ.get("V5C_SPEED_GATE", 1.0))
PUNCH_EXTEND_GATE = float(os.environ.get("V5C_REACH_GATE", 0.40))

PUNCH_CLASSES = {
    "LEFT_JAB": ("L", "STRAIGHT"), "RIGHT_JAB": ("R", "STRAIGHT"),
    "LEFT_HOOK": ("L", "HOOK"), "RIGHT_HOOK": ("R", "HOOK"),
    "LEFT_UPPERCUT": ("L", "UPPERCUT"), "RIGHT_UPPERCUT": ("R", "UPPERCUT"),
}


def load_model():
    with open(os.path.join(MODEL_DIR, "boxing_tcn_scaler.json"), encoding="utf-8") as f:
        scaler = json.load(f)
    median = np.array(scaler["median"], dtype=np.float32)
    scale = np.array(scaler["scale"], dtype=np.float32)
    clip = scaler.get("clip", 8.0)
    model = CausalMotionTCN(input_dim=DIM, num_classes=len(CLASSES))
    state = torch.load(os.path.join(MODEL_DIR, "boxing_tcn.pth"), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model, median, scale, clip


def build_full_sequence():
    prev_l = prev_r = None
    prev_t = None
    t_list, f_list = [], []
    for t_ms, lm, _wl in iter_landmarks_jsonl(BENCHMARK_LANDMARKS):
        if lm is None or len(lm) <= R_WR:
            continue
        feat, prev_l, prev_r = extract_feat17(lm, prev_l, prev_r, prev_t, t_ms)
        prev_t = t_ms
        t_list.append(t_ms)
        f_list.append(feat)
    return np.asarray(t_list, dtype=np.int64), np.asarray(f_list, dtype=np.float32)


def physics_gate_ok(side, raw_frame):
    """raw_frame: 스케일링 전 17차원 원본 피처 (현재 프레임, 실제 물리 단위)."""
    if side == "L":
        return raw_frame[IDX_LEFT_SPEED] > PUNCH_ARM_GATE and raw_frame[IDX_LEFT_REACH] > PUNCH_EXTEND_GATE
    return raw_frame[IDX_RIGHT_SPEED] > PUNCH_ARM_GATE and raw_frame[IDX_RIGHT_REACH] > PUNCH_EXTEND_GATE


@torch.no_grad()
def run_hybrid_trigger(model, median, scale, clip, t_arr, f_arr):
    n = len(t_arr)
    predictions = []
    raw_trace = []
    last_state = "IDLE"
    last_trigger_ms = -10**9
    n_conf_pass_but_gate_blocked = 0

    for i in range(n):
        lo = max(0, i - SEQ_LEN + 1)
        window = f_arr[lo:i + 1]
        if window.shape[0] < SEQ_LEN:
            pad = np.repeat(window[:1], SEQ_LEN - window.shape[0], axis=0)
            window = np.concatenate([pad, window], axis=0)
        scaled = np.clip((window - median) / scale, -clip, clip).astype(np.float32)
        x = torch.tensor(scaled[None])
        probs = torch.softmax(model(x), dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        conf = float(conf)
        pred_class = CLASSES[int(idx)]

        t_ms = int(t_arr[i])
        current_raw = f_arr[i]  # 스케일링 전 원본 (물리 게이트는 원본 단위로 비교)

        is_edge = pred_class != last_state
        conf_ok = pred_class in PUNCH_CLASSES and conf >= TCN_MIN_CONF
        gate_ok = False
        if conf_ok:
            side = PUNCH_CLASSES[pred_class][0]
            gate_ok = physics_gate_ok(side, current_raw)
            if not gate_ok:
                n_conf_pass_but_gate_blocked += 1

        raw_trace.append({
            "t_ms": t_ms, "pred_class": pred_class, "conf": round(conf, 4),
            "conf_ok": bool(conf_ok), "gate_ok": bool(gate_ok) if conf_ok else None,
        })

        if conf_ok and gate_ok and is_edge and (t_ms - last_trigger_ms) >= COOLDOWN_MS:
            side, kind = PUNCH_CLASSES[pred_class]
            predictions.append({"t_ms": t_ms, "side": side, "kind": kind, "conf": round(conf, 4)})
            last_trigger_ms = t_ms

        last_state = pred_class

    return predictions, raw_trace, n_conf_pass_but_gate_blocked


def score_against(predictions, gt_triples, label="TEST_GT"):
    labels = {
        "tolerance_ms": 400,
        "punches": [{"t_ms": t, "side": s, "kind": k} for t, s, k in gt_triples],
    }
    m = score_predictions(predictions, labels)
    print(f"    [{label}] TP/FP/FN={m['tp']}/{m['fp']}/{m['fn']}  "
          f"P={m['precision']} R={m['recall']} F1={m['f1']}  "
          f"kind_acc={m['kind_accuracy']} side_acc={m['side_accuracy']}")
    return m


def main():
    print("=" * 70)
    print(f"[v5c] 하이브리드 트리거 (TCN 확신도 AND 룰베이스 물리조건) — {RUN_NAME}")
    print("=" * 70)

    print("[1] 모델 로드 (v6b_tcn_trigger — v5b와 동일, leakage 방지 학습됨)")
    model, median, scale, clip = load_model()

    print("[2] 피처 시퀀스 재생")
    t_arr, f_arr = build_full_sequence()
    print(f"    프레임 수: {len(t_arr)}")

    print(f"[3] 하이브리드 트리거 (TCN_MIN_CONF={TCN_MIN_CONF}, "
          f"PUNCH_ARM_GATE={PUNCH_ARM_GATE}, PUNCH_EXTEND_GATE={PUNCH_EXTEND_GATE}, "
          f"cooldown={COOLDOWN_MS}ms)")
    predictions, raw_trace, n_blocked = run_hybrid_trigger(model, median, scale, clip, t_arr, f_arr)
    print(f"    90초 전체 예측 이벤트: {len(predictions)}개")
    print(f"    TCN 확신도는 통과했지만 물리 게이트에서 걸러진 프레임 수: {n_blocked}")

    print("\n[4] 채점")
    test_metrics = score_against(predictions, TEST_GT, "TEST_GT (leakage 없음, 핵심 지표)")
    full_gt = json.loads(open(os.path.join(EVAL_DIR, "video", "benchmark_labels.json"), encoding="utf-8").read())
    full_triples = [(p["t_ms"], p["side"], p["kind"]) for p in full_gt["punches"]]
    full_metrics = score_against(predictions, full_triples, "전체 29개 GT (참고용, TRAIN 부분 leaky)")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = dict(test_metrics)
    out["version"] = RUN_NAME
    out["engine"] = "tcn_hybrid_gate_trigger"
    out["source_video"] = os.path.abspath(os.path.join(EVAL_DIR, "video", "benchmark.mp4"))
    out["description"] = (
        "v5c: 트리거를 'TCN confidence>=TCN_MIN_CONF' AND '해당 side의 순간 속도>PUNCH_ARM_GATE "
        "및 뻗음>PUNCH_EXTEND_GATE' 의 AND 결합으로 결정. edge(직전 프레임과 클래스 전환)+"
        f"cooldown({COOLDOWN_MS}ms) 유지. 물리 게이트에서 걸러진 프레임 수={n_blocked}."
    )
    out["tune"] = {
        "TCN_MIN_CONF": TCN_MIN_CONF, "PUNCH_ARM_GATE": PUNCH_ARM_GATE,
        "PUNCH_EXTEND_GATE": PUNCH_EXTEND_GATE, "COOLDOWN_MS": COOLDOWN_MS,
    }
    out["n_gate_blocked_frames"] = n_blocked
    out["all_predictions_full_90s"] = predictions
    out["full_90s_leaky_metrics"] = full_metrics
    out["full_90s_leaky_metrics_WARNING"] = (
        "TRAIN_GT 14개가 정답에 섞여 있어 낙관적으로 부풀려짐 — 헤드라인 지표로 쓰지 말 것. "
        "핵심 지표는 위쪽 TEST_GT 전용 tp/fp/fn/precision/recall/f1."
    )
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "raw_trace.jsonl"), "w", encoding="utf-8") as f:
        for row in raw_trace:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[OK] 저장 완료: {OUT_DIR}/metrics.json, raw_trace.jsonl")


if __name__ == "__main__":
    main()
