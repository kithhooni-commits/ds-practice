"""evaluate_tcn_v6b.py — v6b(v5b) 모델을 "트리거+분류 모두 TCN" 방식으로 90초 전체 재생 평가.

기존 evaluate_video.py 구조와의 차이
------------------------------------
`evaluate_video.py`의 `TCNMotionClassifier`는 트리거를 담당하지 않는다 — 언제 펀치가 나가는지는
`punch_core.js`/`PunchEvaluator.try_punch`(룰베이스 속도·뻗음 임계값)가 결정하고, TCN은 그 순간
"무슨 종류인가"만 답한다(DEVLOG 19차 설계 원칙).

이 스크립트는 그 원칙을 깨고 **TCN이 매 프레임 스스로 "지금 펀치가 나가는 순간인가"까지 결정**하게
한다 — 룰베이스 트리거를 완전히 우회한다. 매 프레임 60프레임 causal 윈도로 추론해 클래스+확신도를
얻고, "직전 프레임과 다른 펀치 클래스로 전환되는 순간"(edge)에서만, 그리고 확신도가 임계값을 넘고
쿨다운이 지났을 때만 이벤트를 발사한다(DEVLOG에 기록된 "level-trigger 버그" — 같은 라벨을 계속
보고 있으면 쿨다운마다 무한 재발동하는 문제 — 를 피하기 위해 edge 조건을 필수로 둠).

Leakage 방지
------------
`train_tcn_v6b_trigger.py`가 만든 모델은 `TRAIN_GT`(14개)만 보고 학습했고, `TEST_GT`(15개) 및
그 주변 ±2500ms는 학습에서 완전히 제외됐다(purge). 이 스크립트는 **90초 전체를 causal하게
재생**하지만(실전과 동일하게 어디가 train/test인지 모르는 채로), **채점은 TEST_GT 15개만을
정답으로 사용**한다 — TRAIN_GT 시각 근방에서 나온 예측은 (모델이 이미 외운 것이므로) 정답 목록에
없어 자동으로 either 무시되거나 FP로 잡힌다. 이렇게 해야 "얼마나 잘 외웠는가"가 아니라 "학습에
안 쓴 진짜 새 이벤트에 얼마나 일반화하는가"를 정직하게 잰다.

실행:
  python motion_learning/evaluate_tcn_v6b.py
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
from train_tcn_v6b_trigger import (
    CLASSES, SEQ_LEN, DIM, TEST_GT, TRAIN_GT, PURGE_MS,
    R_WR,
)

MODEL_DIR = os.path.join(BASE_DIR, "v6b_tcn_trigger")
BENCHMARK_LANDMARKS = os.path.join(EVAL_DIR, "video", "benchmark_landmarks.jsonl")
RUN_NAME = "v5b_tcn_trigger"
OUT_DIR = os.path.join(EVAL_DIR, "runs", RUN_NAME)

DEVICE = torch.device("cpu")

# 룰베이스와 동일한 계열의 트리거 파라미터 (v6_tcn_overfit_hong.json 재사용 — 비교 가능성 유지)
TCN_MIN_CONF = 0.32
COOLDOWN_MS = 390

PUNCH_CLASSES = {
    "LEFT_JAB": ("L", "STRAIGHT"), "RIGHT_JAB": ("R", "STRAIGHT"),
    "LEFT_HOOK": ("L", "HOOK"), "RIGHT_HOOK": ("R", "HOOK"),
    "LEFT_UPPERCUT": ("L", "UPPERCUT"), "RIGHT_UPPERCUT": ("R", "UPPERCUT"),
}
NON_PUNCH_CLASSES = {"IDLE", "OTHER", "TWO_HAND_GUARD", "ENERGY_WAVE"}


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


@torch.no_grad()
def run_causal_trigger(model, median, scale, clip, t_arr, f_arr):
    """매 프레임 60프레임 causal 윈도로 추론 -> edge+cooldown 트리거.

    반환: predictions(list of dict t_ms/side/kind), raw_trace(디버그용 프레임별 예측·확신도)
    """
    n = len(t_arr)
    predictions = []
    raw_trace = []
    last_state = "IDLE"          # 직전 프레임의 (임계값 통과 여부와 무관한) argmax 클래스
    last_trigger_ms = -10**9
    verified_tcn_used = False

    for i in range(n):
        lo = max(0, i - SEQ_LEN + 1)
        window = f_arr[lo:i + 1]
        if window.shape[0] < SEQ_LEN:
            pad = np.repeat(window[:1], SEQ_LEN - window.shape[0], axis=0)
            window = np.concatenate([pad, window], axis=0)
        scaled = np.clip((window - median) / scale, -clip, clip).astype(np.float32)
        x = torch.tensor(scaled[None])
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        conf = float(conf)
        pred_class = CLASSES[int(idx)]
        verified_tcn_used = True  # 이 줄까지 예외 없이 도달했다는 것 자체가 모델 forward 성공의 증거

        t_ms = int(t_arr[i])
        raw_trace.append({"t_ms": t_ms, "pred_class": pred_class, "conf": round(conf, 4)})

        is_edge = pred_class != last_state
        if (
            pred_class in PUNCH_CLASSES
            and conf >= TCN_MIN_CONF
            and is_edge
            and (t_ms - last_trigger_ms) >= COOLDOWN_MS
        ):
            side, kind = PUNCH_CLASSES[pred_class]
            predictions.append({"t_ms": t_ms, "side": side, "kind": kind, "conf": round(conf, 4)})
            last_trigger_ms = t_ms

        last_state = pred_class

    return predictions, raw_trace, verified_tcn_used


def main():
    print("=" * 70)
    print(f"[검증] TCN이 트리거까지 담당 — {RUN_NAME}")
    print("=" * 70)

    print("[1] 모델/스케일러 로드")
    model, median, scale, clip = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    CausalMotionTCN 파라미터 수: {n_params} (0이면 로드 실패 의심)")

    print("[2] benchmark_landmarks.jsonl 전체 재생 -> 17차원 피처")
    t_arr, f_arr = build_full_sequence()
    print(f"    프레임 수: {len(t_arr)}")

    print("[3] 매 프레임 causal TCN 추론 + edge/cooldown 트리거 (룰베이스 트리거 완전 우회)")
    predictions, raw_trace, verified = run_causal_trigger(model, median, scale, clip, t_arr, f_arr)
    print(f"    verified_tcn_forward_success = {verified}")
    print(f"    전체 90초에서 발사된 예측 이벤트: {len(predictions)}개")

    # --- TCN이 실제로 rule-base와 다른 결정을 내리는지 sanity check ---
    pred_classes_seen = sorted({p["kind"] + "_" + p["side"] for p in predictions})
    conf_values = [p["conf"] for p in predictions]
    print(f"    예측에 등장한 클래스: {pred_classes_seen}")
    print(f"    예측 confidence 분포: min={min(conf_values) if conf_values else None}, "
          f"max={max(conf_values) if conf_values else None}")

    print("\n[4] Leakage-safe 채점 — 정답은 TEST_GT(15개, 학습에서 완전 제외됨)만 사용")
    test_labels = {
        "tolerance_ms": 400,
        "punches": [{"t_ms": t, "side": side, "kind": kind} for t, side, kind in TEST_GT],
    }
    # score_predictions 는 pred dict 에 "kind","side","t_ms" 만 참조하므로 conf 필드는 무시됨
    metrics = score_predictions(predictions, test_labels)

    print(f"    TP/FP/FN = {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
    print(f"    Precision={metrics['precision']}  Recall={metrics['recall']}  F1={metrics['f1']}")
    print(f"    kind_accuracy={metrics['kind_accuracy']}  side_accuracy={metrics['side_accuracy']}")

    os.makedirs(OUT_DIR, exist_ok=True)
    metrics_out = dict(metrics)
    metrics_out["version"] = RUN_NAME
    metrics_out["source_video"] = os.path.abspath(os.path.join(EVAL_DIR, "video", "benchmark.mp4"))
    metrics_out["engine"] = "tcn_full_trigger"
    metrics_out["description"] = (
        "v5b: TCN이 '펀치 종류'뿐 아니라 '펀치가 언제 나가는지(트리거)'까지 매 프레임 causal "
        "추론으로 직접 결정. 룰베이스 punch_core.try_punch 는 전혀 관여하지 않음. "
        f"TCN_MIN_CONF={TCN_MIN_CONF}, cooldown={COOLDOWN_MS}ms, edge-trigger(직전 프레임과 "
        "클래스가 달라질 때만 발사)."
    )
    metrics_out["verified_tcn_forward_success"] = verified
    metrics_out["model_param_count"] = n_params
    metrics_out["all_predictions_full_90s"] = predictions
    metrics_out["note_on_scope"] = (
        f"predictions 는 90초 전체에서 나온 것이지만(TRAIN_GT {len(TRAIN_GT)}개 구간 포함), "
        f"채점 정답은 TEST_GT {len(TEST_GT)}개만 사용 — TRAIN_GT 근방 예측은 정답이 없으므로 "
        "매칭되면 안 되고(=TRAIN 시각에 GT 자체가 없음), 매칭 안 되면 FP 로 집계된다."
    )
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "raw_trace.jsonl"), "w", encoding="utf-8") as f:
        for row in raw_trace:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[OK] 저장 완료: {OUT_DIR}/metrics.json, raw_trace.jsonl")


if __name__ == "__main__":
    main()
