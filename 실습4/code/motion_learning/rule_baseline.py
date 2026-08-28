"""
rule_baseline.py — 현재 punch_core.js 의 rule-base 판정 로직을 10-클래스 전체 라벨 체계로
그대로 이식한 Python 버전. TCN과 "같은 실측 데이터, 같은 LOSO 분할"로 비교하기 위한 기준선이다.

punch_core.js 에서 그대로 가져온 부분:
  - classify(k): -vy/speed > UPPERCUT_VY && elbow 충분히 굽음  → UPPERCUT
                  |vx|/speed > HOOK_VX && elbow 충분히 굽음     → HOOK
                  그 외                                          → STRAIGHT(JAB)
  (punch_core.js 는 "어느 팔이 펀치를 냈는지"는 애초에 알고 시작하지만,
   여기서는 그 정보가 없으므로 최고 속도가 나온 팔을 active arm으로 판정한다)

punch_core.js 에 없어서 새로 정의한 부분 (GUARD/IDLE/ENERGY_WAVE):
  - GUARD:  양 손목이 서로 가깝고 팔꿈치도 모임 (fighter_client.html 의 거리 기반 가드 판정과 같은 발상)
  - IDLE:   전체 구간 손목 속도가 낮음
  - ENERGY_WAVE: 양손이 "동시에" 강하게 뻗음 (현재 게임엔 아예 구현이 없는 동작이라 새로 정의)
  - OTHER 를 위한 명시적 규칙은 없다 — 지금의 rule-base 시스템도 "정체불명 동작"을 걸러내는
    개념이 없기 때문에(항상 무언가로 분류), 이 한계를 그대로 재현한다.

임계값은 heuristic_7j_v1 값 분포를 한 번 훑어 사람이 정한 것으로, 실제 PUNCH_TUNE이
하니스 실측으로 튜닝된 것과 같은 성격의 "고정 규칙"이다 (fold별로 다시 맞추지 않는다).
"""
import numpy as np

HAND_DIST_GUARD = 0.85
ELBOW_DIST_GUARD = 1.55
IDLE_SPEED_MAX = 11.0
WAVE_MIN_SPEED = 18.0
WAVE_BALANCE_RATIO = 0.80
UPPERCUT_VY = 0.55
UPPERCUT_ELBOW_RATIO = 0.85
HOOK_VX = 0.56
HOOK_ELBOW_RATIO = 0.90

# heuristic_7j_v1 열 인덱스
L_ELBOW, R_ELBOW = 0, 1
L_VX, L_VY, L_VZ, R_VX, R_VY, R_VZ = 4, 5, 6, 7, 8, 9
L_SPEED, R_SPEED = 10, 11
HANDS_DIST, ELBOW_DIST = 12, 15

# 데이터로 최적화하는 별도 휴리스틱의 초기값이다. 위의 고정 임계값과
# classify_heuristic_sequence()는 기존 rule baseline 재현을 위해 그대로 둔다.
DEFAULT_OPTIMIZED_THRESHOLDS = {
    "lookback_frames": 8,
    "hand_dist_guard": 0.85,
    "elbow_dist_guard": 1.55,
    "idle_speed_max": 11.0,
    "wave_min_speed": 18.0,
    "wave_balance_ratio": 0.80,
    "other_balance_min": 0.88,
    "uppercut_vy": 0.35,
    "uppercut_elbow_ratio": 0.85,
    "hook_vx": 0.22,
    "hook_elbow_ratio": 0.90,
    "jab_vz_min": 0.80,
}


def mirror_heuristic_sequence(heur_seq):
    """좌우 반전 증강. 17차원 heuristic_7j_v1의 좌우 열과 vx 부호를 보정한다."""
    seq = np.asarray(heur_seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 17:
        raise ValueError(f"heuristic sequence must have shape (T, 17), got {seq.shape}")
    out = seq.copy()
    out[:, [0, 1]] = seq[:, [1, 0]]
    out[:, [2, 3]] = seq[:, [3, 2]]
    out[:, [4, 5, 6]] = seq[:, [7, 8, 9]] * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    out[:, [7, 8, 9]] = seq[:, [4, 5, 6]] * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    out[:, [10, 11]] = seq[:, [11, 10]]
    out[:, [13, 14]] = seq[:, [14, 13]]
    return out


def summarize_heuristic_sequence(heur_seq, lookback_frames=8):
    """Python/브라우저 공용 최적화 규칙에 필요한 시퀀스 통계량."""
    seq = np.asarray(heur_seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[0] == 0 or seq.shape[1] != 17:
        raise ValueError(f"heuristic sequence must have shape (T, 17), got {seq.shape}")

    lookback = max(1, min(int(lookback_frames), len(seq)))
    max_l = float(seq[:, L_SPEED].max())
    max_r = float(seq[:, R_SPEED].max())
    overall_max = max(max_l, max_r)
    min_max = min(max_l, max_r)
    side = "L" if max_l >= max_r else "R"
    active_speed = max_l if side == "L" else max_r
    opposite_speed = max_r if side == "L" else max_l
    speed_col = L_SPEED if side == "L" else R_SPEED
    peak_idx = int(np.argmax(seq[:, speed_col]))
    if side == "L":
        vx, vy, vz, speed, elbow = seq[peak_idx, [L_VX, L_VY, L_VZ, L_SPEED, L_ELBOW]]
    else:
        vx, vy, vz, speed, elbow = seq[peak_idx, [R_VX, R_VY, R_VZ, R_SPEED, R_ELBOW]]
    speed = max(float(speed), 1e-6)

    return {
        "hand_dist_last": float(seq[-lookback:, HANDS_DIST].mean()),
        "elbow_dist_last": float(seq[-lookback:, ELBOW_DIST].mean()),
        "max_l": max_l,
        "max_r": max_r,
        "active_speed": active_speed,
        "opposite_speed": opposite_speed,
        "active_side": side,
        "overall_max": overall_max,
        "min_max": min_max,
        "balance_ratio": min_max / max(overall_max, 1e-6),
        "side": side,
        "lateral_ratio": abs(float(vx)) / speed,
        "up_ratio": -float(vy) / speed,
        "forward_ratio": abs(float(vz)) / speed,
        "elbow_ratio": float(elbow),
    }


def classify_optimized_summary(summary, thresholds):
    """학습된 임계값으로 통계량 하나를 10개 클래스 중 하나로 분류한다."""
    t = {**DEFAULT_OPTIMIZED_THRESHOLDS, **thresholds}

    # 양팔 고속 동작은 마지막 자세가 손을 모은 상태여도 가드보다 먼저 본다.
    if (summary["min_max"] > t["wave_min_speed"]
            and summary["balance_ratio"] > t["wave_balance_ratio"]):
        return "ENERGY_WAVE"
    if (summary["hand_dist_last"] < t["hand_dist_guard"]
            and summary["elbow_dist_last"] < t["elbow_dist_guard"]):
        return "TWO_HAND_GUARD"
    if summary["overall_max"] < t["idle_speed_max"]:
        return "IDLE"

    # 양팔이 비슷하게 움직였지만 장풍 조건을 만족하지 못한 동작은 한쪽 펀치로
    # 억지 분류하지 않는다. OTHER/IDLE 오검출 비용을 높인 계획서의 안전장치다.
    if summary["balance_ratio"] > t["other_balance_min"]:
        return "OTHER"

    side_name = "LEFT" if summary["active_side"] == "L" else "RIGHT"
    if (summary["up_ratio"] > t["uppercut_vy"]
            and summary["elbow_ratio"] < t["uppercut_elbow_ratio"]):
        return f"{side_name}_UPPERCUT"
    if (summary["lateral_ratio"] > t["hook_vx"]
            and summary["elbow_ratio"] < t["hook_elbow_ratio"]):
        return f"{side_name}_HOOK"
    if summary["forward_ratio"] >= t["jab_vz_min"]:
        return f"{side_name}_JAB"
    return "OTHER"


def classify_optimized_heuristic_sequence(heur_seq, thresholds=None):
    thresholds = {**DEFAULT_OPTIMIZED_THRESHOLDS, **(thresholds or {})}
    summary = summarize_heuristic_sequence(heur_seq, thresholds["lookback_frames"])
    return classify_optimized_summary(summary, thresholds)


def classify_heuristic_sequence(heur_seq):
    """heur_seq: (T, 17) ndarray → 예측 라벨 문자열"""
    hd_last = heur_seq[-8:, HANDS_DIST].mean()
    ed_last = heur_seq[-8:, ELBOW_DIST].mean()
    max_l = heur_seq[:, L_SPEED].max()
    max_r = heur_seq[:, R_SPEED].max()
    overall_max = max(max_l, max_r)
    min_max = min(max_l, max_r)

    if hd_last < HAND_DIST_GUARD and ed_last < ELBOW_DIST_GUARD:
        return "TWO_HAND_GUARD"

    if overall_max < IDLE_SPEED_MAX:
        return "IDLE"

    if min_max > WAVE_MIN_SPEED and (min_max / overall_max) > WAVE_BALANCE_RATIO:
        return "ENERGY_WAVE"

    # 단일 팔 펀치 종류 판별 — punch_core.js classify() 이식
    side = "L" if max_l >= max_r else "R"
    if side == "L":
        peak_idx = int(np.argmax(heur_seq[:, L_SPEED]))
        vx, vy, speed, elbow = heur_seq[peak_idx, [L_VX, L_VY, L_SPEED, L_ELBOW]]
    else:
        peak_idx = int(np.argmax(heur_seq[:, R_SPEED]))
        vx, vy, speed, elbow = heur_seq[peak_idx, [R_VX, R_VY, R_SPEED, R_ELBOW]]

    s = max(speed, 1e-3)
    if (-vy / s) > UPPERCUT_VY and elbow < UPPERCUT_ELBOW_RATIO:
        kind = "UPPERCUT"
    elif (abs(vx) / s) > HOOK_VX and elbow < HOOK_ELBOW_RATIO:
        kind = "HOOK"
    else:
        kind = "JAB"

    return f"{'LEFT' if side == 'L' else 'RIGHT'}_{kind}"


def evaluate_rule_baseline(heuristic_seqs, label_names):
    preds = [classify_heuristic_sequence(seq) for seq in heuristic_seqs]
    correct = sum(p == t for p, t in zip(preds, label_names))
    acc = correct / len(label_names) if label_names else 0.0
    return acc, preds
