"""train_tcn_v6b_trigger.py — v6b: TCN이 "펀치 종류"뿐 아니라 "펀치가 언제 나가는지(트리거)"까지
직접 담당하도록 학습한다. v6(overfit_hong)과 달리, benchmark.mp4의 29개 정답 중 **테스트셋으로
뺀 15개는 학습에서 완전히 제외**한다(데이터 leakage 방지).

=== Train/Test 분할 설계 (왜 이렇게 나눴는가) ===

단순 "시간 앞부분=train, 뒷부분=test"로 자르면 한쪽 펀치 종류가 test에서 통째로 빠질 위험이
있다(예: uppercut 클러스터가 전부 뒤쪽에 있으면 test에 uppercut이 하나도 안 남음). 그래서
"클러스터(직선/훅/어퍼컷)마다 마지막 1~2회"를 test로 빼는 **클러스터별 홀드아웃**을 쓴다.
대신 "실전 콤보"(75~85s) 구간은 한 덩어리로 test에 통째로 둔다 — 이 구간의 이벤트들이
433ms~1.5s 간격으로 서로 너무 붙어 있어서 학습/테스트 경계에 필요한 버퍼(아래)를 둘 자리가
없고, 오히려 "따로따로 배운 3가지 펀치를 빠른 실전 콤보에서도 알아보는가"라는 더 현실적인
일반화 테스트가 된다.

  TRAIN (14개 정답): 각 클러스터의 앞쪽 반복들
    - 직선: L 7500,9000,11133(3) / R 14500,16000,17200(3)
    - 훅  : L 24500,26666(2) / R 30000,32000(2)
    - 어퍼: L 41500,43500(2) / R 47500,49500(2)
  TEST  (15개 정답): 각 클러스터의 마지막 반복 + 콤보 구간 전체
    - 직선 마지막: L 13133, R 18400
    - 훅   마지막: L 28500, R 34000
    - 어퍼 마지막: L 45500, R 51500
    - 콤보 9개(76500~83000): STRAIGHT x8 + HOOK x1

=== Leakage 방지 규칙 ===

1. **시간 버퍼(purge)**: 어떤 학습 샘플의 60프레임 causal 윈도(끝 시각 기준 과거 약 2초)도
   테스트 이벤트의 시각에서 `PURGE_MS`(2500ms) 이내로 들어오면 학습 후보에서 제외한다.
   윈도 길이(약 2000ms)보다 더 넉넉하게 잡아, 프레임 경계에서 우연히 겹치는 일이 없게 한다.
2. **음성(비펀치) 샘플도 동일 규칙 적용**: 휴식/풋워크 구간에서 뽑는 "펀치 아님" 샘플도
   테스트 존에서 PURGE_MS 이내면 제외한다.
3. **평가는 반대로, 테스트 구간에서만** — `evaluate_tcn_v6b.py`가 전체 90초를 causal하게
   한 번 재생하지만, 채점은 TEST 시각 범위에 해당하는 예측·정답만 사용한다(학습에 쓰인 시간대의
   예측은 점수에 아예 포함하지 않음 — 그 구간에서 잘하는 건 당연하므로 증거가 안 됨).

=== Augmentation (train 쪽에만 적용, v6와 동일한 4종) ===
좌우 미러링 / 타이밍 앵커 지터(±150ms, purge 체크 통과하는 것만) / 시간축 워핑(0.9/1.0/1.1배) /
가우시안 피처 지터(MAD 기반 5%). 상세 근거는 `train_tcn_benchmark_overfit.py` docstring 참고.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "..", "eval")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, EVAL_DIR)

from scoring import iter_landmarks_jsonl
from tcn_model import CausalMotionTCN
from train_tcn_benchmark_overfit import (
    extract_feat17, mirror_sequence, time_warp, gaussian_jitter,
    CLASSES, LABEL_TO_IDX, MIRROR_LABEL, SEQ_LEN, DIM,
    ANCHOR_OFFSETS_MS, SPEED_FACTORS, GAUSSIAN_SIGMA_SCALE,
    N_NOISE_POSITIVE, N_NOISE_NEGATIVE, NEGATIVE_STRIDE_MS,
    NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR,
)

BENCHMARK_LANDMARKS = os.path.join(EVAL_DIR, "video", "benchmark_landmarks.jsonl")
OUT_DIR = os.path.join(BASE_DIR, "v6b_tcn_trigger")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60
BATCH_SIZE = 16
LR = 1e-3
PURGE_MS = 2500  # 테스트 이벤트 주변 이 범위 안의 학습 후보는 전부 제외

SIDE_KIND_TO_LABEL = {
    ("L", "STRAIGHT"): "LEFT_JAB", ("R", "STRAIGHT"): "RIGHT_JAB",
    ("L", "HOOK"): "LEFT_HOOK", ("R", "HOOK"): "RIGHT_HOOK",
    ("L", "UPPERCUT"): "LEFT_UPPERCUT", ("R", "UPPERCUT"): "RIGHT_UPPERCUT",
}

# ---- train/test 분할 (docstring 표와 동일) ----
TRAIN_GT = [
    (7500, "L", "STRAIGHT"), (9000, "L", "STRAIGHT"), (11133, "L", "STRAIGHT"),
    (14500, "R", "STRAIGHT"), (16000, "R", "STRAIGHT"), (17200, "R", "STRAIGHT"),
    (24500, "L", "HOOK"), (26666, "L", "HOOK"),
    (30000, "R", "HOOK"), (32000, "R", "HOOK"),
    (41500, "L", "UPPERCUT"), (43500, "L", "UPPERCUT"),
    (47500, "R", "UPPERCUT"), (49500, "R", "UPPERCUT"),
]
TEST_GT = [
    (13133, "L", "STRAIGHT"), (18400, "R", "STRAIGHT"),
    (28500, "L", "HOOK"), (34000, "R", "HOOK"),
    (45500, "L", "UPPERCUT"), (51500, "R", "UPPERCUT"),
    (76500, "L", "STRAIGHT"), (76933, "R", "STRAIGHT"),
    (78200, "L", "STRAIGHT"), (78900, "R", "STRAIGHT"),
    (80000, "L", "STRAIGHT"), (80466, "R", "STRAIGHT"),
    (81800, "L", "STRAIGHT"), (82366, "R", "STRAIGHT"),
    (83000, "L", "HOOK"),
]
TEST_TIMES_MS = sorted(t for t, _, _ in TEST_GT)

# 학습용 음성(비펀치) 샘플은 전 구간에서 고르게 뽑은 뒤 purge로 테스트 근접분만 쳐낸다.
# (test anchor 근처에도 "펀치 아님" 학습 후보가 purge로 자연히 제거됨)
NON_ACTION_PHASES = [
    (0, 6, "IDLE"), (18, 23, "IDLE"), (35, 40, "IDLE"),
    (52, 57, "IDLE"), (57, 70, "OTHER"), (70, 75, "IDLE"), (85, 90, "IDLE"),
]


def is_purged(t_ms):
    """t_ms 의 60프레임 causal 윈도가 어떤 테스트 이벤트 근처에 들어오면 True."""
    return any(abs(t_ms - tt) < PURGE_MS for tt in TEST_TIMES_MS)


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


def window_ending_at(t_arr, f_arr, anchor_ms):
    idx = np.searchsorted(t_arr, anchor_ms, side="right") - 1
    if idx < 0:
        return None
    buf = f_arr[: idx + 1]
    n = buf.shape[0]
    if n >= SEQ_LEN:
        return buf[-SEQ_LEN:].copy()
    pad = np.repeat(buf[:1], SEQ_LEN - n, axis=0)
    return np.concatenate([pad, buf], axis=0)


def augment_one(seq, label, rng, n_noise, allow_mirror=True):
    out = []
    base_variants = [(seq, label)]
    if allow_mirror:
        base_variants.append((mirror_sequence(seq), MIRROR_LABEL[label]))
    for s, lab in base_variants:
        for factor in SPEED_FACTORS:
            warped = time_warp(s, factor)
            out.append((warped, lab))
            for _ in range(n_noise):
                out.append((gaussian_jitter(warped, GAUSSIAN_SIGMA_SCALE, rng), lab))
    return out


def build_train_dataset(t_arr, f_arr, rng):
    pos_base = []
    n_purged_pos = 0
    for t_ms, side, kind in TRAIN_GT:
        label = SIDE_KIND_TO_LABEL[(side, kind)]
        for off in ANCHOR_OFFSETS_MS:
            anchor = t_ms + off
            if is_purged(anchor):
                n_purged_pos += 1
                continue
            w = window_ending_at(t_arr, f_arr, anchor)
            if w is not None:
                pos_base.append((w, label))

    neg_base = []
    n_purged_neg = 0
    for p0, p1, label in NON_ACTION_PHASES:
        t = p0 * 1000
        while t < p1 * 1000:
            if is_purged(t):
                n_purged_neg += 1
            else:
                w = window_ending_at(t_arr, f_arr, t)
                if w is not None:
                    neg_base.append((w, label))
            t += NEGATIVE_STRIDE_MS

    samples = []
    for seq, label in pos_base:
        samples.extend(augment_one(seq, label, rng, n_noise=N_NOISE_POSITIVE))
    for seq, label in neg_base:
        samples.extend(augment_one(seq, label, rng, n_noise=N_NOISE_NEGATIVE))

    X = np.stack([s for s, _ in samples]).astype(np.float32)
    y_names = np.array([lab for _, lab in samples])
    stats = {
        "n_train_gt_events": len(TRAIN_GT),
        "n_test_gt_events_excluded": len(TEST_GT),
        "purge_radius_ms": PURGE_MS,
        "n_positive_base_windows_kept": len(pos_base),
        "n_positive_anchor_candidates_purged": n_purged_pos,
        "n_negative_base_windows_kept": len(neg_base),
        "n_negative_candidates_purged": n_purged_neg,
        "n_total_augmented_train_samples": len(samples),
        "class_counts": {c: int((y_names == c).sum()) for c in CLASSES if (y_names == c).any()},
    }
    return X, y_names, stats


def fit_scaler(X):
    flat = X.reshape(-1, X.shape[-1])
    median = np.median(flat, axis=0)
    mad = np.median(np.abs(flat - median), axis=0)
    scale = 1.4826 * mad + 1e-3
    return median, scale


def apply_scaler(X, median, scale, clip=8.0):
    return np.clip((X - median) / scale, -clip, clip)


def train_tcn(X_train, y_train_idx):
    model = CausalMotionTCN(input_dim=DIM, num_classes=len(CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train_idx, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict(model, X):
    model.eval()
    x = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    return torch.argmax(model(x), dim=1).cpu().numpy()


def main():
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("v6b: TCN이 트리거+분류를 모두 담당 — leakage 방지 train/test 분할")
    print("=" * 70)

    print("[1] benchmark_landmarks.jsonl 재생 → 17차원 피처 시퀀스")
    t_arr, f_arr = build_full_sequence()
    print(f"    프레임 수: {len(t_arr)}")

    print(f"[2] TRAIN GT {len(TRAIN_GT)}개 / TEST GT {len(TEST_GT)}개 (test는 학습에서 완전 제외)")
    print(f"    purge 반경 ±{PURGE_MS}ms — 이 안에 들어오는 학습 후보(양성+음성)는 전부 버림")
    X, y_names, stats = build_train_dataset(t_arr, f_arr, rng)
    print(f"    양성 base 윈도: {stats['n_positive_base_windows_kept']}개 유지, "
          f"{stats['n_positive_anchor_candidates_purged']}개 purge로 제외")
    print(f"    음성 base 윈도: {stats['n_negative_base_windows_kept']}개 유지, "
          f"{stats['n_negative_candidates_purged']}개 purge로 제외")
    print(f"    증강 후 최종 학습 샘플: {stats['n_total_augmented_train_samples']}개")
    for c, n in sorted(stats["class_counts"].items()):
        print(f"      - {c:16s}: {n}")

    y_idx = np.array([LABEL_TO_IDX[n] for n in y_names])

    print("\n[3] median/MAD 스케일러 적합 (train 데이터만 사용)")
    median, scale = fit_scaler(X)
    X_scaled = apply_scaler(X, median, scale)

    print(f"[4] Causal TCN 학습 ({EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}, device={DEVICE})")
    model = train_tcn(X_scaled, y_idx)

    pred_idx = predict(model, X_scaled)
    train_acc = float((pred_idx == y_idx).mean())
    print(f"[5] 학습셋 자기 재현 정확도: {train_acc*100:.2f}% (참고용 — test 누설 없음의 증거는 아님)")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "boxing_tcn.pth"))
    with open(os.path.join(OUT_DIR, "boxing_tcn_scaler.json"), "w", encoding="utf-8") as f:
        json.dump({
            "feature_set": "heuristic_7j_v1",
            "feature_names": [
                "left_elbow_angle_ratio", "right_elbow_angle_ratio", "left_reach", "right_reach",
                "left_wrist_vx", "left_wrist_vy", "left_wrist_vz",
                "right_wrist_vx", "right_wrist_vy", "right_wrist_vz",
                "left_wrist_speed", "right_wrist_speed", "hands_distance",
                "left_wrist_to_nose", "right_wrist_to_nose", "elbow_distance", "average_wrist_z",
            ],
            "preprocessing": "(x - median) / scale, 이후 [-clip, clip]으로 clip",
            "median": median.tolist(),
            "scale": scale.tolist(),
            "clip": 8.0,
        }, f, indent=2)

    report = {
        "purpose": "v6b: TCN이 트리거(언제 나가는지)까지 직접 담당. benchmark.mp4에서 "
                   "테스트용으로 지정한 15개 정답(및 그 주변 ±2500ms)은 학습에서 완전히 제외했다.",
        "train_gt_events": TRAIN_GT,
        "test_gt_events_excluded_from_training": TEST_GT,
        "purge_radius_ms": PURGE_MS,
        "dataset_stats": stats,
        "augmentation": {
            "mirror": "좌우 반전 (train_tcn_benchmark_overfit.py와 동일)",
            "anchor_time_jitter_ms": ANCHOR_OFFSETS_MS,
            "time_warp_speed_factors": SPEED_FACTORS,
            "gaussian_jitter_sigma_scale": GAUSSIAN_SIGMA_SCALE,
            "n_noise_replicas_positive": N_NOISE_POSITIVE,
            "n_noise_replicas_negative": N_NOISE_NEGATIVE,
            "negative_sampling_stride_ms": NEGATIVE_STRIDE_MS,
        },
        "train_hyperparams": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "device": str(DEVICE)},
        "train_set_self_accuracy": train_acc,
    }
    with open(os.path.join(OUT_DIR, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 저장 완료: {OUT_DIR}/boxing_tcn.pth, boxing_tcn_scaler.json, train_report.json")


if __name__ == "__main__":
    main()
