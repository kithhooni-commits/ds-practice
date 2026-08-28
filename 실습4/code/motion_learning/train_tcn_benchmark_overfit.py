"""train_tcn_benchmark_overfit.py — TCN을 "지금 이 User(hong)·이 benchmark 영상"에만 overfitting.

목적 (검증 실험, 일반화 모델 아님)
-----------------------------------
v4/v5의 `boxing_tcn.pth`는 `collected_pose/`(hong·cheols·kim·suhwan 4명, 브라우저 MediaPipe Pose
수집 클립)로 학습했지만, 평가는 전혀 다른 파이프라인(`pose_landmarker_full.task` Tasks API,
90초 스크립트 영상)으로 뽑은 `benchmark_landmarks.jsonl`에 대해 수행한다. 즉 TCN의 낮은 F1(0.379)이
① 모델 구조/피처 자체의 한계인지, ② 학습 분포와 평가 분포가 달라서 생긴 문제인지 구분이 안 된다.

이 스크립트는 ②를 격리해서 보기 위해, **평가에 쓰는 것과 똑같은 영상·같은 사람(hong)·같은 피처
추출 코드**로 학습 데이터를 직접 만든다. 즉 의도적으로 "학습=평가 영상"이라는 data leakage를
만든다 — 일반화 성능을 보려는 것이 아니라, "데이터 분포만 맞으면 이 모델·피처로 이 사람의 동작을
원칙적으로 배울 수 있는가"라는 capacity/upper-bound 질문에 답하기 위함이다.

데이터 소스
-----------
- `eval/video/benchmark_landmarks.jsonl` — 90초 전체 프레임의 MediaPipe 3D 랜드마크 (캐시됨)
- `eval/video/benchmark_labels.json`     — 29개 펀치의 정답 시각(t_ms)·팔(side)·종류(kind)
- 비동작 구간(준비/숨고르기/풋워크/마무리)은 `run_pipeline.py`의 90초 프로토콜 구간표와 동일하게
  정의해, 거기서 직접 "펀치 아님" 샘플(IDLE/OTHER)을 뽑는다 — 풋워크 오검출이 전 버전 공통
  문제였으므로 이걸 학습에 명시적으로 포함시키는 것이 핵심이다.

피처
----
`heuristic_7j_v1` 17차원. `eval/evaluate_video.py`의 `TCNMotionClassifier.push()`와
**완전히 동일한 수식**으로 여기서도 다시 계산한다(실시간 추론과 학습 데이터의 피처 정의가
어긋나면 비교 자체가 무의미해지므로, 아래 `extract_feat17()`은 그 함수를 그대로 옮긴 것이다).

Augmentation (모두 명시적으로 기록됨 — `overfit_hong/train_report.json` 참고)
---------------------------------------------------------------------------
29개 펀치 정답 중 가장 적은 클래스는 3개(R_HOOK, L_UPPERCUT, R_UPPERCUT)뿐이라 증강 없이는
학습이 불가능하다. 아래 4가지를 조합한다:

  1) **좌우 미러링**: 17차원 중 left/right 열을 맞바꾸고 x축 속도 부호를 반전 (`mirror_sequence`).
     LEFT_JAB <-> RIGHT_JAB 등 라벨도 함께 교체. 복싱 동작은 좌우 대칭이라는 가정.
  2) **타이밍 앵커 지터**: 정답 시각 t_ms 그대로뿐 아니라 {-150, 0, +150} ms 오프셋에서도
     60프레임 causal 윈도를 끊어 학습 샘플로 삼는다. 실시간 트리거는 GT 순간에 정확히 발사되지
     않고 ±수백ms 안에서 발사되므로(평가기 허용오차도 ±400ms), 그 변동성을 학습에 반영한다.
  3) **시간축 워핑(속도 변화)**: 60프레임 윈도를 선형보간으로 0.9배/1.1배 늘리거나 줄인 뒤
     다시 60프레임으로 causal left-pad — 같은 펀치를 조금 빠르게/느리게 수행한 변형을 만든다.
  4) **가우시안 피처 지터**: 채널별 robust 표준편차(MAD 기반)의 5%를 시그마로 하는 가우시안
     잡음을 프레임마다 독립적으로 더한다 — 포즈 추정 자체의 프레임 단위 떨림(jitter)을 모사.

  적용 순서: (좌우미러 있음/없음) x (시간축 워핑 0.9/1.0/1.1) = 6가지 "깨끗한" 변형을 만들고,
  그 각각에 가우시안 지터를 N회 추가로 씌워 복제한다. 펀치(양성) 샘플은 N=2, 비동작(음성) 샘플은
  이미 구간 전체에서 조밀하게 뽑으므로 N=1만 추가한다.

학습
----
`train_tcn_real.py`와 동일한 하이퍼파라미터(60 epoch, AdamW lr=1e-3, batch=16, median/MAD
스케일러)를 그대로 써서 다른 버전과 비교 가능하게 한다. 가중치는 기존 `boxing_tcn.pth`를
덮어쓰지 않고 `motion_learning/overfit_hong/`에 별도 저장한다.

실행:
  python motion_learning/train_tcn_benchmark_overfit.py
"""
import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "..", "eval")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, EVAL_DIR)

from scoring import iter_landmarks_jsonl  # 순수 python, cv2 의존성 없음
from tcn_model import CausalMotionTCN

BENCHMARK_LANDMARKS = os.path.join(EVAL_DIR, "video", "benchmark_landmarks.jsonl")
BENCHMARK_LABELS = os.path.join(EVAL_DIR, "video", "benchmark_labels.json")
OUT_DIR = os.path.join(BASE_DIR, "overfit_hong")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60
BATCH_SIZE = 16
LR = 1e-3

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16

CLASSES = [
    "IDLE", "LEFT_JAB", "RIGHT_JAB", "LEFT_HOOK", "RIGHT_HOOK",
    "LEFT_UPPERCUT", "RIGHT_UPPERCUT", "TWO_HAND_GUARD", "ENERGY_WAVE", "OTHER",
]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SEQ_LEN = 60
DIM = 17

SIDE_KIND_TO_LABEL = {
    ("L", "STRAIGHT"): "LEFT_JAB", ("R", "STRAIGHT"): "RIGHT_JAB",
    ("L", "HOOK"): "LEFT_HOOK", ("R", "HOOK"): "RIGHT_HOOK",
    ("L", "UPPERCUT"): "LEFT_UPPERCUT", ("R", "UPPERCUT"): "RIGHT_UPPERCUT",
}

MIRROR_LABEL = {
    "LEFT_JAB": "RIGHT_JAB", "RIGHT_JAB": "LEFT_JAB",
    "LEFT_HOOK": "RIGHT_HOOK", "RIGHT_HOOK": "LEFT_HOOK",
    "LEFT_UPPERCUT": "RIGHT_UPPERCUT", "RIGHT_UPPERCUT": "LEFT_UPPERCUT",
    "IDLE": "IDLE", "OTHER": "OTHER",
}

# 90초 표준 프로토콜 중 "펀치가 없어야 하는" 구간 (run_pipeline.py calculate_phase_metrics와 동일 경계).
# 풋워크는 몸을 움직이므로 OTHER(활동성 비-펀치), 나머지는 IDLE(정지성 비-펀치)로 구분해 학습시킨다.
NON_ACTION_PHASES = [
    (0, 6, "IDLE"),      # 1. 준비 (Calibration)
    (18, 23, "IDLE"),    # Rest 1
    (35, 40, "IDLE"),    # Rest 2
    (52, 57, "IDLE"),    # Rest 3
    (57, 70, "OTHER"),   # 5. 풋워크 (Footwork) — 몸은 움직이지만 펀치가 아님
    (70, 75, "IDLE"),    # Rest 4
    (85, 90, "IDLE"),    # 7. 마무리 (Cooldown)
]

ANCHOR_OFFSETS_MS = [-150, 0, 150]
SPEED_FACTORS = [0.9, 1.0, 1.1]
GAUSSIAN_SIGMA_SCALE = 0.05
N_NOISE_POSITIVE = 2
N_NOISE_NEGATIVE = 1
NEGATIVE_STRIDE_MS = 400


# ==================== 피처 추출 (evaluate_video.py::TCNMotionClassifier.push 와 동일 수식) ====================
def dist3(a, b):
    return math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)


def angle_deg(a, b, c):
    abx, aby, abz = a.x - b.x, a.y - b.y, a.z - b.z
    cbx, cby, cbz = c.x - b.x, c.y - b.y, c.z - b.z
    d = math.hypot(abx, aby, abz) * math.hypot(cbx, cby, cbz)
    if d < 1e-6:
        return 180.0
    cosv = max(-1.0, min(1.0, (abx * cbx + aby * cby + abz * cbz) / d))
    return math.degrees(math.acos(cosv))


def extract_feat17(lm, prev_l, prev_r, prev_t, now_ms):
    nose, lsh, rsh = lm[NOSE], lm[L_SH], lm[R_SH]
    lel, rel, lwr, rwr = lm[L_EL], lm[R_EL], lm[L_WR], lm[R_WR]

    sh2d = max(math.hypot(lsh.x - rsh.x, lsh.y - rsh.y), 1e-3)
    l_el_ratio = angle_deg(lsh, lel, lwr) / 180.0
    r_el_ratio = angle_deg(rsh, rel, rwr) / 180.0
    l_reach = dist3(lwr, lsh) / sh2d
    r_reach = dist3(rwr, rsh) / sh2d
    hands_dist = dist3(lwr, rwr) / sh2d
    l_wrist_nose = dist3(lwr, nose) / sh2d
    r_wrist_nose = dist3(rwr, nose) / sh2d
    elbow_dist = dist3(lel, rel) / sh2d
    avg_wrist_z = ((lwr.z + rwr.z) / 2.0) / sh2d

    lvx = lvy = lvz = rvx = rvy = rvz = 0.0
    dt = (now_ms - prev_t) / 1000.0 if prev_t else 0.0
    if prev_l and 0.008 < dt < 0.4:
        lvx = (lwr.x - prev_l[0]) / dt / sh2d
        lvy = (lwr.y - prev_l[1]) / dt / sh2d
        lvz = (lwr.z - prev_l[2]) / dt / sh2d
        rvx = (rwr.x - prev_r[0]) / dt / sh2d
        rvy = (rwr.y - prev_r[1]) / dt / sh2d
        rvz = (rwr.z - prev_r[2]) / dt / sh2d

    l_speed = math.hypot(lvx, lvy, lvz)
    r_speed = math.hypot(rvx, rvy, rvz)

    feat17 = [
        l_el_ratio, r_el_ratio, l_reach, r_reach,
        lvx, lvy, lvz, rvx, rvy, rvz,
        l_speed, r_speed, hands_dist, l_wrist_nose, r_wrist_nose, elbow_dist, avg_wrist_z,
    ]
    return feat17, (lwr.x, lwr.y, lwr.z), (rwr.x, rwr.y, rwr.z)


def build_full_sequence():
    """benchmark_landmarks.jsonl 전체를 순서대로 재생해 (t_ms 리스트, 17차원 피처 배열)을 만든다."""
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
    """anchor_ms 시점까지의 causal 60프레임 윈도.
    TCNMotionClassifier.guess_punch_kind() / real_data._left_pad_causal 과 동일하게,
    프레임이 60개보다 적으면 가장 오래된(첫) 프레임을 반복해 과거 쪽을 채운다."""
    idx = np.searchsorted(t_arr, anchor_ms, side="right") - 1
    if idx < 0:
        return None
    buf = f_arr[: idx + 1]
    n = buf.shape[0]
    if n >= SEQ_LEN:
        return buf[-SEQ_LEN:].copy()
    pad = np.repeat(buf[:1], SEQ_LEN - n, axis=0)
    return np.concatenate([pad, buf], axis=0)


# ==================== Augmentation ====================
def mirror_sequence(seq):
    """좌우 반전: left/right 열을 맞바꾸고 x축 속도(vx) 부호를 뒤집는다. (train_tcn_real.py와 동일 로직)"""
    out = seq.copy()
    out[:, [0, 1]] = seq[:, [1, 0]]
    out[:, [2, 3]] = seq[:, [3, 2]]
    out[:, [4, 5, 6]] = seq[:, [7, 8, 9]] * np.array([-1, 1, 1])
    out[:, [7, 8, 9]] = seq[:, [4, 5, 6]] * np.array([-1, 1, 1])
    out[:, [10, 11]] = seq[:, [11, 10]]
    out[:, [13, 14]] = seq[:, [14, 13]]
    return out


def time_warp(seq, factor):
    """60프레임을 `factor`배 길이로 선형보간 리샘플한 뒤, causal left-pad로 다시 60프레임에 맞춘다."""
    if factor == 1.0:
        return seq.copy()
    t_orig = np.linspace(0.0, 1.0, SEQ_LEN)
    n_new = max(2, int(round(SEQ_LEN * factor)))
    t_new = np.linspace(0.0, 1.0, n_new)
    warped = np.stack([np.interp(t_new, t_orig, seq[:, c]) for c in range(seq.shape[1])], axis=1)
    n = warped.shape[0]
    if n >= SEQ_LEN:
        return warped[-SEQ_LEN:].astype(np.float32)
    pad = np.repeat(warped[:1], SEQ_LEN - n, axis=0)
    return np.concatenate([pad, warped], axis=0).astype(np.float32)


def gaussian_jitter(seq, sigma_scale, rng):
    """채널별 robust 표준편차(MAD 기반)의 sigma_scale배를 시그마로 하는 가우시안 잡음을 더한다."""
    med = np.median(seq, axis=0, keepdims=True)
    mad = np.median(np.abs(seq - med), axis=0)
    sigma = (1.4826 * mad * sigma_scale + 1e-4).astype(np.float32)
    noise = rng.normal(0.0, sigma, size=seq.shape).astype(np.float32)
    return seq + noise


def augment_one(seq, label, rng, n_noise, allow_mirror=True):
    """(좌우미러 유/무) x (속도 워핑 0.9/1.0/1.1) 6변형 + 각 변형에 가우시안 잡음 n_noise회 복제."""
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


def build_dataset(t_arr, f_arr, rng):
    labels_data = json.loads(open(BENCHMARK_LABELS, encoding="utf-8").read())
    punches = labels_data["punches"]

    pos_base = []
    for p in punches:
        label = SIDE_KIND_TO_LABEL[(p["side"], p["kind"])]
        for off in ANCHOR_OFFSETS_MS:
            w = window_ending_at(t_arr, f_arr, p["t_ms"] + off)
            if w is not None:
                pos_base.append((w, label))

    neg_base = []
    for t0, t1, label in NON_ACTION_PHASES:
        t = t0 * 1000
        while t < t1 * 1000:
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
        "n_positive_base_windows": len(pos_base),
        "n_negative_base_windows": len(neg_base),
        "n_total_augmented_samples": len(samples),
        "class_counts": {c: int((y_names == c).sum()) for c in CLASSES if (y_names == c).any()},
    }
    return X, y_names, stats


# ==================== 학습 (train_tcn_real.py와 동일 하이퍼파라미터) ====================
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
    for ep in range(EPOCHS):
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
    print("TCN overfit-to-user(hong) 실험 — benchmark 영상 자체로 학습 데이터 구성")
    print("=" * 70)

    print("[1] benchmark_landmarks.jsonl 재생 → 17차원 피처 시퀀스 생성")
    t_arr, f_arr = build_full_sequence()
    print(f"    프레임 수: {len(t_arr)} (영상 길이 {t_arr[-1]/1000:.1f}s)")

    print("[2] 정답 펀치(29개) + 비동작 구간에서 학습 샘플 추출 + Augmentation")
    X, y_names, stats = build_dataset(t_arr, f_arr, rng)
    print(f"    증강 전 base 윈도: 양성 {stats['n_positive_base_windows']}개, "
          f"음성 {stats['n_negative_base_windows']}개")
    print(f"    증강 후 최종 샘플: {stats['n_total_augmented_samples']}개")
    for c, n in sorted(stats["class_counts"].items()):
        print(f"      - {c:16s}: {n}")

    y_idx = np.array([LABEL_TO_IDX[n] for n in y_names])

    print("\n[3] median/MAD 스케일러 적합 + 정규화")
    median, scale = fit_scaler(X)
    X_scaled = apply_scaler(X, median, scale)

    print(f"[4] Causal TCN 학습 ({EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}, device={DEVICE})")
    model = train_tcn(X_scaled, y_idx)

    print("[5] 학습 데이터 자체에 대한 적합도 확인 (overfitting 여부 점검용, 일반화 지표 아님)")
    pred_idx = predict(model, X_scaled)
    train_acc = float((pred_idx == y_idx).mean())
    print(f"    학습셋 재현 정확도: {train_acc*100:.2f}%  (100%에 가까우면 '이 데이터는 외울 수 있다'는 뜻)")

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
        "purpose": "TCN을 benchmark.mp4(hong)에 overfitting시켜 분포 불일치 가설을 검증하는 실험용 모델. "
                   "train/eval 데이터가 같은 영상에서 나왔으므로 일반화 성능 지표가 아님.",
        "source_video": "eval/video/benchmark.mp4",
        "n_frames_replayed": int(len(t_arr)),
        "dataset_stats": stats,
        "augmentation": {
            "mirror": "좌우 반전 (left/right 채널 swap + vx 부호 반전, 라벨도 LEFT<->RIGHT 교체)",
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
    print("다음 단계: evaluate_video.py --engine tcn --tcn-model-dir motion_learning/overfit_hong 로 "
          "benchmark 전체 영상에 대해 end-to-end 채점")


if __name__ == "__main__":
    main()
