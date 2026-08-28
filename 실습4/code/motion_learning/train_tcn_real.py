"""
train_tcn_real.py — 실측 collected_pose 데이터로 causal TCN을 학습하고,
같은 데이터·같은 분할에서 rule-base와 정면으로 비교한다.

파이프라인:
  1) manifest.jsonl 로드 (캡처 품질 미달 샘플은 real_data.py 에서 이미 제외됨)
  2) outlier_filter.py 로 "라벨과 실제 동작이 어긋난" 샘플 탐지 → 제외
  3) 참가자 단위 LOSO(Leave-One-Subject-Out) 교차검증
       - 같은 사람이 반복한 동작이 train/test에 동시에 섞이면 정확도가 부풀려지므로
         반드시 참가자 단위로 쪼갠다.
       - 각 fold: TCN은 좌우 미러링 증강으로 학습, rule-base는 애초에 학습이 없으므로
         test fold에 대해서만 그대로 평가.
  4) 전체 LOSO 예측을 모아 rule-base vs TCN 정확도/혼동행렬을 나란히 비교, JSON으로 저장
  5) 필터링된 전체 데이터로 최종 모델을 다시 학습해 .pth / .onnx 로 내보냄
     (LOSO는 "일반화 성능 검증"용이고, 최종 배포 모델은 가진 데이터를 전부 쓴다)

실행: python train_tcn_real.py
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from real_data import build_raw_dataset, CLASSES, LABEL_TO_IDX
from outlier_filter import detect_outliers
from rule_baseline import evaluate_rule_baseline, classify_heuristic_sequence
from tcn_model import CausalMotionTCN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60
BATCH_SIZE = 16
LR = 1e-3

MIRROR_LABEL = {
    "LEFT_JAB": "RIGHT_JAB", "RIGHT_JAB": "LEFT_JAB",
    "LEFT_HOOK": "RIGHT_HOOK", "RIGHT_HOOK": "LEFT_HOOK",
    "LEFT_UPPERCUT": "RIGHT_UPPERCUT", "RIGHT_UPPERCUT": "LEFT_UPPERCUT",
    "IDLE": "IDLE", "TWO_HAND_GUARD": "TWO_HAND_GUARD",
    "ENERGY_WAVE": "ENERGY_WAVE", "OTHER": "OTHER",
}
# heuristic_7j_v1 열 순서(17차원):
#  0 left_elbow_angle_ratio   1 right_elbow_angle_ratio
#  2 left_reach               3 right_reach
#  4 left_wrist_vx  5 left_wrist_vy  6 left_wrist_vz
#  7 right_wrist_vx 8 right_wrist_vy 9 right_wrist_vz
# 10 left_wrist_speed        11 right_wrist_speed
# 12 hands_distance          13 left_wrist_to_nose  14 right_wrist_to_nose
# 15 elbow_distance          16 average_wrist_z
def mirror_sequence(seq):
    """좌우 반전 증강. left/right 열을 맞바꾸고, x축 속도(vx) 부호를 뒤집는다."""
    out = seq.copy()
    out[:, [0, 1]] = seq[:, [1, 0]]
    out[:, [2, 3]] = seq[:, [3, 2]]
    out[:, [4, 5, 6]] = seq[:, [7, 8, 9]] * np.array([-1, 1, 1])
    out[:, [7, 8, 9]] = seq[:, [4, 5, 6]] * np.array([-1, 1, 1])
    out[:, [10, 11]] = seq[:, [11, 10]]
    out[:, [13, 14]] = seq[:, [14, 13]]
    return out


def augment_with_mirror(X, y_names):
    mirrored_X = np.stack([mirror_sequence(s) for s in X])
    mirrored_y = np.array([MIRROR_LABEL[n] for n in y_names])
    return np.concatenate([X, mirrored_X], axis=0), np.concatenate([y_names, mirrored_y], axis=0)


def fit_scaler(X):
    """
    train fold 데이터만으로 채널별 median/MAD를 구한다 (test로 새는 정보 없음).
    game_7j_temporal_v2의 속도·가속도 채널은 프레임 간 dt가 짧을 때 이상치성 급증값이
    섞여 있어(관측치 최대 ~225), 이 스케일 그대로 넣으면 학습이 불안정해진다.
    """
    flat = X.reshape(-1, X.shape[-1])
    median = np.median(flat, axis=0)
    mad = np.median(np.abs(flat - median), axis=0)
    scale = 1.4826 * mad + 1e-3
    return median, scale


def apply_scaler(X, median, scale, clip=8.0):
    return np.clip((X - median) / scale, -clip, clip)


def train_tcn(X_train, y_train_idx, X_val_for_es=None, y_val_for_es=None):
    model = CausalMotionTCN().to(DEVICE)
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
def predict_tcn(model, X):
    model.eval()
    x = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    logits = model(x)
    return torch.argmax(logits, dim=1).cpu().numpy()


def run_loso(data, keep_mask):
    sequences = data["sequences"][keep_mask]
    labels_idx = data["labels"][keep_mask]
    labels_name = np.array([CLASSES[i] for i in labels_idx])
    participants = data["participants"][keep_mask]
    heuristic_seqs = [h for h, k in zip(data["heuristic_seqs"], keep_mask) if k]

    all_rule_pred, all_tcn_pred, all_true, all_true_idx = [], [], [], []
    fold_reports = []

    for held_out in sorted(set(participants)):
        train_mask = participants != held_out
        test_mask = ~train_mask

        X_train_raw = sequences[train_mask]
        y_train_names_raw = labels_name[train_mask]
        X_test = sequences[test_mask]
        y_test_idx = labels_idx[test_mask]
        y_test_names = labels_name[test_mask]
        heur_test = [h for h, m in zip(heuristic_seqs, test_mask) if m]

        # --- TCN: 좌우 미러링 증강 후 학습 ---
        X_train_aug, y_train_names_aug = augment_with_mirror(X_train_raw, y_train_names_raw)
        y_train_idx_aug = np.array([LABEL_TO_IDX[n] for n in y_train_names_aug])

        median, scale = fit_scaler(X_train_aug)
        X_train_scaled = apply_scaler(X_train_aug, median, scale)
        X_test_scaled = apply_scaler(X_test, median, scale)

        model = train_tcn(X_train_scaled, y_train_idx_aug)
        tcn_pred_idx = predict_tcn(model, X_test_scaled)
        tcn_pred_names = [CLASSES[i] for i in tcn_pred_idx]

        # --- Rule baseline: 학습 없이 test fold 그대로 평가 ---
        rule_acc, rule_pred_names = evaluate_rule_baseline(heur_test, list(y_test_names))

        tcn_acc = accuracy_score(y_test_idx, tcn_pred_idx)
        fold_reports.append({
            "held_out_participant": held_out,
            "n_test": int(test_mask.sum()),
            "rule_accuracy": float(rule_acc),
            "tcn_accuracy": float(tcn_acc),
        })
        print(f"[LOSO fold={held_out:10s} n={test_mask.sum():2d}] "
              f"rule={rule_acc*100:5.1f}%  tcn={tcn_acc*100:5.1f}%")

        all_rule_pred.extend(rule_pred_names)
        all_tcn_pred.extend(tcn_pred_names)
        all_true.extend(y_test_names)
        all_true_idx.extend(y_test_idx)

    return fold_reports, all_true, all_rule_pred, all_tcn_pred


def main():
    print("=" * 70)
    print("실측 collected_pose 데이터 기반 causal TCN 학습/평가")
    print("=" * 70)

    data = build_raw_dataset()
    n_total = len(data["labels"])
    print(f"[1] 캡처 품질 accepted 샘플: {n_total}개")

    # --- outlier(라벨-동작 불일치) 탐지 ---
    keep_mask, outlier_report = detect_outliers(
        data["summary_vecs"], data["labels"], data["sample_ids"], CLASSES
    )
    dropped = [r for r in outlier_report if r["dropped"]]
    print(f"[2] 라벨-동작 불일치 outlier {len(dropped)}개 제외 (남은 샘플 {keep_mask.sum()}개)")
    for r in dropped:
        print(f"    - drop {r['sample_id']}  label={r['label']:16s} z={r['distance_z']:.2f}")

    with open(os.path.join(BASE_DIR, "outlier_report.json"), "w", encoding="utf-8") as f:
        json.dump(outlier_report, f, ensure_ascii=False, indent=2)

    # --- LOSO 교차검증: rule-base vs TCN ---
    print("\n[3] LOSO 교차검증 (참가자 4명, 한 명씩 held-out)")
    fold_reports, y_true, rule_pred, tcn_pred = run_loso(data, keep_mask)

    rule_acc = accuracy_score(y_true, rule_pred)
    tcn_acc = accuracy_score(y_true, tcn_pred)
    rule_f1 = f1_score(y_true, rule_pred, average="macro", zero_division=0)
    tcn_f1 = f1_score(y_true, tcn_pred, average="macro", zero_division=0)

    print("\n" + "-" * 70)
    print(f"[전체 LOSO 결과]  rule-base: acc={rule_acc*100:.2f}%  macro-F1={rule_f1:.3f}")
    print(f"                  TCN      : acc={tcn_acc*100:.2f}%  macro-F1={tcn_f1:.3f}")
    print(f"                  개선폭   : +{(tcn_acc - rule_acc)*100:.2f}%p (accuracy)")
    print("-" * 70)

    print("\n--- Rule-base classification report ---")
    print(classification_report(y_true, rule_pred, labels=CLASSES, zero_division=0))
    print("--- TCN classification report ---")
    print(classification_report(y_true, tcn_pred, labels=CLASSES, zero_division=0))

    cm_rule = confusion_matrix(y_true, rule_pred, labels=CLASSES).tolist()
    cm_tcn = confusion_matrix(y_true, tcn_pred, labels=CLASSES).tolist()

    results = {
        "device": str(DEVICE),
        "n_total_accepted_capture": n_total,
        "n_outliers_dropped": len(dropped),
        "n_used_for_training": int(keep_mask.sum()),
        "classes": CLASSES,
        "loso_folds": fold_reports,
        "overall": {
            "rule_accuracy": float(rule_acc),
            "tcn_accuracy": float(tcn_acc),
            "rule_macro_f1": float(rule_f1),
            "tcn_macro_f1": float(tcn_f1),
            "improvement_pct_points": float((tcn_acc - rule_acc) * 100),
        },
        "confusion_matrix_rule": cm_rule,
        "confusion_matrix_tcn": cm_tcn,
    }
    with open(os.path.join(BASE_DIR, "eval_results_real.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] 평가 결과 저장: eval_results_real.json")

    # --- 최종 배포용 모델: 필터링된 데이터 전부 사용 ---
    print("\n[4] 최종 배포 모델 학습 (outlier 제외, 4명 데이터 전부 사용)")
    sequences = data["sequences"][keep_mask]
    labels_idx = data["labels"][keep_mask]
    labels_name = np.array([CLASSES[i] for i in labels_idx])
    X_aug, y_names_aug = augment_with_mirror(sequences, labels_name)
    y_idx_aug = np.array([LABEL_TO_IDX[n] for n in y_names_aug])

    final_median, final_scale = fit_scaler(X_aug)
    X_aug_scaled = apply_scaler(X_aug, final_median, final_scale)

    final_model = train_tcn(X_aug_scaled, y_idx_aug)
    torch.save(final_model.state_dict(), os.path.join(BASE_DIR, "boxing_tcn.pth"))
    print("[OK] 가중치 저장: boxing_tcn.pth")

    # 배포(브라우저/실시간) 쪽에서 입력 프레임에 반드시 같은 스케일링을 적용해야 하므로 함께 저장한다.
    scaler_path = os.path.join(BASE_DIR, "boxing_tcn_scaler.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
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
            "median": final_median.tolist(),
            "scale": final_scale.tolist(),
            "clip": 8.0,
        }, f, indent=2)
    print(f"[OK] 정규화 통계 저장: {scaler_path}")

    final_model.eval()
    dummy = torch.randn(1, 60, 17, device=DEVICE)
    onnx_path = os.path.join(BASE_DIR, "boxing_tcn.onnx")
    torch.onnx.export(
        final_model, dummy, onnx_path,
        input_names=["pose_sequence"], output_names=["class_logits"],
        dynamic_axes={"pose_sequence": {0: "batch"}, "class_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # torch 2.x 기본(dynamo) 경로는 onnxscript 필요 — 레거시 TorchScript 경로 사용
    )
    print(f"[✓] ONNX 내보내기: {onnx_path}")


if __name__ == "__main__":
    main()
