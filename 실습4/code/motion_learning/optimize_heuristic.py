"""JSON 실측 데이터로 휴리스틱 임계값을 LOSO 방식으로 최적화한다.

산출물:
  artifacts/heuristic_thresholds.json
  artifacts/heuristic_metrics.json
  server/static/models/heuristic_thresholds.json (브라우저 배포본)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from outlier_filter import detect_outliers
from real_data import CLASSES, build_raw_dataset
from rule_baseline import (
    DEFAULT_OPTIMIZED_THRESHOLDS,
    classify_heuristic_sequence,
    classify_optimized_summary,
    mirror_heuristic_sequence,
    summarize_heuristic_sequence,
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DEPLOY_PATH = BASE_DIR.parent / "server" / "static" / "models" / "heuristic_thresholds.json"
ATTACK_LABELS = {
    "LEFT_JAB", "RIGHT_JAB", "LEFT_HOOK", "RIGHT_HOOK",
    "LEFT_UPPERCUT", "RIGHT_UPPERCUT", "ENERGY_WAVE",
}
SAFE_LABELS = {"IDLE", "OTHER"}

RANGES = {
    "hand_dist_guard": (0.55, 1.15),
    "elbow_dist_guard": (1.05, 1.90),
    "idle_speed_max": (3.0, 16.0),
    "wave_min_speed": (5.0, 28.0),
    "wave_balance_ratio": (0.45, 0.95),
    "other_balance_min": (0.55, 0.99),
    "uppercut_vy": (0.10, 0.60),
    "uppercut_elbow_ratio": (0.55, 0.95),
    "hook_vx": (0.08, 0.55),
    "hook_elbow_ratio": (0.55, 0.98),
    "jab_vz_min": (0.65, 0.995),
}


def _predict(summaries, thresholds):
    return [classify_optimized_summary(s, thresholds) for s in summaries]


MIRROR_LABEL = {
    "LEFT_JAB": "RIGHT_JAB", "RIGHT_JAB": "LEFT_JAB",
    "LEFT_HOOK": "RIGHT_HOOK", "RIGHT_HOOK": "LEFT_HOOK",
    "LEFT_UPPERCUT": "RIGHT_UPPERCUT", "RIGHT_UPPERCUT": "LEFT_UPPERCUT",
    "IDLE": "IDLE", "OTHER": "OTHER",
    "TWO_HAND_GUARD": "TWO_HAND_GUARD", "ENERGY_WAVE": "ENERGY_WAVE",
}
SPATIAL_COLUMNS = np.array([2, 3, 12, 13, 14, 15, 16])
VELOCITY_COLUMNS = np.array([4, 5, 6, 7, 8, 9, 10, 11])


def _resample(seq, length):
    """Resample one feature sequence without using data outside its clip."""
    seq = np.asarray(seq, dtype=np.float32)
    if length < 2 or len(seq) < 2:
        return seq.copy()
    source = np.linspace(0.0, 1.0, len(seq))
    target = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(target, source, seq[:, col]) for col in range(seq.shape[1])], axis=1).astype(np.float32)


def _time_scale(seq, factor):
    """0.8x/1.2x action duration; velocity features scale with elapsed time."""
    out = _resample(seq, max(2, int(round(len(seq) * factor))))
    out[:, VELOCITY_COLUMNS] /= factor
    return out


def _drop_frames_and_impute(seq, rng, ratio=0.08):
    """Simulate dropped pose frames, then linearly impute them."""
    out = np.asarray(seq, dtype=np.float32).copy()
    if len(out) < 5:
        return out
    count = max(1, int(round((len(out) - 2) * ratio)))
    dropped = rng.choice(np.arange(1, len(out) - 1), size=count, replace=False)
    kept = np.ones(len(out), dtype=bool); kept[dropped] = False
    positions = np.arange(len(out)); source = positions[kept]
    for col in range(out.shape[1]):
        out[:, col] = np.interp(positions, source, out[kept, col])
    return out


def _feature_noise(seq, rng):
    """Small landmark-coordinate noise expressed in the derived feature space."""
    out = np.asarray(seq, dtype=np.float32).copy()
    floor = np.array([.015, .015, .01, .01, .08, .08, .08, .08, .08, .08, .10, .10, .01, .01, .01, .01, .01], dtype=np.float32)
    scale = np.maximum(out.std(axis=0), floor)
    out += rng.normal(0.0, scale * 0.025, size=out.shape).astype(np.float32)
    out[:, [0, 1]] = np.clip(out[:, [0, 1]], 0.0, 1.0)
    out[:, [2, 3, 10, 11, 12, 13, 14, 15]] = np.maximum(out[:, [2, 3, 10, 11, 12, 13, 14, 15]], 0.0)
    return out


def _visibility_loss_proxy(seq, rng):
    """Brief low visibility represented by an imputed 2–4 frame feature span.

    heuristic_7j_v1 has no visibility channel, so this is the safe equivalent for
    the derived-feature optimizer; raw pose visibility is not invented.
    """
    out = np.asarray(seq, dtype=np.float32).copy()
    if len(out) < 6:
        return out
    width = int(rng.integers(2, min(5, len(out) - 2)))
    start = int(rng.integers(1, len(out) - width))
    end = start + width
    out[start:end] = np.linspace(out[start - 1], out[end], width + 2, dtype=np.float32)[1:-1]
    return out


def _camera_distance_variant(seq, rng):
    """Perturb relative-distance channels for small framing/camera-distance changes."""
    out = np.asarray(seq, dtype=np.float32).copy()
    out[:, SPATIAL_COLUMNS] *= float(rng.uniform(0.92, 1.08))
    out[:, 16] *= float(rng.uniform(0.96, 1.04))
    return out


def _prefix(seq, fraction):
    """Return an action-prefix window; only called for attack-labelled clips."""
    return np.asarray(seq, dtype=np.float32)[:max(2, int(round(len(seq) * fraction)))].copy()


def _augment_all(sequences, labels, seed):
    """Generate every plan-recommended augmentation for training data only."""
    rng = np.random.default_rng(seed)
    augmented_sequences, augmented_labels = list(sequences), list(labels)
    counts = {"original": len(labels), "horizontal_mirror": 0, "time_scale_0.8": 0,
              "time_scale_1.2": 0, "frame_dropout_imputed": 0, "feature_noise": 0,
              "visibility_loss_imputed": 0, "camera_distance": 0, "action_prefix": 0}
    for seq, label in zip(sequences, labels):
        variants = [
            (mirror_heuristic_sequence(seq), MIRROR_LABEL[label], "horizontal_mirror"),
            (_time_scale(seq, 0.8), label, "time_scale_0.8"),
            (_time_scale(seq, 1.2), label, "time_scale_1.2"),
            (_drop_frames_and_impute(seq, rng), label, "frame_dropout_imputed"),
            (_feature_noise(seq, rng), label, "feature_noise"),
            (_visibility_loss_proxy(seq, rng), label, "visibility_loss_imputed"),
            (_camera_distance_variant(seq, rng), label, "camera_distance"),
        ]
        if label in ATTACK_LABELS:
            variants.extend([(_prefix(seq, 0.65), label, "action_prefix"), (_prefix(seq, 0.82), label, "action_prefix")])
        for variant, variant_label, name in variants:
            augmented_sequences.append(variant)
            augmented_labels.append(variant_label)
            counts[name] += 1
    return augmented_sequences, augmented_labels, counts


def _training_set(sequences, labels, augmentation, seed):
    """Keep validation/test untouched; choose augmentation only for a train split."""
    if augmentation == "all":
        return _augment_all(sequences, labels, seed)
    return list(sequences), list(labels), {"original": len(labels)}


def _confusion(y_true, y_pred):
    index = {name: i for i, name in enumerate(CLASSES)}
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[index[truth], index[pred]] += 1
    return matrix


def _macro_f1(y_true, y_pred):
    matrix = _confusion(y_true, y_pred)
    scores = []
    for i in range(len(CLASSES)):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores))


def _objective(y_true, y_pred):
    macro_f1 = _macro_f1(y_true, y_pred)
    safe_count = sum(y in SAFE_LABELS for y in y_true)
    false_attacks = sum(t in SAFE_LABELS and p in ATTACK_LABELS for t, p in zip(y_true, y_pred))
    false_attack_rate = false_attacks / max(safe_count, 1)
    # IDLE/OTHER의 공격 오검출을 일반 클래스 오류보다 더 비싸게 보되,
    # Macro-F1 자체를 희생해 안전 클래스만 예측하는 해를 피한다.
    return macro_f1 - 0.15 * false_attack_rate


def optimize(summaries, labels, trials, seed):
    rng = np.random.default_rng(seed)
    best = dict(DEFAULT_OPTIMIZED_THRESHOLDS)
    best_pred = _predict(summaries, best)
    best_score = _objective(labels, best_pred)

    for _ in range(trials):
        candidate = {"lookback_frames": 8}
        for name, (low, high) in RANGES.items():
            candidate[name] = float(rng.uniform(low, high))
        pred = _predict(summaries, candidate)
        score = _objective(labels, pred)
        if score > best_score:
            best, best_score = candidate, score

    # 무작위 최적점 주변에서 좁은 범위 좌표 탐색으로 마무리한다.
    for _ in range(3):
        improved = False
        for name, (low, high) in RANGES.items():
            span = (high - low) * 0.04
            for value in np.linspace(max(low, best[name] - span), min(high, best[name] + span), 9):
                candidate = dict(best); candidate[name] = float(value)
                score = _objective(labels, _predict(summaries, candidate))
                if score > best_score:
                    best, best_score, improved = candidate, score, True
        if not improved:
            break
    return best, best_score


def _metrics(y_true, y_pred):
    matrix = _confusion(y_true, y_pred)
    per_class = {}
    f1_values = []
    for i, name in enumerate(CLASSES):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        support = int(matrix[i, :].sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    safe_count = sum(y in SAFE_LABELS for y in y_true)
    false_attacks = sum(t in SAFE_LABELS and p in ATTACK_LABELS for t, p in zip(y_true, y_pred))
    return {
        "accuracy": float(sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)),
        "macro_f1": float(np.mean(f1_values)),
        "safe_false_attack_rate": float(false_attacks / max(safe_count, 1)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-outliers", action="store_true")
    parser.add_argument("--augmentation", choices=("none", "all"), default="all")
    parser.add_argument("--artifact-name", default="heuristic_thresholds")
    parser.add_argument("--deploy-name", default="heuristic_thresholds")
    args = parser.parse_args()

    data = build_raw_dataset()
    if args.keep_outliers:
        keep = np.ones(len(data["labels"]), dtype=bool)
    else:
        keep, _ = detect_outliers(data["summary_vecs"], data["labels"], data["sample_ids"], CLASSES)

    seqs = [seq for seq, use in zip(data["heuristic_seqs"], keep) if use]
    participants = data["participants"][keep]
    labels = [CLASSES[i] for i in data["labels"][keep]]
    summaries = [summarize_heuristic_sequence(seq) for seq in seqs]

    loso_true, loso_baseline, loso_optimized, folds = [], [], [], []
    for fold_index, held_out in enumerate(sorted(set(participants.tolist()))):
        train_idx = [i for i, p in enumerate(participants) if p != held_out]
        test_idx = [i for i, p in enumerate(participants) if p == held_out]
        train_sequences = [seqs[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        train_aug_sequences, train_aug_labels, train_augmentation = _training_set(
            train_sequences, train_labels, args.augmentation, args.seed + fold_index
        )
        train_s = [summarize_heuristic_sequence(seq) for seq in train_aug_sequences]
        train_y = train_aug_labels
        fold_thresholds, _ = optimize(train_s, train_y, args.trials, args.seed + fold_index)
        test_s = [summaries[i] for i in test_idx]; test_y = [labels[i] for i in test_idx]
        opt_pred = _predict(test_s, fold_thresholds)
        base_pred = [classify_heuristic_sequence(seqs[i]) for i in test_idx]
        loso_true.extend(test_y); loso_baseline.extend(base_pred); loso_optimized.extend(opt_pred)
        folds.append({
            "held_out_participant": str(held_out), "n_train": len(train_idx),
            "n_train_augmented": len(train_aug_labels), "n_test": len(test_idx),
            "augmentation_counts": train_augmentation,
            "baseline_macro_f1": _metrics(test_y, base_pred)["macro_f1"],
            "optimized_macro_f1": _metrics(test_y, opt_pred)["macro_f1"],
            "optimized_safe_false_attack_rate": _metrics(test_y, opt_pred)["safe_false_attack_rate"],
            "thresholds": fold_thresholds,
        })

    augmented_sequences, augmented_labels, augmentation_counts = _training_set(
        seqs, labels, args.augmentation, args.seed + 100
    )
    augmented_summaries = [summarize_heuristic_sequence(seq) for seq in augmented_sequences]
    final_thresholds, final_objective = optimize(augmented_summaries, augmented_labels, args.trials * 2, args.seed + 100)
    generated_at = datetime.now(timezone.utc).isoformat()
    threshold_doc = {
        "schema_version": 1,
        "generated_at": generated_at,
        "feature_set": "heuristic_7j_v1",
        "sequence_length": 60,
        "classifier": "ordered_thresholds_v1",
        "classes": CLASSES,
        "thresholds": final_thresholds,
        "training": {
            "original_samples": len(labels), "augmented_samples": len(augmented_labels),
            "augmentation": {
            "profile": "all_recommended_v1" if args.augmentation == "all" else "none",
                "horizontal_mirror": args.augmentation == "all",
                "left_right_labels_swapped": args.augmentation == "all",
                "time_scale": [0.8, 1.2] if args.augmentation == "all" else [],
                "frame_dropout": {"ratio": 0.08, "imputation": "linear"} if args.augmentation == "all" else None,
                "coordinate_noise": {"space": "derived_feature", "std_fraction": 0.025} if args.augmentation == "all" else None,
                "partial_visibility": {"space": "derived_feature", "method": "short_span_linear_imputation"} if args.augmentation == "all" else None,
                "camera_distance": [0.92, 1.08] if args.augmentation == "all" else [],
                "action_prefix_fractions": [0.65, 0.82] if args.augmentation == "all" else [],
                "counts": augmentation_counts,
            },
            "participants": sorted(set(participants.tolist())), "trials": args.trials, "seed": args.seed,
            "outliers_excluded": int((~keep).sum()),
        },
    }
    metric_doc = {
        "schema_version": 2, "generated_at": generated_at, "classes": CLASSES,
        "n_total": int(len(data["labels"])), "n_used": len(labels), "n_outliers_excluded": int((~keep).sum()),
        "objective": "macro_f1 - 0.15 * safe_false_attack_rate",
        "loso_folds": folds,
        "loso_baseline": _metrics(loso_true, loso_baseline),
        "loso_optimized": _metrics(loso_true, loso_optimized),
        "final_training_objective": float(final_objective),
        "final_augmentation_counts": augmentation_counts,
        "final_thresholds": final_thresholds,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOY_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / f"{args.artifact_name}.json"
    deploy_path = DEPLOY_PATH.parent / f"{args.deploy_name}.json"
    metric_path = ARTIFACT_DIR / f"{args.artifact_name}_metrics.json"
    text_thresholds = json.dumps(threshold_doc, ensure_ascii=False, indent=2)
    artifact_path.write_text(text_thresholds, encoding="utf-8")
    deploy_path.write_text(text_thresholds, encoding="utf-8")
    metric_path.write_text(json.dumps(metric_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"samples={len(labels)} participants={threshold_doc['training']['participants']}")
    print(f"baseline LOSO macro-F1={metric_doc['loso_baseline']['macro_f1']:.4f}")
    print(f"optimized LOSO macro-F1={metric_doc['loso_optimized']['macro_f1']:.4f}")
    print(f"safe false-attack rate={metric_doc['loso_optimized']['safe_false_attack_rate']:.4f}")
    print(f"saved: {artifact_path}")
    print(f"deployed: {deploy_path}")


if __name__ == "__main__":
    main()
