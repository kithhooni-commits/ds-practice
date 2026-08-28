"""
outlier_filter.py — 라벨과 실제 동작이 어긋난 샘플(outlier) 탐지

방식: robust z-score (median/MAD 기반), 클래스 내부에서만 비교.
  1) summary_vec(68차원: heuristic_7j_v1 의 mean/std/max/min)를 전체 샘플 기준으로
     median/MAD 로 스케일링한다 (feature 스케일 차이를 없애기 위함).
  2) 같은 라벨 안에서 "클래스 중심(median)까지의 거리"를 구한다.
     → 예를 들어 LEFT_JAB 라벨인데 실제로는 훅처럼 움직였다면, 이 벡터가
       LEFT_JAB 클래스 중심에서 멀리 떨어져 있을 것이다.
  3) 그 거리 자체를 다시 클래스 내부에서 robust z-score 로 변환해 임계값을 넘으면 outlier.

n=20/class 로 표본이 적어 통계적으로 아주 정교하진 않지만, "명백히 라벨과 다른 동작"을
걸러내는 데는 median/MAD 기반이 평균/표준편차보다 이상치에 안전하다.
"""
import numpy as np

EPS = 1e-6
Z_THRESHOLD = 3.0          # 이 값을 넘으면 outlier로 판정
MAX_DROP_RATIO_PER_CLASS = 0.25  # 한 클래스에서 25% 넘게 잘라내지 않도록 안전장치


def _robust_scale(X):
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = 1.4826 * mad + EPS  # 1.4826: MAD -> std 근사 상수
    return (X - med) / scale


def detect_outliers(summary_vecs, labels, sample_ids, classes, z_threshold=Z_THRESHOLD):
    """
    Returns:
      keep_mask : (N,) bool, True면 학습에 사용
      report    : list of dict, 클래스별 거리·판정 내역 (score 내림차순)
    """
    scaled = _robust_scale(summary_vecs)
    n = len(labels)
    keep_mask = np.ones(n, dtype=bool)
    report = []

    for label_idx in np.unique(labels):
        idx = np.where(labels == label_idx)[0]
        class_vecs = scaled[idx]
        centroid = np.median(class_vecs, axis=0)
        dist = np.linalg.norm(class_vecs - centroid, axis=1)

        dist_med = np.median(dist)
        dist_mad = np.median(np.abs(dist - dist_med))
        dist_scale = 1.4826 * dist_mad + EPS
        z = (dist - dist_med) / dist_scale

        order = np.argsort(-z)  # 의심스러운 순서대로
        max_drop = max(0, int(np.floor(len(idx) * MAX_DROP_RATIO_PER_CLASS)))
        dropped_in_class = 0

        for rank, local_i in enumerate(order):
            global_i = idx[local_i]
            is_outlier = bool(z[local_i] > z_threshold) and dropped_in_class < max_drop
            if is_outlier:
                keep_mask[global_i] = False
                dropped_in_class += 1
            report.append({
                "sample_id": str(sample_ids[global_i]),
                "label": classes[label_idx],
                "distance_z": float(z[local_i]),
                "dropped": is_outlier,
            })

    report.sort(key=lambda r: -r["distance_z"])
    return keep_mask, report
