"""
real_data.py — collected_pose/ 실측 샘플 로더

manifest.jsonl 을 읽어 각 샘플의 JSON을 열고, 두 종류 피처를 뽑는다.
  - summary vector : outlier(라벨-동작 불일치) 탐지용, 시간축을 통계량으로 뭉갠 고정길이 벡터
  - sequence       : TCN 학습용, causal(과거만) 피처를 고정 길이로 맞춘 시퀀스

manifest.jsonl 의 quality.status != "accepted" 인 샘플은 애초에 캡처 품질 미달이므로
여기서도 함께 걸러낸다 (라벨 불일치 outlier와는 다른 축의 필터).
"""
import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "collected_pose")
MANIFEST_PATH = os.path.join(DATA_ROOT, "manifest.jsonl")

CLASSES = [
    "IDLE", "LEFT_JAB", "RIGHT_JAB", "LEFT_HOOK", "RIGHT_HOOK",
    "LEFT_UPPERCUT", "RIGHT_UPPERCUT", "TWO_HAND_GUARD", "ENERGY_WAVE", "OTHER",
]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

SEQ_FEATURE_KEY = "heuristic_7j_v1"       # TCN 실시간 입력 (17차원)
SUMMARY_FEATURE_KEY = "heuristic_7j_v1"   # outlier 탐지 + rule baseline 입력도 동일
TARGET_LEN = 60  # 실측 샘플 frame_count 최대치(60)에 맞춰 causal left-pad
SEQ_FEATURE_DIM = 17

# 왜 game_7j_temporal_v2(70차원, shoulder-normalized 절대좌표+속도+가속도)가 아니라
# heuristic_7j_v1(17차원, 각도비/거리/속도 등 전부 상대값)을 쓰는가:
# 실측 확인 결과 game_7j_temporal_v2의 관절 "위치" 채널은 참가자별 카메라 거리·프레임 구도가
# 그대로 남아 있어(같은 사람 안에서는 표준편차가 아주 작고, 사람 사이에서는 크게 벌어짐),
# 모델이 동작이 아니라 "이 사람이 카메라에서 얼마나 떨어져 있는지"로 참가자를 맞히는
# shortcut을 학습해버린다 — LOSO(새 참가자) 검증에서 rule-base보다도 낮은 정확도로 드러났다.
# heuristic_7j_v1은 애초에 각도비·거리비·속도처럼 전부 "관절 간 상대값"이라 이 문제가 없다.


def load_manifest():
    entries = []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def load_sample(entry):
    path = os.path.join(DATA_ROOT, entry["path"])
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _left_pad_causal(arr, target_len):
    """앞쪽(과거)을 첫 프레임 반복으로 채운다. 뒤쪽(현재/행동 구간)의 시간 정렬을 보존하기 위함."""
    t = arr.shape[0]
    if t >= target_len:
        return arr[-target_len:]
    pad = np.repeat(arr[:1], target_len - t, axis=0)
    return np.concatenate([pad, arr], axis=0)


def sequence_of(sample_json, feature_key=SEQ_FEATURE_KEY, target_len=TARGET_LEN):
    feats = np.asarray(sample_json["features"][feature_key], dtype=np.float32)
    return _left_pad_causal(feats, target_len)


def summary_vector_of(sample_json, feature_key=SUMMARY_FEATURE_KEY):
    """시간축 mean/std/max/min 으로 뭉갠 고정 길이 벡터. outlier 탐지 + rule baseline 계산에 쓴다."""
    feats = np.asarray(sample_json["features"][feature_key], dtype=np.float32)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    mx = feats.max(axis=0)
    mn = feats.min(axis=0)
    return np.concatenate([mean, std, mx, mn]), feats  # (68,), (T,17)


def build_raw_dataset():
    """
    필터링 전 원본 데이터셋을 만든다.
    Returns: dict with keys
      entries        : manifest 항목 리스트 (accepted 캡처만)
      labels         : (N,) int
      participants   : (N,) str
      sequences      : (N, TARGET_LEN, 70) float32  — TCN 입력
      summary_vecs   : (N, 68) float32               — outlier 탐지 입력
      heuristic_seqs : list of (T,17) float32         — rule baseline 입력 (프레임 수가 제각각이라 리스트)
      sample_ids     : (N,) str
    """
    entries = load_manifest()
    labels, participants, sequences, summary_vecs, heuristic_seqs, sample_ids = [], [], [], [], [], []
    kept_entries = []

    for entry in entries:
        if entry.get("quality", {}).get("status") != "accepted":
            continue  # 캡처 품질 자체가 미달 (라벨 불일치와는 다른 축)
        sample = load_sample(entry)
        label = entry["label"]
        if label not in LABEL_TO_IDX:
            continue

        seq = sequence_of(sample)
        summary_vec, heur_seq = summary_vector_of(sample)

        kept_entries.append(entry)
        labels.append(LABEL_TO_IDX[label])
        participants.append(entry["participant_id"])
        sequences.append(seq)
        summary_vecs.append(summary_vec)
        heuristic_seqs.append(heur_seq)
        sample_ids.append(entry["sample_id"])

    return {
        "entries": kept_entries,
        "labels": np.array(labels, dtype=np.int64),
        "participants": np.array(participants),
        "sequences": np.stack(sequences).astype(np.float32),
        "summary_vecs": np.stack(summary_vecs).astype(np.float32),
        "heuristic_seqs": heuristic_seqs,
        "sample_ids": np.array(sample_ids),
    }
