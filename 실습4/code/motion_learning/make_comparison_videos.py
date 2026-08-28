"""
make_comparison_videos.py — 3분할 비교 영상 생성
  [원본 sample webm] | [Rule-base 스켈레톤+실시간 판정] | [TCN 스켈레톤+실시간 판정]

선정 기준:
  - suhwan 세션에서 outlier_filter로 걸러지지 않은(라벨-동작 불일치 아닌) 샘플만
  - 50%는 "rule-base가 틀렸지만 TCN은 맞춘" 사례, 50%는 "둘 다 맞춘" 사례로 균형있게 구성

주의(정직하게 밝혀둠): 여기서 쓰는 TCN은 배포용 최종 모델(boxing_tcn.pth) — 4명 데이터 전부로
학습했기 때문에 suhwan도 학습에 포함되어 있다. 즉 이 영상은 "실제로 어떻게 동작하는지 보여주는
데모"이지 rule-base 33.3% vs TCN 49.2%라는 논문급 수치의 재현이 아니다(그건 LOSO로 참가자를
아예 안 보여주고 낸 값). 데모와 엄밀한 성능 수치를 섞어 보면 안 된다.
"""
import os
import json
import subprocess
import numpy as np
import cv2
import torch

from real_data import CLASSES, load_manifest, load_sample, DATA_ROOT
from rule_baseline import classify_heuristic_sequence
from tcn_model import CausalMotionTCN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "comparison_videos")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 640, 480
OUT_FPS = 20
SEQ_LEN = 60

# raw joint_set: nose, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow,
#                left_wrist, right_wrist, ... (손가락은 스켈레톤에서 생략)
RAW_IDX = {"nose": 0, "l_sh": 3, "r_sh": 4, "l_el": 5, "r_el": 6, "l_wr": 7, "r_wr": 8}
BONES = [("l_sh", "r_sh"), ("l_sh", "l_el"), ("l_el", "l_wr"), ("r_sh", "r_el"), ("r_el", "r_wr"),
         ("nose", "l_sh"), ("nose", "r_sh")]

# (sample_id, 카테고리) — 50:50 구성. 전부 outlier_filter를 통과한(라벨-동작 불일치 아닌) 샘플.
SELECTED = [
    ("20260826T043549220745Z_2e598e0b", "both_correct"),      # RIGHT_UPPERCUT
    ("20260826T043622247575Z_beeb26a3", "both_correct"),      # TWO_HAND_GUARD
    ("20260826T043806453222Z_73333695", "both_correct"),      # RIGHT_HOOK
    ("20260826T043530588158Z_895d82ee", "both_correct"),      # LEFT_JAB
    ("20260826T043544623078Z_f3d89837", "rule_wrong_tcn_right"),  # LEFT_HOOK (rule→LEFT_JAB 오판)
    ("20260826T043723632831Z_dcba6980", "rule_wrong_tcn_right"),  # RIGHT_UPPERCUT (rule→RIGHT_JAB 오판)
    ("20260826T043540044352Z_69aee6f6", "rule_wrong_tcn_right"),  # ENERGY_WAVE (rule은 이 클래스 자체가 없음)
    ("20260826T043747471642Z_4e273671", "rule_wrong_tcn_right"),  # TWO_HAND_GUARD (rule→IDLE 오판)
]


def left_pad(arr, n=SEQ_LEN):
    t = arr.shape[0]
    if t >= n:
        return arr[-n:]
    pad = np.repeat(arr[:1], n - t, axis=0)
    return np.concatenate([pad, arr], axis=0)


def load_tcn():
    model = CausalMotionTCN()
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "boxing_tcn.pth"), map_location="cpu"))
    model.eval()
    scaler = json.load(open(os.path.join(BASE_DIR, "boxing_tcn_scaler.json")))
    return model, np.array(scaler["median"]), np.array(scaler["scale"]), scaler["clip"]


def video_duration_sec(path):
    r = subprocess.run(["ffmpeg", "-i", path, "-f", "null", "-"], capture_output=True, text=True)
    for line in reversed(r.stderr.splitlines()):
        if "time=" in line:
            t = line.split("time=")[1].split(" ")[0]
            h, m, s = t.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return None


def draw_skeleton_frame(landmarks_2d, title, pred_text, pred_ok):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (18, 18, 22)
    pts = {}
    for name, idx in RAW_IDX.items():
        lm = landmarks_2d[idx]
        pts[name] = (int(lm[0] * W), int(lm[1] * H))
    for a, b in BONES:
        cv2.line(img, pts[a], pts[b], (0, 210, 255), 4, cv2.LINE_AA)
    for name, p in pts.items():
        cv2.circle(img, p, 7, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.putText(img, title, (14, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 210, 255), 2, cv2.LINE_AA)
    color = (80, 230, 90) if pred_ok else (60, 80, 240)
    cv2.putText(img, pred_text, (14, H - 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)
    return img


def build_sample_video(sample_id, entry, sample, model, median, scale, clip):
    true_label = entry["label"]
    heur = np.asarray(sample["features"]["heuristic_7j_v1"], dtype=np.float32)
    raw_frames = sample["raw_frames"]
    rel_ms = np.array([f["relative_time_ms"] for f in raw_frames], dtype=np.float32)
    total_ms = rel_ms[-1]

    video_path = os.path.join(DATA_ROOT, entry["video_path"])
    dur = video_duration_sec(video_path)
    if not dur:
        dur = total_ms / 1000.0
    n_out = max(1, int(round(dur * OUT_FPS)))

    rule_writer_path = os.path.join(OUT_DIR, f"{sample_id}_rule.mp4")
    tcn_writer_path = os.path.join(OUT_DIR, f"{sample_id}_tcn.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rw = cv2.VideoWriter(rule_writer_path, fourcc, OUT_FPS, (W, H))
    tw = cv2.VideoWriter(tcn_writer_path, fourcc, OUT_FPS, (W, H))

    for i in range(n_out):
        if i == n_out - 1:
            # 마지막 출력 프레임은 반드시 실제 마지막 포즈 프레임을 가리키게 한다 — 반올림 탓에
            # 동작이 "완성되는" 바로 그 프레임(예: 훅이 다 돌아간 순간) 하나 직전에서
            # 잘리는 걸 막기 위함이다.
            idx = len(raw_frames) - 1
        else:
            t_ms = (i / OUT_FPS) * 1000.0 * (total_ms / (dur * 1000.0))  # 영상 진행률 -> 포즈 타임라인 매핑
            idx = int(np.searchsorted(rel_ms, t_ms, side="right") - 1)
            idx = max(0, min(idx, len(raw_frames) - 1))

        landmarks_2d = [(lm["x"], lm["y"]) for lm in raw_frames[idx]["landmarks"]]

        rule_pred = classify_heuristic_sequence(heur[: idx + 1])
        rule_ok = rule_pred == true_label
        rw.write(draw_skeleton_frame(landmarks_2d, "RULE-BASE", f"{rule_pred}", rule_ok))

        seq = left_pad(heur[: idx + 1].astype(np.float32))
        seq_s = np.clip((seq - median) / scale, -clip, clip).astype(np.float32)
        with torch.no_grad():
            probs = torch.softmax(model(torch.tensor(seq_s).unsqueeze(0)), dim=1)[0].numpy()
        tcn_pred = CLASSES[int(np.argmax(probs))]
        tcn_conf = float(probs.max())
        tcn_ok = tcn_pred == true_label
        tw.write(draw_skeleton_frame(landmarks_2d, "TCN", f"{tcn_pred} {tcn_conf*100:.0f}%", tcn_ok))

    rw.release()
    tw.release()
    return rule_writer_path, tcn_writer_path, video_path, dur


def hstack_with_labels(orig_path, rule_path, tcn_path, out_path, true_label, category, dur):
    """세 영상을 가로로 이어붙이고, 상단에 참라벨/카테고리 배너를 넣는다."""
    banner = f"TRUE: {true_label}   ({'rule wrong -> TCN correct' if category == 'rule_wrong_tcn_right' else 'both correct'})"
    banner = banner.replace(":", "\\:")  # ffmpeg drawtext escape
    filter_complex = (
        f"[0:v]scale={W}:{H},setsar=1,fps={OUT_FPS}[v0];"
        f"[1:v]scale={W}:{H},setsar=1,fps={OUT_FPS}[v1];"
        f"[2:v]scale={W}:{H},setsar=1,fps={OUT_FPS}[v2];"
        f"[v0][v1][v2]hstack=inputs=3[stacked];"
        f"[stacked]drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='{banner}':"
        f"fontcolor=white:fontsize=20:x=(w-text_w)/2:y=8:box=1:boxcolor=black@0.6:boxborderw=6[out]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", orig_path, "-i", rule_path, "-i", tcn_path,
        "-filter_complex", filter_complex,
        "-map", "[out]", "-t", str(dur),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    manifest = {e["sample_id"]: e for e in load_manifest()}
    model, median, scale, clip = load_tcn()

    combined_paths = []
    for sample_id, category in SELECTED:
        entry = manifest[sample_id]
        sample = load_sample(entry)
        print(f"[*] {sample_id}  {entry['label']:16s}  ({category})")
        rule_path, tcn_path, orig_path, dur = build_sample_video(sample_id, entry, sample, model, median, scale, clip)
        combined_path = os.path.join(OUT_DIR, f"{sample_id}_combined.mp4")
        hstack_with_labels(orig_path, rule_path, tcn_path, combined_path, entry["label"], category, dur)
        combined_paths.append(combined_path)
        os.remove(rule_path)
        os.remove(tcn_path)
        print(f"    -> {combined_path}")

    # 전부 이어붙여 하나의 비교 영상으로
    concat_list = os.path.join(OUT_DIR, "concat_list.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in combined_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    final_path = os.path.join(OUT_DIR, "comparison_all.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", final_path],
        check=True, capture_output=True,
    )
    print(f"\n[OK] 최종 비교 영상: {final_path}")


if __name__ == "__main__":
    main()
