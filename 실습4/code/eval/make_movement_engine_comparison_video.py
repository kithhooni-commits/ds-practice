"""
make_movement_engine_comparison_video.py — 펀치 비교 영상(make_comparison_videos.py)과
동일한 포맷의 이동/회전 비교 영상.

  [원본 웹캠] | [V1 BASELINE 스켈레톤+판정] | [V2 IMPROVED 스켈레톤+판정]

펀치 쪽은 [원본|RULE-BASE|TCN]으로 "같은 입력, 다른 판정기"를 비교했다. 이동/회전엔
TCN이 없으므로(학습 데이터가 없음) 자연스러운 대응은 [원본|v1 룰베이스|v2 개선 룰베이스] —
"같은 입력, 다른 튜닝"이다. 스켈레톤은 evaluate_full_actions.py 가 report.json 에 저장해 둔
lm7(7노드 정규화 좌표)로 그리므로, mediapipe 를 다시 돌리지 않는다.

Usage:
  python eval/make_movement_engine_comparison_video.py \
    --video eval/video/benchmark_movement_fixed.mp4 \
    --report-v1 eval/output/movement_report_v1.json \
    --report-v2 eval/output/movement_report_v2.json \
    --labels eval/video/benchmark_movement_labels.json \
    --out eval/output/movement_engine_comparison.mp4
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from score_movement import build_gt_lookup  # noqa: E402

# evaluate_full_actions.py 의 NODE_IDS 순서와 동일: NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR
NAMES = ["nose", "l_sh", "r_sh", "l_el", "r_el", "l_wr", "r_wr"]
BONES = [("l_sh", "r_sh"), ("l_sh", "l_el"), ("l_el", "l_wr"),
         ("r_sh", "r_el"), ("r_el", "r_wr"), ("nose", "l_sh"), ("nose", "r_sh")]


def draw_skeleton_panel(w, h, lm7, title, move_pred, rot_pred, move_ok, rot_ok):
    """make_comparison_videos.py의 draw_skeleton_frame()과 동일한 스타일 —
    검은 배경 + 노란 뼈대 + 흰 관절점 + 좌상단 제목 + 하단 판정 텍스트(정답이면 초록/틀리면 빨강)."""
    img = np.full((h, w, 3), (18, 18, 22), dtype=np.uint8)
    if lm7:
        pts = {name: (int(x * w), int(y * h)) for name, (x, y) in zip(NAMES, lm7)}
        for a, b in BONES:
            cv2.line(img, pts[a], pts[b], (0, 210, 255), 4, cv2.LINE_AA)
        for p in pts.values():
            cv2.circle(img, p, 7, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.putText(img, title, (14, 34), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 210, 255), 2, cv2.LINE_AA)

    move_col = (80, 230, 90) if move_ok else (60, 80, 240)
    rot_col = (80, 230, 90) if rot_ok else (60, 80, 240)
    cv2.putText(img, f"MOVE: {move_pred}", (14, h - 48), cv2.FONT_HERSHEY_DUPLEX, 0.75, move_col, 2, cv2.LINE_AA)
    cv2.putText(img, f"ROT : {rot_pred}", (14, h - 16), cv2.FONT_HERSHEY_DUPLEX, 0.75, rot_col, 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="Build [original | v1 | v2] movement/rotation comparison video (punch-comparison format)")
    ap.add_argument("--video", required=True, help="원본 웹캠 영상 (annotate 안 된 원본)")
    ap.add_argument("--report-v1", required=True)
    ap.add_argument("--report-v2", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tl_v1 = json.loads(Path(args.report_v1).read_text(encoding="utf-8"))["state_timeline"]
    tl_v2 = json.loads(Path(args.report_v2).read_text(encoding="utf-8"))["state_timeline"]
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    move_gt = build_gt_lookup(labels.get("move_segments", []))
    rot_gt = build_gt_lookup(labels.get("rot_segments", []))

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out) if args.out else Path(args.video).parent / "movement_engine_comparison.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w * 3, h))

    n = min(len(tl_v1), len(tl_v2), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(f"프레임 {n}개 처리 중...")
    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        x1, x2 = tl_v1[i], tl_v2[i]
        t_ms = x1["t_ms"]
        gt_move, gt_rot = move_gt(t_ms), rot_gt(t_ms)

        panel_v1 = draw_skeleton_panel(w, h, x1.get("lm7"), "V1 BASELINE",
                                        x1["move"], x1["rot"],
                                        x1["move"] == gt_move, x1["rot"] == gt_rot)
        panel_v2 = draw_skeleton_panel(w, h, x2.get("lm7"), "V2 IMPROVED",
                                        x2["move"], x2["rot"],
                                        x2["move"] == gt_move, x2["rot"] == gt_rot)

        banner = f"GT move:{gt_move} rot:{gt_rot}   t={t_ms/1000:.1f}s"
        cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(frame, banner, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        combined = np.hstack([frame, panel_v1, panel_v2])
        writer.write(combined)
        if i % 500 == 0:
            print(f"  {i}/{n} ({t_ms/1000:.1f}s)")

    cap.release()
    writer.release()
    print(f"완료: {out_path.resolve()}")

    import subprocess
    import shutil
    ffmpeg_cmd = shutil.which("ffmpeg")
    if ffmpeg_cmd:
        h264_tmp = out_path.with_name(out_path.stem + "_h264.mp4")
        res = subprocess.run(
            [ffmpeg_cmd, "-y", "-i", str(out_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", str(h264_tmp)],
            capture_output=True, text=True)
        if res.returncode == 0 and h264_tmp.exists():
            h264_tmp.replace(out_path)
            print("H.264 변환 완료")


if __name__ == "__main__":
    main()
