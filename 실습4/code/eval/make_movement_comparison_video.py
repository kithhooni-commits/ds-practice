"""
make_movement_comparison_video.py — [사용자 웹캠 + 스켈레톤] | [간이 아바타 반응] 좌우 비교 영상

motion_learning/make_comparison_videos.py(펀치용 3분할)의 이동/회전판. 다만 이동/회전은
Three.js 휴머노이드를 오프라인으로 렌더링할 수 없으므로(브라우저 전용), 대신 OpenCV로
그린 간단한 스틱 피겨가 "룰베이스가 방금 판정한 상태"에 실시간으로 반응하게 그린다.

입력:
  --annotated  evaluate_full_actions.py 의 --annotate 출력 (스켈레톤+HUD 오버레이된 영상)
  --report     같은 실행의 --report 출력 (state_timeline 필요 — annotated와 프레임 순서가 1:1 대응)
  --labels     GT 라벨 (score_movement.py와 동일 스키마)

출력: [annotated 원본] | [아바타 패널(예측 상태로 반응) + GT/예측 텍스트, 일치 여부 색상]
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from score_movement import build_gt_lookup, MOVE_CLASSES, ROT_CLASSES  # noqa: E402


def draw_avatar(w, h, move, rot, move_ok, rot_ok):
    img = np.full((h, w, 3), (24, 24, 28), dtype=np.uint8)
    cx, cy = w // 2, h // 2 - 20

    scale = 1.0
    shift_y = 0
    tilt_deg = 0.0
    if move == "FORWARD":
        scale, shift_y = 1.18, 26
    elif move == "BACK":
        scale, shift_y = 0.84, -18
    elif move == "LEFT":
        tilt_deg = -14.0
    elif move == "RIGHT":
        tilt_deg = 14.0

    # --- 기본 스틱 피겨 좌표 (중립 가드 자세) ---
    head_r = int(26 * scale)
    sh_y = cy - 10 + shift_y
    sh_half = int(60 * scale)
    hip_y = cy + 150 * scale + shift_y
    pts = {
        "head": (cx, sh_y - head_r - 14),
        "l_sh": (cx - sh_half, sh_y),
        "r_sh": (cx + sh_half, sh_y),
        "l_el": (cx - sh_half - 18, sh_y + int(55 * scale)),
        "r_el": (cx + sh_half + 18, sh_y + int(55 * scale)),
        "l_wr": (cx - int(20 * scale), sh_y + int(30 * scale)),   # 가드 위치 (턱 옆)
        "r_wr": (cx + int(20 * scale), sh_y + int(30 * scale)),
        "hip": (cx, int(hip_y)),
    }

    # 좌/우 스텝: 어깨선(및 팔) 기울기 회전 — 힙은 고정 축으로 둔다
    if tilt_deg != 0.0:
        rad = math.radians(tilt_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        pivot = pts["hip"]
        rotated = {}
        for k, (x, y) in pts.items():
            if k == "hip":
                rotated[k] = (x, y)
                continue
            dx, dy = x - pivot[0], y - pivot[1]
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            rotated[k] = (int(pivot[0] + rx), int(pivot[1] + ry))
        pts = rotated

    bones = [("l_sh", "r_sh"), ("l_sh", "l_el"), ("l_el", "l_wr"),
             ("r_sh", "r_el"), ("r_el", "r_wr"), ("l_sh", "hip"), ("r_sh", "hip")]
    body_color = (0, 210, 255)
    for a, b in bones:
        cv2.line(img, pts[a], pts[b], body_color, 5, cv2.LINE_AA)
    cv2.circle(img, pts["head"], head_r, (255, 255, 255), -1, cv2.LINE_AA)
    for k in ("l_sh", "r_sh", "l_el", "r_el", "l_wr", "r_wr", "hip"):
        cv2.circle(img, pts[k], 6, (255, 255, 255), -1, cv2.LINE_AA)

    # 회전(ROT) 화살표 아이콘 — 상단에 방향 표시 (좌우 이동으로는 요(yaw) 회전을 표현할 수 없어
    # 별도 아이콘으로 대체한다)
    if rot in ("ROT_LEFT", "ROT_RIGHT"):
        arrow_center = (cx, pts["head"][1] - head_r - 30)
        radius = 24
        color = (0, 255, 255)
        start, end = (200, 340) if rot == "ROT_LEFT" else (-20, 160)
        cv2.ellipse(img, arrow_center, (radius, radius), 0, start, end, color, 3, cv2.LINE_AA)
        tip_angle = math.radians(end if rot == "ROT_LEFT" else start)
        tip = (int(arrow_center[0] + radius * math.cos(tip_angle)),
               int(arrow_center[1] + radius * math.sin(tip_angle)))
        cv2.circle(img, tip, 5, color, -1, cv2.LINE_AA)
        label = "<< ROT_LEFT" if rot == "ROT_LEFT" else "ROT_RIGHT >>"
        cv2.putText(img, label, (cx - 70, arrow_center[1] - radius - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # 패널 테두리 — 예측이 정답과 맞으면 초록, 틀리면 빨강
    border_color = (60, 220, 60) if (move_ok and rot_ok) else (50, 50, 230)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), border_color, 6)
    return img


def main():
    ap = argparse.ArgumentParser(description="Build side-by-side [user footage] | [reacting avatar] comparison video")
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    timeline = report["state_timeline"]

    move_gt = build_gt_lookup(labels.get("move_segments", []))
    rot_gt = build_gt_lookup(labels.get("rot_segments", []))

    cap = cv2.VideoCapture(args.annotated)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out) if args.out else Path(args.annotated).parent / "movement_comparison.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w * 2, h))

    n = min(len(timeline), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(f"프레임 {n}개 처리 중...")
    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        t_ms = timeline[i]["t_ms"]
        pred_move, pred_rot = timeline[i]["move"], timeline[i]["rot"]
        gt_move, gt_rot = move_gt(t_ms), rot_gt(t_ms)
        move_ok, rot_ok = (pred_move == gt_move), (pred_rot == gt_rot)

        avatar = draw_avatar(w, h, pred_move, pred_rot, move_ok, rot_ok)

        move_col = (80, 230, 90) if move_ok else (60, 80, 240)
        rot_col = (80, 230, 90) if rot_ok else (60, 80, 240)
        cv2.putText(avatar, f"PRED move: {pred_move}", (16, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, move_col, 2, cv2.LINE_AA)
        cv2.putText(avatar, f"GT   move: {gt_move}", (16, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(avatar, f"PRED rot : {pred_rot}", (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rot_col, 2, cv2.LINE_AA)
        cv2.putText(avatar, f"GT: {gt_move}/{gt_rot}   t={t_ms/1000:.1f}s", (w - 260, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        combined = np.hstack([frame, avatar])
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
