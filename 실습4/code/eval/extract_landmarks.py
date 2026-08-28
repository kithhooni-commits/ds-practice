"""Extract MediaPipe Pose landmarks from a video into a cached JSONL file.

The cache decouples slow pose inference from fast detection-logic iteration:
when the punch pipeline changes, re-run the evaluator on the same JSONL
instead of re-processing the video.

Usage:
  python iter4/eval/extract_landmarks.py video/foo.mp4 -o datasets/foo/landmarks.jsonl
"""
import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

DEFAULT_MODEL = Path(__file__).with_name("models") / "pose_landmarker_full.task"


def landmark_rows(points, with_visibility):
    if points is None:
        return None
    rows = []
    for p in points:
        row = [round(p.x, 5), round(p.y, 5), round(p.z, 5)]
        if with_visibility:
            row.append(round(getattr(p, "visibility", 1.0), 4))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Extract pose landmarks from a video to JSONL")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"모델 파일 없음: {model_path}")
        raise SystemExit(1)
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"영상 파일 없음: {video_path}")
        raise SystemExit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"영상을 열 수 없음: {video_path}")
        raise SystemExit(1)

    written = 0
    detected = 0
    with out_path.open("w", encoding="utf-8") as out, vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, ts_ms)
            norm = result.pose_landmarks[0] if result.pose_landmarks else None
            world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
            rec = {
                "t_ms": ts_ms,
                "lm": landmark_rows(norm, True),
                "wl": landmark_rows(world, False),
            }
            out.write(json.dumps(rec) + "\n")
            written += 1
            if norm is not None and world is not None and len(world) > 16:
                detected += 1
    cap.release()

    coverage = round(100.0 * detected / written, 1) if written else 0.0
    print(f"{written}프레임 기록 → {out_path} (world 검출 커버리지 {coverage}%)")


if __name__ == "__main__":
    main()
