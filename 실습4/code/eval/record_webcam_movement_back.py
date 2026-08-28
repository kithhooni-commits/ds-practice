"""BACK 세그먼트만 다시 찍는 짧은 레코더.

전체 155초를 다시 찍을 필요 없이, benchmark_movement.mp4에서 실패했던 BACK(후진)
구간만 재촬영한다. 원래 세션에서 BACK이 완전히 실패한 원인은 알고리즘이 아니라
**의자에 깊이 기대 앉아 물리적으로 젖힐 공간이 없었던 것**으로 확인됐다
(annotated 영상에서 BACK 구간 Pitch가 오히려 +0.06으로 찍힘). 그래서 이번엔
**의자 앞쪽에 걸터앉아서** 뒤로 젖힐 여지를 확보하고 찍는다.

record_webcam_movement.py와 동일한 설계(같은 open_camera(), 같은 저장-비반전/미리보기-반전
원칙, 같은 자동 라벨 생성)를 따르되 스케줄만 BACK 전용으로 축소했다.

Usage:
  conda activate pjt-4
  python eval/record_webcam_movement_back.py
"""
import json
import sys
import time
from pathlib import Path
import cv2

OUTPUT_DIR = Path(__file__).parent / "video"
OUTPUT_FILE = OUTPUT_DIR / "benchmark_movement_back.mp4"
LABELS_FILE = OUTPUT_DIR / "benchmark_movement_back_labels.json"

# 1차 재촬영 결과: BACK 구간 Pitch가 세 구간 전부 -0.01~+0.12 (평균 +0.05, 오히려 전진 쪽)로
# 한 번도 마이너스로 안 내려갔다. 의자 자세는 고쳤는데도 실패 — 카메라를 계속 보려고
# 무의식적으로 고개를 숙이는 보정 동작이 "얼굴이 어깨선 대비 내려감" 신호(전진 신호)를
# 만들어 후진 신호를 상쇄한 것으로 보인다. 그래서 이번엔 "고개는 어깨와 통짜로, 숙이지 않기"를
# 지시문에 명시적으로 넣는다.
COACH_CUE = "고개를 숙이지 마세요! 어깨~머리가 통짜인 것처럼 몸 전체를 젖히세요"

# (start_s, end_s, title, desc, is_action, gt)
SCHEDULE = [
    (0, 6, "0. 캘리브레이션", "의자 앞쪽에 걸터앉아 기립 자세로 정지 (양 어깨 수평)", False, None),
    (6, 11, "1. BACK (크게)", f"뒤로 최대한 젖히기 — {COACH_CUE}", True, ("move", "BACK")),
    (11, 16, "⏸️ 중립", "기립 자세로 천천히 복귀 (고개는 계속 정면)", False, None),
    (16, 21, "2. BACK (살짝)", f"살짝만 뒤로 젖히기 — {COACH_CUE}", True, ("move", "BACK")),
    (21, 26, "⏸️ 중립", "기립 자세로 천천히 복귀", False, None),
    (26, 31, "3. BACK (크게, 반복)", f"다시 한 번 크게 젖히기 — {COACH_CUE}", True, ("move", "BACK")),
    (31, 36, "4. 마무리", "기립 자세로 정지 후 녹화 종료", False, None),
]
TOTAL_RECORD_SEC = SCHEDULE[-1][1]
PRE_COUNTDOWN_SEC = 5


def get_phase_info(elapsed_sec):
    for start, end, title, desc, is_action, gt in SCHEDULE:
        if start <= elapsed_sec < end:
            return title, desc, is_action, end - elapsed_sec, gt
    return "녹화 완료", "영상을 저장 중입니다...", False, 0.0, None


def build_labels(schedule, onset_tolerance_ms=400, offset_tolerance_ms=300):
    move_segments = []
    for start, end, _title, desc, _is_action, gt in schedule:
        if gt is None:
            continue
        axis, state = gt
        if axis == "move":
            move_segments.append({
                "start_ms": round(start * 1000), "end_ms": round(end * 1000),
                "state": state, "desc": desc,
            })
    return {
        "case": "benchmark_movement_back_v1",
        "source": "eval/video/benchmark_movement_back.mp4",
        "note": "BACK 세그먼트만 재촬영한 짧은 보조 벤치마크. 의자 앞쪽에 걸터앉아 촬영.",
        "onset_tolerance_ms": onset_tolerance_ms,
        "offset_tolerance_ms": offset_tolerance_ms,
        "move_segments": move_segments,
        "rot_segments": [],
        "negative_windows": [],
    }


def draw_text(img, text, pos, font_scale=0.7, color=(255, 255, 255), thickness=2, bg_box=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    if bg_box:
        cv2.rectangle(img, (x - 6, y - h - 6), (x + w + 6, y + baseline + 6), (20, 20, 20), -1)
        cv2.rectangle(img, (x - 6, y - h - 6), (x + w + 6, y + baseline + 6), (80, 80, 80), 1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def open_camera(max_index=3):
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY")]
    for index in range(max_index + 1):
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ 카메라 연결됨 — index={index}, backend={name}")
                return cap
            cap.release()
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print(f"🪑 BACK 전용 재촬영 레코더 ({TOTAL_RECORD_SEC:.0f}초)")
    print("=" * 65)
    print("⚠️  의자 앞쪽 절반 정도에 걸터앉아, 등받이에 닿지 않게 하세요.")
    print("   뒤로 젖힐 때 실제로 상체가 넘어갈 물리적 여유가 있어야 합니다.")
    print(f"• 저장 경로: {OUTPUT_FILE.resolve()}")
    print(f"• 라벨 자동 생성: {LABELS_FILE.resolve()}")
    print("• 조작법: [Q] 키를 누르면 녹화를 조기 종료하고 저장합니다.")
    print("-" * 65)

    cap = open_camera()
    if cap is None:
        print("\n❌ 오류: 웹캠을 열 수 없습니다. (record_webcam_movement.py와 같은 원인일 수 있습니다 —")
        print("   다른 프로그램의 카메라 점유, Windows 카메라 권한, 드라이버 문제를 확인하세요.)")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ret, _ = cap.read()
    if not ret:
        print("⚠️ 1280x720 설정 후 프레임을 못 읽어 기본 해상도로 계속 진행합니다.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 120:
        fps = 30.0
    print(f"• 카메라 해상도: {width} x {height} ({fps:.1f} FPS)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_FILE), fourcc, fps, (width, height))

    # Phase 1: 준비 카운트다운
    t_start_prep = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        elapsed_prep = time.time() - t_start_prep
        remain_prep = PRE_COUNTDOWN_SEC - elapsed_prep
        if remain_prep <= 0:
            break
        display = cv2.flip(frame, 1)
        draw_text(display, "BACK-ONLY RE-RECORDER", (20, 45), font_scale=0.9, color=(0, 255, 255))
        draw_text(display, f"Starting in {int(remain_prep) + 1} seconds...", (20, 90), font_scale=0.8, color=(200, 200, 200))
        draw_text(display, "Sit on the FRONT EDGE of your chair — leave room to lean back!", (20, height - 30), font_scale=0.65, color=(0, 220, 255))
        count_text = str(int(remain_prep) + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (cw, ch), _ = cv2.getTextSize(count_text, font, 3.5, 6)
        cx, cy = (width - cw) // 2, (height + ch) // 2
        cv2.rectangle(display, (cx - 20, cy - ch - 20), (cx + cw + 20, cy + 20), (0, 0, 0), -1)
        cv2.putText(display, count_text, (cx, cy), font, 3.5, (0, 255, 0), 6, cv2.LINE_AA)
        cv2.imshow("Back Re-recorder (Press Q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("녹화가 취소되었습니다.")
            return

    print(f"\n🔴 녹화 시작! {TOTAL_RECORD_SEC:.0f}초간 화면 지시에 맞춰 동작을 수행하세요.")
    t_start_record = time.time()
    frames_recorded = 0
    consecutive_fails = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_fails += 1
            if consecutive_fails > 15:
                print("\n⚠️ 웹캠 신호가 끊겨 녹화를 종료합니다.")
                break
            time.sleep(0.01)
            continue
        consecutive_fails = 0

        now = time.time()
        elapsed = now - t_start_record
        remain = max(0.0, TOTAL_RECORD_SEC - elapsed)
        if elapsed >= TOTAL_RECORD_SEC:
            break

        # 저장은 원본(비반전) 방향 — record_webcam_movement.py와 동일 원칙.
        out.write(frame)
        frames_recorded += 1

        display = cv2.flip(frame, 1).copy()
        title, desc, is_action, phase_remain, gt = get_phase_info(elapsed)
        rec_dot = "REC [LIVE]" if int(elapsed * 2) % 2 == 0 else "REC       "
        draw_text(display, f"{rec_dot}  {elapsed:04.1f}s / {TOTAL_RECORD_SEC:.0f}s (Left: {remain:04.1f}s)", (20, 45), font_scale=0.75, color=(0, 0, 255))
        step_col = (0, 255, 255) if is_action else (0, 200, 255)
        draw_text(display, f"{title} (Left: {phase_remain:04.1f}s)", (20, 85), font_scale=0.8, color=step_col)
        desc_col = (255, 255, 255) if is_action else (100, 255, 100)
        draw_text(display, f"Action: {desc}", (20, 125), font_scale=0.7, color=desc_col)
        if not is_action and 0 < elapsed < TOTAL_RECORD_SEC - 5:
            draw_text(display, f"NEXT IN: {int(phase_remain) + 1}s", (width // 2 - 140, height // 2), font_scale=1.1, color=(0, 255, 255), thickness=3)
        # BACK 액션 중엔 화면 정중앙에 큼직한 자세 경고를 계속 띄운다 — 1차 재촬영에서
        # 이 지시를 놓쳐(무의식적으로 고개를 숙여) 신호가 아예 반대로 나왔기 때문이다.
        if is_action and gt == ("move", "BACK"):
            cv2.rectangle(display, (0, height // 2 - 55), (width, height // 2 + 15), (0, 0, 0), -1)
            draw_text(display, "고개 고정! 어깨~머리를 통짜로 젖히기", (width // 2 - 260, height // 2 - 15),
                      font_scale=0.9, color=(0, 220, 255), thickness=3, bg_box=False)
            draw_text(display, "턱을 당기지 말고, 천장을 본다는 느낌으로 시선까지 같이 넘기세요", (width // 2 - 300, height // 2 + 10),
                      font_scale=0.55, color=(150, 220, 255), thickness=1, bg_box=False)
        draw_text(display, "Press [Q] to finish recording early", (20, height - 25), font_scale=0.6, color=(180, 180, 180))

        cv2.imshow("Back Re-recorder (Press Q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n⏹ 사용자가 [Q] 키를 눌러 녹화를 조기 종료했습니다.")
            break

    actual_record_sec = time.time() - t_start_record
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    import subprocess
    import shutil
    ffmpeg_cmd = shutil.which("ffmpeg")
    if ffmpeg_cmd:
        print("⚙️ 브라우저(HTML5) 호환을 위한 H.264 비디오 최적화 중...")
        h264_tmp = OUTPUT_DIR / "benchmark_movement_back_h264.mp4"
        try:
            res = subprocess.run(
                [ffmpeg_cmd, "-y", "-i", str(OUTPUT_FILE), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", str(h264_tmp)],
                capture_output=True, text=True)
            if res.returncode == 0 and h264_tmp.exists():
                h264_tmp.replace(OUTPUT_FILE)
                print("✨ H.264 변환 완료")
        except Exception as e:
            print(f"⚠️ H.264 자동 변환 경고: {e}")

    used_schedule = [s for s in SCHEDULE if s[0] < actual_record_sec]
    labels = build_labels(used_schedule)
    LABELS_FILE.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    actual_duration = frames_recorded / fps
    print("\n" + "=" * 65)
    print("✅ 녹화가 성공적으로 완료되었습니다!")
    print("=" * 65)
    print(f"• 비디오 파일: {OUTPUT_FILE.resolve()}")
    print(f"• 라벨 파일(자동 생성): {LABELS_FILE.resolve()}")
    print(f"• 총 녹화 시간: {actual_duration:.1f}초 ({frames_recorded} 프레임, {fps:.1f} FPS)")
    print("-" * 65)
    print("다음 명령어로 평가하세요 (v1/v2 둘 다 비교 권장):")
    print("  python eval/evaluate_full_actions.py eval/video/benchmark_movement_back.mp4 \\")
    print("      --config eval/configs/v1_movement_baseline.json \\")
    print("      --report eval/output/back_report_v1.json --annotate eval/output/back_annotated_v1.mp4")
    print("  python eval/evaluate_full_actions.py eval/video/benchmark_movement_back.mp4 \\")
    print("      --config eval/configs/v2_movement_improved.json \\")
    print("      --report eval/output/back_report_v2.json --annotate eval/output/back_annotated_v2.mp4")
    print("  python eval/score_movement.py --report eval/output/back_report_v1.json \\")
    print("      --labels eval/video/benchmark_movement_back_labels.json --out eval/output/back_score_v1.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
