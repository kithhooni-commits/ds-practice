"""60-Second Boxing Benchmark Video Recorder with On-Screen Protocol Overlay.

Records webcam footage directly to `iter4/eval/video/benchmark.mp4` while showing
real-time timer, countdown, and boxing protocol instructions on the screen.

Usage:
  conda activate pjt-4
  python iter4/eval/record_webcam.py
"""
import sys
import time
from pathlib import Path
import cv2

OUTPUT_DIR = Path(__file__).parent / "video"
OUTPUT_FILE = OUTPUT_DIR / "benchmark.mp4"
LABELS_TEMPLATE = OUTPUT_DIR / "benchmark_labels_template.json"

# 90-Second Protocol Schedule with 5-Second Rest/Prep Intervals Between Actions
SCHEDULE = [
    (0, 6, "1. 준비 & 캘리브레이션", "양손 가드 자세로 가만히 웹캠을 바라봅니다.", False),
    (6, 18, "2. 직선 펀치 (Straight)", "왼손 잽 4회 -> 오른손 스트레이트 4회 (여유 있게 1.5초 간격)", True),
    (18, 23, "⏸️ 숨고르기 & 다음 준비", "가드 자세를 유지하며 호흡합니다. (다음: 훅 펀치)", False),
    (23, 35, "3. 곡선 펀치 (Hook)", "왼손 훅 3회 -> 오른손 훅 3회 (호를 그리며 휘두름)", True),
    (35, 40, "⏸️ 숨고르기 & 다음 준비", "가드 자세를 유지하며 호흡합니다. (다음: 어퍼컷)", False),
    (40, 52, "4. 하단 펀치 (Uppercut)", "왼손 어퍼 3회 -> 오른손 어퍼 3회 (아래에서 위로)", True),
    (52, 57, "⏸️ 숨고르기 & 다음 준비", "가드 자세를 유지하며 호흡합니다. (다음: 풋워크)", False),
    (57, 70, "5. 풋워크 & 몸 기울임", "상체 앞으로 2회 -> 뒤로 2회 -> 좌/우 각 2회", True),
    (70, 75, "⏸️ 숨고르기 & 다음 준비", "가드 자세를 유지하며 호흡합니다. (다음: 콤보)", False),
    (75, 85, "6. 실전 콤보 (Combos)", "원투(잽-스트레이트) 3회 -> 잽-크로스-훅 2회", True),
    (85, 90, "7. 마무리", "가드 자세 유지 후 녹화를 종료합니다.", False),
]

TOTAL_RECORD_SEC = 90
PRE_COUNTDOWN_SEC = 5


def get_phase_info(elapsed_sec):
    for start, end, title, desc, is_action in SCHEDULE:
        if start <= elapsed_sec < end:
            return title, desc, is_action, end - elapsed_sec
    return "녹화 완료", "영상을 저장 중입니다...", False, 0.0


def draw_text(img, text, pos, font_scale=0.7, color=(255, 255, 255), thickness=2, bg_box=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    if bg_box:
        cv2.rectangle(img, (x - 6, y - h - 6), (x + w + 6, y + baseline + 6), (20, 20, 20), -1)
        cv2.rectangle(img, (x - 6, y - h - 6), (x + w + 6, y + baseline + 6), (80, 80, 80), 1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("🥊 60초 복싱 벤치마크 비디오 레코더")
    print("=" * 65)
    print(f"• 저장 경로: {OUTPUT_FILE.resolve()}")
    print("• 조작법: [Q] 키를 누르면 녹화를 조기 종료하고 저장합니다.")
    print("-" * 65)

    # Windows 환경에서 가장 안정적인 DirectShow (CAP_DSHOW) 백엔드 우선 시도
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # DSHOW 실패 시 기본 백엔드로 폴백
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 오류: 웹캠을 열 수 없습니다. 브라우저나 다른 앱에서 카메라를 사용 중인지 확인하세요.")
        sys.exit(1)

    # 웹캠 해상도 및 FPS 설정 (1280x720 시도, 실패 시 640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 최초 프레임 읽기 테스트
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("⚠️ 1280x720 해상도 실패, 기본 해상도로 재시도합니다...")
        cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("❌ 오류: 웹캠에서 프레임을 읽어올 수 없습니다.")
            print("   브라우저(Fighter Client 등)나 다른 프로그램이 웹캠을 점유하고 있는지 확인해 주세요.")
            sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 120:
        fps = 30.0

    print(f"• 카메라 해상도: {width} x {height} ({fps:.1f} FPS)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_FILE), fourcc, fps, (width, height))

    # Phase 1: 5초 준비 카운트다운
    t_start_prep = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # 거울 모드 프리뷰
        elapsed_prep = time.time() - t_start_prep
        remain_prep = PRE_COUNTDOWN_SEC - elapsed_prep

        if remain_prep <= 0:
            break

        # UI Overlay (카운트다운)
        display = frame.copy()
        draw_text(display, "BOXING BENCHMARK RECORDER", (20, 45), font_scale=0.9, color=(0, 255, 255))
        draw_text(display, f"Starting in {int(remain_prep) + 1} seconds...", (20, 90), font_scale=0.8, color=(200, 200, 200))

        # 중앙 대형 카운트다운
        count_text = str(int(remain_prep) + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (cw, ch), _ = cv2.getTextSize(count_text, font, 3.5, 6)
        cx, cy = (width - cw) // 2, (height + ch) // 2
        cv2.rectangle(display, (cx - 20, cy - ch - 20), (cx + cw + 20, cy + 20), (0, 0, 0), -1)
        cv2.putText(display, count_text, (cx, cy), font, 3.5, (0, 255, 0), 6, cv2.LINE_AA)

        draw_text(display, "Stand 1.5m away and get into boxing guard stance.", (20, height - 30), font_scale=0.7, color=(200, 200, 200))

        cv2.imshow("Boxing Benchmark Recorder (Press Q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("녹화가 취소되었습니다.")
            return

    # Phase 2: 본 녹화 (60초)
    print("\n🔴 녹화 시작! 60초간 프로토콜에 맞춰 동작을 수행하세요.")
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
        frame = cv2.flip(frame, 1)  # 거울 모드로 통일 저장

        now = time.time()
        elapsed = now - t_start_record
        remain = max(0.0, TOTAL_RECORD_SEC - elapsed)

        if elapsed >= TOTAL_RECORD_SEC:
            break

        # 원본 프레임을 파일에 기록
        out.write(frame)
        frames_recorded += 1

        # 프리뷰 화면에만 안내 오버레이 그리기
        display = frame.copy()
        title, desc, is_action, phase_remain = get_phase_info(elapsed)

        # 상단 HUD
        rec_dot = "REC [LIVE]" if int(elapsed * 2) % 2 == 0 else "REC       "
        draw_text(display, f"{rec_dot}  {elapsed:04.1f}s / {TOTAL_RECORD_SEC}s (Total Left: {remain:04.1f}s)", (20, 45), font_scale=0.75, color=(0, 0, 255))

        step_col = (0, 255, 255) if is_action else (0, 200, 255)
        step_title = f"{title} (Left: {phase_remain:04.1f}s)"
        draw_text(display, step_title, (20, 85), font_scale=0.85, color=step_col)

        desc_col = (255, 255, 255) if is_action else (100, 255, 100)
        draw_text(display, f"Action: {desc}", (20, 125), font_scale=0.75, color=desc_col)

        # 휴식 구간 중앙 배너
        if not is_action and 0 < elapsed < TOTAL_RECORD_SEC - 5:
            banner_text = f"REST / READY: {int(phase_remain) + 1}s"
            draw_text(display, banner_text, (width // 2 - 160, height // 2), font_scale=1.1, color=(0, 255, 255), thickness=3)

        # 하단 조작 가이드
        draw_text(display, "Press [Q] to finish recording early", (20, height - 25), font_scale=0.6, color=(180, 180, 180))

        cv2.imshow("Boxing Benchmark Recorder (Press Q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n⏹ 사용자가 [Q] 키를 눌러 녹화를 조기 종료했습니다.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 웹 브라우저(HTML5 Video) 호환을 위한 H.264 (avc1) 자동 변환
    import subprocess
    import shutil
    ffmpeg_cmd = shutil.which("ffmpeg")
    if ffmpeg_cmd:
        print("⚙️ 브라우저(HTML5) 호환을 위한 H.264 비디오 최적화 중...")
        h264_tmp = OUTPUT_DIR / "benchmark_h264.mp4"
        try:
            res = subprocess.run(
                [ffmpeg_cmd, "-y", "-i", str(OUTPUT_FILE), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", str(h264_tmp)],
                capture_output=True, text=True
            )
            if res.returncode == 0 and h264_tmp.exists():
                h264_tmp.replace(OUTPUT_FILE)
                print("✨ H.264 변환 완료 (Chrome/Edge 브라우저 완벽 호환)")
        except Exception as e:
            print(f"⚠️ H.264 자동 변환 경고: {e}")

    actual_duration = frames_recorded / fps
    print("\n" + "=" * 65)
    print("✅ 녹화가 성공적으로 완료되었습니다!")
    print("=" * 65)
    print(f"• 비디오 파일: {OUTPUT_FILE.resolve()}")
    print(f"• 총 녹화 시간: {actual_duration:.1f}초 ({frames_recorded} 프레임, {fps:.1f} FPS)")
    print("-" * 65)
    print("다음 명령어로 즉시 동작인식 정확도 평가 또는 3D 리플레이를 구동할 수 있습니다:")
    print(f"  conda activate pjt-4")
    print(f"  python iter3/eval/evaluate_video.py iter3/eval/video/benchmark.mp4 --annotate iter3/eval/output/annotated_benchmark.mp4")
    print(f"  (브라우저 3D 뷰어: https://localhost:8000/replay)")
    print("=" * 65)


if __name__ == "__main__":
    main()
