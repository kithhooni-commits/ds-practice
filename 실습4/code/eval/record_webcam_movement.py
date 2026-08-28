"""Movement & Rotation Benchmark Video Recorder with On-Screen Protocol Overlay.

`record_webcam.py`(펀치용)의 자매 스크립트. `RECORDING_GUIDE_MOVEMENT.md`의 92초
프로토콜을 화면에 안내하며 웹캠을 녹화하고, **녹화가 끝나면 같은 스케줄에서 정답
라벨(benchmark_movement_labels.json)을 자동 생성**한다.

라벨을 손으로 다시 매길 필요가 없는 이유: 이 스크립트가 화면에 "지금 이 동작을 하라"고
지시하는 시각과, 그 지시를 GT로 기록하는 시각이 **같은 타이머에서 나온 같은 값**이기 때문이다.
다만 사람이 지시에 반응하는 데 약간의 지연이 있으므로, 채점기(`score_movement.py`)의
`onset_tolerance_ms`를 이 반응 지연을 흡수할 수 있게 넉넉히(400ms) 잡는다. 특정 구간이
크게 어긋났다면(예: 지시보다 한참 늦게 반응) `benchmark_movement_labels.json`을 영상을 보며
손으로 보정하면 된다 — 자동 생성본은 항상 그 출발점이다.

Usage:
  conda activate pjt-4
  python eval/record_webcam_movement.py
"""
import json
import sys
import time
from pathlib import Path
import cv2

OUTPUT_DIR = Path(__file__).parent / "video"
OUTPUT_FILE = OUTPUT_DIR / "benchmark_movement.mp4"
LABELS_FILE = OUTPUT_DIR / "benchmark_movement_labels.json"

# (start_s, end_s, title, desc, is_action, gt) — gt: None | (axis, state)
# axis: "move" | "rot" | "negative". negative 구간은 move/rot 모두 NONE 이어야 정상.
#
# 동작 사이 텀: 액션 5s + 중립(휴식) 5s. 예전 버전(3s/3s)은 "지시가 뜨자마자 반응해서
# 딱 맞춰 끝내야" 해서 실제로 해보면 빠듯했다 — 자세를 잡는 데도, 원위치로 돌아오는 데도
# 시간이 걸리기 때문에 액션 구간 앞뒤에 여유가 있어야 재현 가능한 동작이 나온다.
SCHEDULE = [
    (0, 6, "0. 캘리브레이션", "기립 자세로 가만히 정지 (양 어깨 수평)", False, None),

    (6, 11, "1. FORWARD (크게)", "어깨를 수평으로 유지한 채 크게 앞으로 숙이기", True, ("move", "FORWARD")),
    (11, 16, "⏸️ 중립", "기립 자세로 천천히 복귀", False, None),
    (16, 21, "2. FORWARD (살짝)", "어깨 수평 유지, 살짝만 앞으로 숙이기", True, ("move", "FORWARD")),
    (21, 26, "⏸️ 중립", "기립 자세로 천천히 복귀", False, None),

    (26, 31, "3. BACK (크게)", "어깨 수평 유지, 크게 뒤로 젖히기", True, ("move", "BACK")),
    (31, 36, "⏸️ 중립", "기립 자세로 천천히 복귀", False, None),
    (36, 41, "4. BACK (살짝)", "어깨 수평 유지, 살짝만 뒤로 젖히기", True, ("move", "BACK")),
    (41, 46, "⏸️ 중립", "기립 자세로 천천히 복귀", False, None),

    (46, 51, "5. LEFT 스텝", "왼쪽 어깨를 내려 어깨선을 기울이기", True, ("move", "LEFT")),
    (51, 56, "⏸️ 중립", "기립 자세로 복귀 (어깨 수평)", False, None),
    (56, 61, "6. RIGHT 스텝", "오른쪽 어깨를 내려 어깨선을 기울이기", True, ("move", "RIGHT")),
    (61, 66, "⏸️ 중립", "기립 자세로 복귀 (어깨 수평)", False, None),

    (66, 68, "7. 빠른 왕복 (LEFT)", "왼쪽으로 기울이기", True, ("move", "LEFT")),
    (69, 71, "7. 빠른 왕복 (RIGHT)", "오른쪽으로 기울이기", True, ("move", "RIGHT")),
    (72, 74, "7. 빠른 왕복 (LEFT)", "왼쪽으로 기울이기", True, ("move", "LEFT")),

    (74, 79, "⏸️ 중립", "기립 자세로 복귀 — 어깨 수평 확인", False, None),
    (79, 84, "8. ROT_LEFT", "어깨는 수평! 양 주먹을 함께 왼쪽으로 한 번 쓸기", True, ("rot", "ROT_LEFT")),
    (84, 89, "⏸️ 중립", "주먹을 가운데로 천천히 되돌리기", False, None),
    (89, 94, "9. ROT_RIGHT", "어깨는 수평! 양 주먹을 함께 오른쪽으로 한 번 쓸기", True, ("rot", "ROT_RIGHT")),
    (94, 99, "⏸️ 중립", "주먹을 가운데로 천천히 되돌리기", False, None),
    (99, 105, "10. ROT_LEFT x2", "왼쪽으로 쓸기 -> 가운데로 되돌리기 -> 다시 왼쪽으로", True, ("rot", "ROT_LEFT")),

    (105, 110, "⏸️ 중립", "기립 자세로 복귀", False, None),

    (110, 114, "11. [오탐 테스트] 잽 4회", "왼손 잽을 빠르게 4회 (몸이 흔들려도 이동/회전이 뜨면 안 됨)", True, ("negative", None)),
    (114, 118, "12. [오탐 테스트] 훅 3회", "훅 3회 (팔꿈치를 접고 휘두르기)", True, ("negative", None)),
    (118, 122, "13. [오탐 테스트] 어퍼컷 3회", "어퍼컷 3회 (아래에서 위로)", True, ("negative", None)),
    (122, 126, "14. [오탐 테스트] 살짝 흔들기", "몸을 푸는 정도로만 살짝 좌우로 흔들기 (기울이지 않기)", True, ("negative", None)),
    (126, 130, "15. [오탐 테스트] 듀얼 가드", "양 주먹을 얼굴 옆으로 올려 가드 자세 유지", True, ("negative", None)),

    (130, 135, "⏸️ 중립", "기립 자세로 복귀", False, None),

    (135, 140, "16. 전진 + 좌회전 동시", "앞으로 숙인 채 양 주먹을 왼쪽으로 쓸기", True, [("move", "FORWARD"), ("rot", "ROT_LEFT")]),
    (140, 145, "⏸️ 중립", "기립 자세로 복귀", False, None),
    (145, 150, "17. 잽 뿌리며 전진", "앞으로 숙인 채로 잽을 계속 뻗기 (전진 상태 유지되어야 함)", True, ("move", "FORWARD")),

    (150, 155, "18. 마무리", "기립 자세로 정지 후 녹화 종료", False, None),
]

TOTAL_RECORD_SEC = SCHEDULE[-1][1]
PRE_COUNTDOWN_SEC = 5


def get_phase_info(elapsed_sec):
    for start, end, title, desc, is_action, _gt in SCHEDULE:
        if start <= elapsed_sec < end:
            return title, desc, is_action, end - elapsed_sec
    return "녹화 완료", "영상을 저장 중입니다...", False, 0.0


def build_labels(schedule, record_start_ms=0, onset_tolerance_ms=400, offset_tolerance_ms=300):
    """SCHEDULE(초 단위, gt 포함)을 score_movement.py 스키마의 라벨 JSON으로 변환한다."""
    move_segments, rot_segments, negative_windows = [], [], []
    for start, end, _title, desc, _is_action, gt in schedule:
        if gt is None:
            continue
        start_ms = record_start_ms + round(start * 1000)
        end_ms = record_start_ms + round(end * 1000)
        entries = gt if isinstance(gt, list) else [gt]
        for axis, state in entries:
            if axis == "move":
                move_segments.append({"start_ms": start_ms, "end_ms": end_ms, "state": state, "desc": desc})
            elif axis == "rot":
                rot_segments.append({"start_ms": start_ms, "end_ms": end_ms, "state": state, "desc": desc})
            elif axis == "negative":
                negative_windows.append({"start_ms": start_ms, "end_ms": end_ms, "desc": desc})

    return {
        "case": "benchmark_movement_v1",
        "source": "eval/video/benchmark_movement.mp4",
        "note": "record_webcam_movement.py 가 촬영 스케줄(지시 시각)로부터 자동 생성한 라벨입니다. "
                "사람의 반응 지연을 흡수하도록 tolerance를 넉넉히 잡았지만, 특정 구간에서 "
                "반응이 유난히 늦었다면 영상을 보며 start_ms/end_ms를 손으로 보정하세요.",
        "onset_tolerance_ms": onset_tolerance_ms,
        "offset_tolerance_ms": offset_tolerance_ms,
        "move_segments": move_segments,
        "rot_segments": rot_segments,
        "negative_windows": negative_windows,
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
    """인덱스 0..max_index × 백엔드(DSHOW, MSMF, 기본값) 조합을 순서대로 시도해
    실제로 프레임을 읽을 수 있는 첫 조합을 반환한다. isOpened()만으로는 부족하다 —
    Windows에서는 열기는 성공했다고 보고해도 grabFrame이 실패하는 경우가 흔하다
    (다른 프로세스의 점유, 카메라 개인정보 설정, 드라이버 문제 등).
    """
    backends = [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_ANY, "ANY"),
    ]
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
    print(f"\U0001f9cd {TOTAL_RECORD_SEC:.0f}초 이동/회전 벤치마크 비디오 레코더")
    print("=" * 65)
    print(f"• 저장 경로: {OUTPUT_FILE.resolve()}")
    print(f"• 라벨 자동 생성: {LABELS_FILE.resolve()}")
    print("• 조작법: [Q] 키를 누르면 녹화를 조기 종료하고 저장합니다.")
    print("-" * 65)

    cap = open_camera()
    if cap is None:
        print("\n❌ 오류: 어떤 인덱스/백엔드 조합으로도 웹캠을 열 수 없습니다.")
        print("   흔한 원인들 (위에서부터 순서대로 확인해 보세요):")
        print("   1) 다른 프로그램이 카메라를 점유 중 — 브라우저의 fighter_client 탭,")
        print("      Zoom/Teams, 다른 OpenCV 스크립트를 전부 닫고 재시도")
        print("   2) Windows 설정 > 개인정보 보호 및 보안 > 카메라 >")
        print("      '데스크톱 앱이 카메라에 액세스하도록 허용' 이 꺼져 있음")
        print("   3) 카메라 드라이버가 멈춘 상태 — USB 웹캠이면 뽑았다 다시 꽂기,")
        print("      내장 카메라면 재부팅으로 대부분 해결됨")
        print("   4) 카메라가 여러 대(가상 카메라 포함)라 인덱스가 0이 아닐 수 있음 —")
        print("      python -c \"import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(5)]\" 로 확인")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
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
        frame = cv2.flip(frame, 1)
        elapsed_prep = time.time() - t_start_prep
        remain_prep = PRE_COUNTDOWN_SEC - elapsed_prep
        if remain_prep <= 0:
            break

        display = frame.copy()
        draw_text(display, "MOVEMENT/ROTATION BENCHMARK RECORDER", (20, 45), font_scale=0.9, color=(0, 255, 255))
        draw_text(display, f"Starting in {int(remain_prep) + 1} seconds...", (20, 90), font_scale=0.8, color=(200, 200, 200))

        count_text = str(int(remain_prep) + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (cw, ch), _ = cv2.getTextSize(count_text, font, 3.5, 6)
        cx, cy = (width - cw) // 2, (height + ch) // 2
        cv2.rectangle(display, (cx - 20, cy - ch - 20), (cx + cw + 20, cy + 20), (0, 0, 0), -1)
        cv2.putText(display, count_text, (cx, cy), font, 3.5, (0, 255, 0), 6, cv2.LINE_AA)

        draw_text(display, "Stand 1.2-1.8m away, shoulders level, guard stance.", (20, height - 30), font_scale=0.7, color=(200, 200, 200))

        cv2.imshow("Movement Benchmark Recorder (Press Q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("녹화가 취소되었습니다.")
            return

    # Phase 2: 본 녹화
    print(f"\n🔴 녹화 시작! {TOTAL_RECORD_SEC:.0f}초간 화면 지시에 맞춰 동작을 수행하세요.")
    t_start_record = time.time()
    record_start_wall_ms = int(t_start_record * 1000)
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

        # 저장은 원본(비반전) 방향으로 — MediaPipe는 이 파일을 그대로 읽어 left/right
        # 랜드마크를 판정하므로, 여기서 거울 반전된 걸 저장하면 왼쪽/오른쪽이 통째로
        # 뒤바뀐 채 채점된다 (실제로 한 번 겪은 버그). 화면 미리보기만 거울 모드로 보여준다.
        out.write(frame)
        frames_recorded += 1

        display = cv2.flip(frame, 1).copy()
        title, desc, is_action, phase_remain = get_phase_info(elapsed)

        rec_dot = "REC [LIVE]" if int(elapsed * 2) % 2 == 0 else "REC       "
        draw_text(display, f"{rec_dot}  {elapsed:04.1f}s / {TOTAL_RECORD_SEC:.0f}s (Left: {remain:04.1f}s)", (20, 45), font_scale=0.75, color=(0, 0, 255))

        step_col = (0, 255, 255) if is_action else (0, 200, 255)
        draw_text(display, f"{title} (Left: {phase_remain:04.1f}s)", (20, 85), font_scale=0.8, color=step_col)

        desc_col = (255, 255, 255) if is_action else (100, 255, 100)
        draw_text(display, f"Action: {desc}", (20, 125), font_scale=0.7, color=desc_col)

        # 참고용 가이드선일 뿐이다 — 실제 판정(evaluate_full_actions.py)은 두 어깨 랜드마크
        # 사이의 "기울기 각도"만 본다. 화면상 어느 높이에 있는지는 전혀 상관없으므로
        # 이 선에 어깨를 맞추려 애쓸 필요가 없다. 얼굴+어깨가 편하게 다 나오는 위치면 충분하다.
        # (화면 중간쯤에 둔 것도 "대략 이 높이에 서면 상반신이 다 나온다"는 프레이밍 참고용일 뿐.)
        guide_y = int(height * 0.5)
        cv2.line(display, (0, guide_y), (width, guide_y), (60, 160, 60), 1, cv2.LINE_AA)
        draw_text(display, "(참고용 프레이밍 가이드 — 여기 맞출 필요 없음. 기울기만 중요)",
                  (20, guide_y - 8), font_scale=0.5, color=(120, 200, 120), bg_box=False)

        if not is_action and 0 < elapsed < TOTAL_RECORD_SEC - 5:
            banner_text = f"NEXT IN: {int(phase_remain) + 1}s"
            draw_text(display, banner_text, (width // 2 - 140, height // 2), font_scale=1.1, color=(0, 255, 255), thickness=3)

        draw_text(display, "Press [Q] to finish recording early", (20, height - 25), font_scale=0.6, color=(180, 180, 180))

        cv2.imshow("Movement Benchmark Recorder (Press Q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n⏹ 사용자가 [Q] 키를 눌러 녹화를 조기 종료했습니다.")
            break

    actual_record_sec = time.time() - t_start_record
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # H.264 변환 (있으면)
    import subprocess
    import shutil
    ffmpeg_cmd = shutil.which("ffmpeg")
    if ffmpeg_cmd:
        print("⚙️ 브라우저(HTML5) 호환을 위한 H.264 비디오 최적화 중...")
        h264_tmp = OUTPUT_DIR / "benchmark_movement_h264.mp4"
        try:
            res = subprocess.run(
                [ffmpeg_cmd, "-y", "-i", str(OUTPUT_FILE), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", str(h264_tmp)],
                capture_output=True, text=True
            )
            if res.returncode == 0 and h264_tmp.exists():
                h264_tmp.replace(OUTPUT_FILE)
                print("✨ H.264 변환 완료")
        except Exception as e:
            print(f"⚠️ H.264 자동 변환 경고: {e}")

    # 조기 종료됐다면(actual_record_sec < TOTAL_RECORD_SEC) 그 이후 스케줄은 라벨에서 잘라낸다.
    used_schedule = [s for s in SCHEDULE if s[0] < actual_record_sec]
    labels = build_labels(used_schedule, record_start_ms=0)
    LABELS_FILE.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    actual_duration = frames_recorded / fps
    print("\n" + "=" * 65)
    print("✅ 녹화가 성공적으로 완료되었습니다!")
    print("=" * 65)
    print(f"• 비디오 파일: {OUTPUT_FILE.resolve()}")
    print(f"• 라벨 파일(자동 생성): {LABELS_FILE.resolve()}")
    print(f"• 총 녹화 시간: {actual_duration:.1f}초 ({frames_recorded} 프레임, {fps:.1f} FPS)")
    if actual_record_sec < TOTAL_RECORD_SEC - 1:
        print(f"⚠️ 조기 종료됨 ({actual_record_sec:.1f}s) — {actual_record_sec:.0f}초 이후 구간은 라벨에서 제외했습니다.")
    print("-" * 65)
    print("다음 명령어로 룰베이스 평가를 실행하세요:")
    print("  conda activate pjt-4")
    print("  python eval/evaluate_full_actions.py eval/video/benchmark_movement.mp4 \\")
    print("      --report eval/output/movement_report.json --annotate eval/output/annotated_movement.mp4")
    print("  python eval/score_movement.py --report eval/output/movement_report.json \\")
    print("      --labels eval/video/benchmark_movement_labels.json --out eval/output/movement_score.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
