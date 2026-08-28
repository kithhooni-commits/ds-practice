"""movement_pass.py — 랜드마크 캐시로부터 FullActionEvaluator 를 돌려
프레임별 move/rot/guard 상태와 요약 통계를 뽑는 얇은 러너.

`evaluate_full_actions.py` 는 비디오 파일과 MediaPipe 를 직접 물고 있어
캐시를 재활용하기 어렵다. 이 모듈은 그 안의 `FullActionEvaluator` 만
가져다 쓰기 때문에 cv2/mediapipe 없이도 돌아간다 (run_pipeline 의
Stage 2.5 에서 이 사실이 중요).

산출물:
  * movement_timeline.jsonl : 프레임당 { t_ms, frame, move, rot, guard,
                              move_intensity, roll, pitch, locked }
  * movement 요약 dict       : distribution / guard_coverage_pct /
                              top_segments / phase_analysis

phase_analysis 는 punches 채점과 같은 phase 정의(90초 프로토콜 또는
labels.phases override)를 재사용한다. 라벨이 없어도 다음이 계산된다:
  - footwork_recall_proxy  : is_action=False 이고 이름에 "풋워크" 포함된
                             구간에서 move!=NONE 프레임 비율 (풋워크
                             구간에 실제로 무빙이 잡혔는가)
  - static_fp_proxy        : 그 외 is_action=False 구간에서 move!=NONE
                             프레임 비율 (Calibration/Rest/Cooldown 등
                             정지 구간의 오검출)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import sys

# 로컬 모듈 임포트
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from evaluate_full_actions import FullActionEvaluator  # noqa: E402
from scoring import iter_landmarks_jsonl  # noqa: E402


def run_movement_pass(landmarks_path: Path, out_jsonl: Path | None = None):
    """랜드마크 캐시를 순회하며 무빙 상태 timeline 을 만든다.

    Returns:
        timeline (list[dict]): frame_result 리스트
    """
    evaluator = FullActionEvaluator()
    timeline: list[dict] = []
    for frame_idx, (ts_ms, lm, wl) in enumerate(iter_landmarks_jsonl(landmarks_path)):
        info = evaluator.process_frame(lm, wl or None, ts_ms, frame_idx)
        # process_frame 은 punches 도 뱉지만 여기선 관심 없음 (Stage 2 이 처리)
        # info: t_ms, frame, punches, move, move_intensity, rot, guard,
        #       locked, roll, pitch
        timeline.append({
            "t_ms": info["t_ms"],
            "frame": info["frame"],
            "move": info["move"],
            "move_intensity": info["move_intensity"],
            "rot": info["rot"],
            "guard": info["guard"],
            "locked": info["locked"],
            "roll": info["roll"],
            "pitch": info["pitch"],
        })

    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for row in timeline:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return timeline


def _extract_segments(timeline: list[dict], key: str) -> list[dict]:
    """timeline 에서 연속된 동일 상태 구간을 뽑는다. NONE 은 제외."""
    segments = []
    if not timeline:
        return segments
    cur_state = timeline[0][key]
    seg_start_i = 0
    for i in range(1, len(timeline)):
        if timeline[i][key] != cur_state:
            if cur_state != "NONE":
                segments.append({
                    "state": cur_state,
                    "start_ms": timeline[seg_start_i]["t_ms"],
                    "end_ms": timeline[i - 1]["t_ms"],
                    "dwell_ms": timeline[i - 1]["t_ms"] - timeline[seg_start_i]["t_ms"],
                    "frames": i - seg_start_i,
                })
            cur_state = timeline[i][key]
            seg_start_i = i
    # 마지막 구간
    if cur_state != "NONE":
        last_i = len(timeline) - 1
        segments.append({
            "state": cur_state,
            "start_ms": timeline[seg_start_i]["t_ms"],
            "end_ms": timeline[last_i]["t_ms"],
            "dwell_ms": timeline[last_i]["t_ms"] - timeline[seg_start_i]["t_ms"],
            "frames": last_i - seg_start_i + 1,
        })
    return segments


def summarize_movement(timeline: list[dict], phases: list[tuple] | None = None) -> dict:
    """movement 통계 요약 + 라벨 없는 phase 분석.

    phases: (t0_s, t1_s, name, is_action) 튜플 리스트. None 이면
    run_pipeline 의 기본 90초 프로토콜을 그대로 사용한다.

    footwork/static proxy 정의:
      * 이름에 "풋워크" 를 포함하는 비동작 구간의 move!=NONE 비율
        → footwork_recall_proxy (높을수록 좋음, 풋워크 구간에서 무빙이
          실제로 잡혔다는 뜻)
      * 그 외 비동작 구간(Calibration/Rest/Cooldown)의 move!=NONE 비율
        → static_fp_proxy (낮을수록 좋음, 정지 구간에 오검출이 적음)
    """
    n = len(timeline)
    if n == 0:
        return {"note": "empty timeline"}

    # --- 전체 분포 ---
    move_counter = Counter(x["move"] for x in timeline)
    rot_counter = Counter(x["rot"] for x in timeline)
    guard_frames = sum(1 for x in timeline if x["guard"])
    locked_frames = sum(1 for x in timeline if x["locked"])

    move_distribution = {k: round(v / n * 100, 1) for k, v in move_counter.items()}
    rot_distribution = {k: round(v / n * 100, 1) for k, v in rot_counter.items()}

    # --- 세그먼트 ---
    move_segments = _extract_segments(timeline, "move")
    rot_segments = _extract_segments(timeline, "rot")
    # top-N (dwell 긴 것만 저장 — 리포트 부피 관리)
    move_segments_top = sorted(move_segments, key=lambda s: -s["dwell_ms"])[:10]
    rot_segments_top = sorted(rot_segments, key=lambda s: -s["dwell_ms"])[:10]

    summary = {
        "total_frames": n,
        "move_distribution_pct": move_distribution,
        "rot_distribution_pct": rot_distribution,
        "guard_coverage_pct": round(guard_frames / n * 100, 1),
        "punch_lock_pct": round(locked_frames / n * 100, 1),
        "move_segments_count": len(move_segments),
        "rot_segments_count": len(rot_segments),
        "move_segments_top": move_segments_top,
        "rot_segments_top": rot_segments_top,
    }

    # --- phase 기반 정성 지표 ---
    if phases is not None:
        phase_stats = []
        footwork_recall_num = footwork_recall_den = 0
        static_fp_num = static_fp_den = 0

        for t0, t1, name, is_action in phases:
            in_range = [x for x in timeline if t0 * 1000 <= x["t_ms"] < t1 * 1000]
            if not in_range:
                continue
            move_active = sum(1 for x in in_range if x["move"] != "NONE")
            rot_active = sum(1 for x in in_range if x["rot"] != "NONE")
            guard_here = sum(1 for x in in_range if x["guard"])
            # 다수결 dominant move (NONE 포함)
            dom = Counter(x["move"] for x in in_range).most_common(1)[0][0]
            phase_stats.append({
                "phase": name,
                "range": f"{t0:02d}~{t1:02d}s",
                "is_action": is_action,
                "frames": len(in_range),
                "move_active_pct": round(move_active / len(in_range) * 100, 1),
                "rot_active_pct": round(rot_active / len(in_range) * 100, 1),
                "guard_pct": round(guard_here / len(in_range) * 100, 1),
                "dominant_move": dom,
            })
            # proxy 집계
            if not is_action:
                if "풋워크" in name:
                    footwork_recall_num += move_active
                    footwork_recall_den += len(in_range)
                else:
                    static_fp_num += move_active
                    static_fp_den += len(in_range)

        summary["phase_analysis"] = {
            "phases": phase_stats,
            "footwork_recall_proxy": (
                round(footwork_recall_num / footwork_recall_den * 100, 1)
                if footwork_recall_den else None
            ),
            "static_fp_proxy": (
                round(static_fp_num / static_fp_den * 100, 1)
                if static_fp_den else None
            ),
        }

    return summary
