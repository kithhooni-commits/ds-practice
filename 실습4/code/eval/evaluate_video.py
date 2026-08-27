"""Offline punch evaluation on a video file.

Ports the runtime punch pipeline of iter3/server/templates/fighter_client.html
(MediaPipe Pose upper-body 7 nodes + window-latch punch detector) to Python,
runs it frame-by-frame on an input video, and reports punch counts, types,
speeds and form metrics.

Usage:
  conda activate pjt-4
  python iter3/eval/evaluate_video.py video/iter3-poc.mp4
  python iter3/eval/evaluate_video.py video/iter3-poc.mp4 --annotate out.mp4
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

# cv2/mediapipe 는 **영상을 디코딩할 때만** 필요하다.
# --landmarks 로 캐시된 JSONL 을 재생할 때는 한 줄도 쓰지 않는데,
# 최상단에서 import 하면 무거운 CV 스택이 없는 환경에서 채점조차 못 돌린다.
# (mediapipe 는 Python 3.13 휠이 아직 없다.) 그래서 필요한 시점에 불러온다.
cv2 = None
mp = None
BaseOptions = None
vision = None


def _require_cv():
    """영상 경로에서만 호출한다. 없으면 무엇을 설치해야 하는지 알려주고 끝낸다."""
    global cv2, mp, BaseOptions, vision
    if cv2 is not None:
        return
    try:
        import cv2 as _cv2
        import mediapipe as _mp
        from mediapipe.tasks.python import BaseOptions as _BaseOptions
        from mediapipe.tasks.python import vision as _vision
    except ImportError as exc:
        raise SystemExit(
            "영상 처리에는 opencv-python 과 mediapipe 가 필요합니다 "
            f"({exc})." + chr(10) +
            "  pip install opencv-python mediapipe" + chr(10) +
            "이미 뽑아 둔 랜드마크가 있다면 --landmarks <jsonl> 로 영상 없이 채점할 수 있습니다."
        ) from exc
    cv2, mp, BaseOptions, vision = _cv2, _mp, _BaseOptions, _vision

from scoring import iter_landmarks_jsonl, load_labels_file, score_predictions

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16
NODE_IDS = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR]
SKELETON = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6)]

VIS_MIN = 0.5
ARM_VIS_MIN = 0.3
SHOULDER_M = 0.40

# Synced with server/static/punch_core.js (PUNCH_TUNE)
PUNCH_ARM = 1.0
PUNCH_EXTEND = 0.40
PUNCH_SPEED = 1.6
PUNCH_REACH_N = 0.88
PUNCH_GROW_N = 0.28
PUNCH_WINDOW_MS = 380
PUNCH_CD_MS = 400
PUNCH_CD_ANY_MS = 200
UPPERCUT_VY = 0.55
UPPERCUT_ELBOW = 150
HOOK_VX = 0.56
HOOK_ELBOW = 158

SCALE_TAU_UP = 0.12
SCALE_TAU_DOWN = 0.45
CALIB_MS = 1800

LOCK_MIN_MS = 180
LOCK_MAX_MS = 1100

PUNCH_NAME = {
    "L": {"STRAIGHT": "LEFT_JAB", "HOOK": "LEFT_HOOK", "UPPERCUT": "LEFT_UPPERCUT"},
    "R": {"STRAIGHT": "RIGHT_CROSS", "HOOK": "RIGHT_HOOK", "UPPERCUT": "RIGHT_UPPERCUT"},
}

DEFAULT_MODEL = Path(__file__).with_name("models") / "pose_landmarker_full.task"


def dist3(a, b):
    return math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)


def angle_deg(a, b, c):
    abx, aby, abz = a.x - b.x, a.y - b.y, a.z - b.z
    cbx, cby, cbz = c.x - b.x, c.y - b.y, c.z - b.z
    d = math.hypot(abx, aby, abz) * math.hypot(cbx, cby, cbz)
    if d < 1e-6:
        return 180.0
    cosv = max(-1.0, min(1.0, (abx * cbx + aby * cby + abz * cbz) / d))
    return math.degrees(math.acos(cosv))


def classify_punch(speed, vx, vy, elbow):
    s = max(speed, 1e-3)
    up_ratio = -vy / s
    hook_ratio = abs(vx) / s
    if up_ratio > UPPERCUT_VY and elbow < UPPERCUT_ELBOW:
        kind = "UPPERCUT"
    elif hook_ratio > HOOK_VX and elbow < HOOK_ELBOW:
        kind = "HOOK"
    else:
        kind = "STRAIGHT"
    scores = {
        "UPPERCUT": min(up_ratio / UPPERCUT_VY, 1.5),
        "HOOK": min(hook_ratio / HOOK_VX, 1.5),
        "STRAIGHT": min(1.0 - max(0.0, max(up_ratio, hook_ratio)), 1.5),
    }
    margin = scores[kind] - max(v for k, v in scores.items() if k != kind)
    return kind, margin


class ArmState:
    def __init__(self, side):
        self.side = side
        self.x = self.y = self.z = 0.0
        self.reach = 0.0
        self.t = None
        self.last_punch = -1e9
        self.armed = False
        self.arm_t = 0
        self.peak = 0.0
        self.reach0 = 0.0
        self.pvx = self.pvy = 0.0
        self.pelbow = 180.0
        self.last_speed = 0.0
        self.last_reach_n = 0.0
        self.windows_opened = 0
        self.windows_fired = 0


class Kinematics:
    def __init__(self, side, reach, reach_n, elbow, vx, vy, vz, speed, d_reach):
        self.side = side
        self.reach = reach
        self.reach_n = reach_n
        self.elbow = elbow
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.speed = speed
        self.d_reach = d_reach


def arm_kinematics(st, sh, el, wr, shoulder_w, now_ms):
    reach = dist3(wr, sh)
    elbow = angle_deg(sh, el, wr)
    vx = vy = vz = speed = d_reach = 0.0
    dt = None if st.t is None else (now_ms - st.t) / 1000.0
    if dt is not None and 0.008 < dt < 0.4:
        vx = (wr.x - st.x) / dt
        vy = (wr.y - st.y) / dt
        vz = (wr.z - st.z) / dt
        speed = math.hypot(vx, vy, vz)
        d_reach = (reach - st.reach) / dt
    st.x, st.y, st.z = wr.x, wr.y, wr.z
    st.reach = reach
    st.t = now_ms
    st.last_speed = speed
    st.last_reach_n = reach / shoulder_w
    return Kinematics(st.side, reach, st.last_reach_n, elbow, vx, vy, vz, speed, d_reach)


class PunchEvaluator:
    def __init__(self, calib=True):
        self.calib_enabled = calib
        self.neutral_ready = not calib
        self.calib_t0 = None
        self.calib_n = 0
        self.s_scale = 0.0
        self.last_result_t = None
        self.dt_pose = 0.033
        self.last_punch_any = -1e9
        self.posture_locked = False
        self.arms = {"L": ArmState("L"), "R": ArmState("R")}
        self.events = []
        self.windows_expired = 0

    def process(self, lm, wl, now_ms):
        fired = []
        if self.last_result_t is not None and now_ms > self.last_result_t:
            self.dt_pose = min(max((now_ms - self.last_result_t) / 1000.0, 0.008), 0.4)
        self.last_result_t = now_ms

        if lm is None or len(lm) <= R_WR:
            for st in self.arms.values():
                if st.armed and now_ms - st.arm_t > PUNCH_WINDOW_MS:
                    st.armed = False
                    self.windows_expired += 1
            return fired

        nose, lsh, rsh = lm[NOSE], lm[L_SH], lm[R_SH]
        top_vis_ok = all(
            getattr(p, "visibility", 1.0) >= VIS_MIN for p in (nose, lsh, rsh)
        )
        if not top_vis_ok:
            return fired

        arm_vis = {}
        for side, ids in (("L", (L_SH, L_EL, L_WR)), ("R", (R_SH, R_EL, R_WR))):
            arm_vis[side] = (
                getattr(lm[ids[2]], "visibility", 1.0) >= ARM_VIS_MIN
                and getattr(lm[ids[1]], "visibility", 1.0) >= ARM_VIS_MIN
            )

        since_punch = now_ms - self.last_punch_any
        arms_busy = (
            max(self.arms["L"].last_speed, self.arms["R"].last_speed) > 0.9
            or max(self.arms["L"].last_reach_n, self.arms["R"].last_reach_n) > 1.00
        )
        posture_locked = since_punch < LOCK_MIN_MS or (
            since_punch < LOCK_MAX_MS and arms_busy
        )
        self.posture_locked = posture_locked

        sh_dx, sh_dy = lsh.x - rsh.x, lsh.y - rsh.y
        scale = max(math.hypot(sh_dx, sh_dy), 0.06)
        sh_mid_x = (lsh.x + rsh.x) / 2
        sh_mid_y = (lsh.y + rsh.y) / 2

        if self.s_scale == 0:
            self.s_scale = scale
        elif not posture_locked:
            tau = SCALE_TAU_UP if scale > self.s_scale else SCALE_TAU_DOWN
            self.s_scale += (scale - self.s_scale) * (1 - math.exp(-self.dt_pose / tau))

        if not self.neutral_ready:
            if self.calib_t0 is None:
                self.calib_t0 = now_ms
                self.calib_n = 0
            self.calib_n += 1
            if now_ms - self.calib_t0 >= CALIB_MS and self.calib_n >= 10:
                self.neutral_ready = True

        if wl is not None and len(wl) > R_WR:
            world = lambda i: wl[i]
        else:
            m2 = SHOULDER_M / scale

            class FallbackPoint:
                def __init__(self, p):
                    self.x = (p.x - sh_mid_x) * m2
                    self.y = (p.y - sh_mid_y) * m2
                    self.z = p.z * m2

            world = lambda i: FallbackPoint(lm[i])

        w_sh = max(dist3(world(L_SH), world(R_SH)), 0.15)
        k_l = arm_kinematics(self.arms["L"], world(L_SH), world(L_EL), world(L_WR), w_sh, now_ms)
        k_r = arm_kinematics(self.arms["R"], world(R_SH), world(R_EL), world(R_WR), w_sh, now_ms)

        v_l, v_r = abs(k_l.vx), abs(k_r.vx)
        sweeping = (
            k_l.vx * k_r.vx > 0
            and min(v_l, v_r) > 0.8
            and min(v_l, v_r) > 0.6 * max(v_l, v_r)
        )

        if self.neutral_ready and not sweeping:
            candidates = []
            if arm_vis["L"]:
                candidates.append(self.try_punch(k_l, now_ms))
            if arm_vis["R"]:
                candidates.append(self.try_punch(k_r, now_ms))
            candidates = [c for c in candidates if c is not None]
            if candidates:
                best = max(candidates, key=lambda e: e["speed_ms"])
                self.last_punch_any = now_ms
                self.events.append(best)
                fired.append(best)

        for st in self.arms.values():
            if st.armed and now_ms - st.arm_t > PUNCH_WINDOW_MS:
                st.armed = False
                self.windows_expired += 1

        return fired

    def try_punch(self, k, now_ms):
        st = self.arms[k.side]
        if not st.armed and k.speed > PUNCH_ARM and k.d_reach > PUNCH_EXTEND:
            st.armed = True
            st.arm_t = now_ms
            st.peak = 0.0
            st.reach0 = k.reach_n
            st.windows_opened += 1
        if not st.armed:
            return None
        if now_ms - st.arm_t > PUNCH_WINDOW_MS:
            st.armed = False
            self.windows_expired += 1
            return None
        if k.speed > st.peak:
            st.peak = k.speed
            st.pvx = k.vx
            st.pvy = k.vy
            st.pelbow = k.elbow
        if st.peak < PUNCH_SPEED:
            return None
        if k.reach_n < PUNCH_REACH_N and (k.reach_n - st.reach0) < PUNCH_GROW_N:
            return None
        if (
            now_ms - st.last_punch < PUNCH_CD_MS
            or now_ms - self.last_punch_any < PUNCH_CD_ANY_MS
        ):
            st.armed = False
            return None
        st.armed = False
        st.last_punch = now_ms
        st.windows_fired += 1
        kind, margin = classify_punch(st.peak, st.pvx, st.pvy, st.pelbow)
        return {
            "t_ms": now_ms,
            "frame": None,
            "side": k.side,
            "action": PUNCH_NAME[k.side][kind],
            "kind": kind,
            "speed_ms": round(st.peak, 3),
            "speed_kmh": round(st.peak * 3.6, 2),
            "reach_n": round(k.reach_n, 3),
            "elbow_deg": round(st.pelbow, 1),
            "vx": round(st.pvx, 3),
            "vy": round(st.pvy, 3),
            "conf_margin": round(margin, 3),
        }


def draw_overlay(frame, lm, fired, hud_lines, label_until, now_ms):
    h, w = frame.shape[:2]

    if lm is not None and len(lm) > R_WR:
        def pt(i):
            p = lm[i]
            return int(p.x * w), int(p.y * h)

        for a, b in SKELETON:
            cv2.line(frame, pt(NODE_IDS[a]), pt(NODE_IDS[b]), (200, 200, 200), 2)
        for nid in NODE_IDS:
            cv2.circle(frame, pt(nid), 5, (60, 220, 60), -1)

    y = 26
    for line in hud_lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
        y += 22

    if now_ms < label_until:
        for ev in fired[-1:]:
            text = f"{ev['action']}  {ev['speed_kmh']:.1f} km/h"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            x = (w - size[0]) // 2
            yy = int(h * 0.16)
            cv2.putText(frame, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
            cv2.putText(
                frame, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 255, 120), 2
            )
    return frame


def summarize(events, coverage, pose_fps_mean, duration_s, windows_opened, windows_expired):
    summary = {
        "video_duration_s": round(duration_s, 2),
        "pose_coverage_pct": round(coverage * 100, 1),
        "pose_fps_mean": round(pose_fps_mean, 1),
        "total_punches": len(events),
        "windows_opened": windows_opened,
        "windows_expired_without_fire": windows_expired,
        "by_action": {},
        "by_side": {"L": 0, "R": 0},
    }
    speeds = [e["speed_kmh"] for e in events]
    if speeds:
        summary["speed_kmh"] = {
            "mean": round(statistics.mean(speeds), 2),
            "median": round(statistics.median(speeds), 2),
            "max": round(max(speeds), 2),
            "min": round(min(speeds), 2),
        }
        intervals = [
            b["t_ms"] - a["t_ms"] for a, b in zip(events, events[1:]) if b["t_ms"] > a["t_ms"]
        ]
        if intervals:
            summary["cadence_ms"] = {
                "mean": round(statistics.mean(intervals), 1),
                "median": round(statistics.median(intervals), 1),
                "min": round(min(intervals), 1),
            }
        summary["ppm"] = round(len(events) / duration_s * 60, 1)
    for e in events:
        summary["by_side"][e["side"]] += 1
        entry = summary["by_action"].setdefault(
            e["action"], {"count": 0, "speed_kmh": [], "reach_n": [], "elbow_deg": []}
        )
        entry["count"] += 1
        entry["speed_kmh"].append(e["speed_kmh"])
        entry["reach_n"].append(e["reach_n"])
        entry["elbow_deg"].append(e["elbow_deg"])
    for action, entry in summary["by_action"].items():
        entry["speed_kmh_mean"] = round(statistics.mean(entry.pop("speed_kmh")), 2)
        entry["reach_n_mean"] = round(statistics.mean(entry.pop("reach_n")), 3)
        entry["elbow_deg_mean"] = round(statistics.mean(entry.pop("elbow_deg")), 1)

    straight = [e for e in events if e["kind"] == "STRAIGHT"]
    hooks = [e for e in events if e["kind"] == "HOOK"]
    uppers = [e for e in events if e["kind"] == "UPPERCUT"]

    def rate(sub, pred):
        return round(100.0 * sum(1 for e in sub if pred(e)) / len(sub), 1) if sub else None

    summary["form_checks_pct"] = {
        "straight_full_extension(elbow>150)": rate(straight, lambda e: e["elbow_deg"] > 150),
        "hook_low_elbow(<=165)": rate(hooks, lambda e: e["elbow_deg"] <= 165),
        "uppercut_low_elbow(<=155)": rate(uppers, lambda e: e["elbow_deg"] <= 155),
        "fired_at_or_beyond_095_reach": rate(
            events, lambda e: e["reach_n"] >= PUNCH_REACH_N
        ),
        "confident_classification(margin>=0.15)": rate(
            events, lambda e: e["conf_margin"] >= 0.15
        ),
    }
    return summary


def print_report(summary, events):
    print("=" * 62)
    print(f"펀치 평가 리포트  |  총 {summary['total_punches']}발")
    print("=" * 62)
    print(
        f"영상 {summary['video_duration_s']}s · 포즈 커버리지 {summary['pose_coverage_pct']}% · "
        f"평균 {summary['pose_fps_mean']} fps · 페이스 {summary.get('ppm', 0)}발/분"
    )
    print(
        f"창(window) 열림 {summary['windows_opened']}회 → 발사 {summary['total_punches']}회 "
        f"(미발사 만료 {summary['windows_expired_without_fire']}회)"
    )
    print("-" * 62)
    header = f"{'기술':<15}{'횟수':>5}{'평균속도':>11}{'뻗음':>8}{'팔꿈치':>9}"
    print(header)
    for action in sorted(summary["by_action"]):
        e = summary["by_action"][action]
        print(
            f"{action:<15}{e['count']:>5}{e['speed_kmh_mean']:>9.1f}km/h"
            f"{e['reach_n_mean']:>8.2f}x{e['elbow_deg_mean']:>8.1f}deg"
        )
    sp = summary.get("speed_kmh")
    if sp:
        print("-" * 62)
        print(
            f"속도 km/h — 평균 {sp['mean']} · 중앙값 {sp['median']} · 최대 {sp['max']} · 최소 {sp['min']}"
        )
        cad = summary.get("cadence_ms")
        if cad:
            print(
                f"연타 간격 ms — 평균 {cad['mean']} · 중앙값 {cad['median']} · 최소 {cad['min']}"
            )
    print("-" * 62)
    print("형태 정확도:")
    for name, val in summary["form_checks_pct"].items():
        shown = "-" if val is None else f"{val}%"
        print(f"  {name:<40} {shown:>7}")

    scoring = summary.get("scoring")
    if scoring:
        print("-" * 62)
        print(
            f"정답 대비 채점 (허용오차 ±{scoring['tolerance_ms']}ms) — "
            f"정답 {scoring['ground_truth']}발 / 예측 {scoring['predicted']}발"
        )
        print(
            f"  TP {scoring['tp']} · FP {scoring['fp']} · FN {scoring['fn']} | "
            f"정밀도 {scoring['precision']} · 재현율 {scoring['recall']} · F1 {scoring['f1']}"
        )
        print(
            f"  종류 정확도 {scoring['kind_accuracy']} · 팔 정확도 {scoring['side_accuracy']} · "
            f"타이밍 오차 평균 {scoring['timing_error_ms_mean']}ms"
        )
        if scoring["confusion"]:
            print(f"  혼동: {scoring['confusion']}")
    print("-" * 62)
    print("타격 로그:")
    for e in events:
        t = e["t_ms"] / 1000.0
        print(
            f"  [{t:7.2f}s] {e['action']:<15} {e['speed_kmh']:6.1f}km/h "
            f"reach={e['reach_n']:.2f}x elbow={e['elbow_deg']:.0f}deg margin={e['conf_margin']:+.2f}"
        )


def apply_tune_config(config_path):
    global PUNCH_ARM, PUNCH_EXTEND, PUNCH_SPEED, PUNCH_REACH_N, PUNCH_GROW_N
    global PUNCH_WINDOW_MS, PUNCH_CD_MS, PUNCH_CD_ANY_MS, UPPERCUT_VY, UPPERCUT_ELBOW
    global HOOK_VX, HOOK_ELBOW, LOCK_MIN_MS, LOCK_MAX_MS
    if not config_path or not Path(config_path).exists():
        return
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    tune = data.get("tune", data)
    PUNCH_ARM = float(tune.get("PUNCH_ARM", PUNCH_ARM))
    PUNCH_EXTEND = float(tune.get("PUNCH_EXTEND", PUNCH_EXTEND))
    PUNCH_SPEED = float(tune.get("PUNCH_SPEED", PUNCH_SPEED))
    PUNCH_REACH_N = float(tune.get("PUNCH_REACH_N", PUNCH_REACH_N))
    PUNCH_GROW_N = float(tune.get("PUNCH_GROW_N", PUNCH_GROW_N))
    PUNCH_WINDOW_MS = float(tune.get("PUNCH_WINDOW_MS", PUNCH_WINDOW_MS))
    PUNCH_CD_MS = float(tune.get("PUNCH_CD_MS", PUNCH_CD_MS))
    PUNCH_CD_ANY_MS = float(tune.get("PUNCH_CD_ANY_MS", PUNCH_CD_ANY_MS))
    UPPERCUT_VY = float(tune.get("UPPERCUT_VY", UPPERCUT_VY))
    UPPERCUT_ELBOW = float(tune.get("UPPERCUT_ELBOW", UPPERCUT_ELBOW))
    HOOK_VX = float(tune.get("HOOK_VX", HOOK_VX))
    HOOK_ELBOW = float(tune.get("HOOK_ELBOW", HOOK_ELBOW))
    LOCK_MIN_MS = float(tune.get("LOCK_MIN_MS", LOCK_MIN_MS))
    LOCK_MAX_MS = float(tune.get("LOCK_MAX_MS", LOCK_MAX_MS))
    print(f"🔧 [Config Applied] {Path(config_path).name} (SPEED={PUNCH_SPEED}, EXTEND={PUNCH_EXTEND})")


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate punches with the iter3 runtime pipeline (video or cached landmarks JSONL)"
    )
    ap.add_argument("video", nargs="?", default=None, help="input video file")
    ap.add_argument("--landmarks", default=None, help="cached landmarks JSONL instead of video (see extract_landmarks.py)")
    ap.add_argument("--config", default=None, help="Path to tune config JSON")
    ap.add_argument("--labels", default=None, help="labels.json for accuracy scoring (default: auto-discover next to input)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out-dir", default=None, help="output directory (default: iter3/eval/output/<stem>)")
    ap.add_argument("--annotate", nargs="?", const="AUTO", help="write annotated mp4 (optional path)")
    ap.add_argument("--start", type=float, default=0.0, help="analysis start second")
    ap.add_argument("--end", type=float, default=None, help="analysis end second")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--no-calib", action="store_true", help="skip neutral stance calibration window")
    args = ap.parse_args()

    if args.config:
        apply_tune_config(args.config)

    if not args.video and not args.landmarks:
        print("영상 경로 또는 --landmarks JSONL 중 하나는 필수")
        raise SystemExit(1)

    model_needed = bool(args.video)
    model_path = Path(args.model)
    if model_needed and not model_path.exists():
        print(f"모델 파일 없음: {model_path}")
        raise SystemExit(1)

    video_path = None
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"영상 파일 없음: {video_path}")
            raise SystemExit(1)
    elif not Path(args.landmarks).exists():
        print(f"랜드마크 파일 없음: {args.landmarks}")
        raise SystemExit(1)

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else Path(__file__).parent / "output" / (video_path.stem if video_path else Path(args.landmarks).parent.name)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    start_ms = args.start * 1000.0
    end_ms = args.end * 1000.0 if args.end is not None else float("inf")

    writer = None
    annotate_path = None
    if args.annotate:
        if not video_path:
            print("--annotate는 영상 입력에서만 지원")
            raise SystemExit(1)
        annotate_path = out_dir / "annotated.mp4" if args.annotate == "AUTO" else Path(args.annotate)
        _require_cv()
        cap_meta = cv2.VideoCapture(str(video_path))
        src_fps = (cap_meta.get(cv2.CAP_PROP_FPS) or 30.0) / max(args.stride, 1)
        width = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_meta.release()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotate_path), fourcc, src_fps, (width, height))

    evaluator = PunchEvaluator(calib=not args.no_calib)
    label_until = -1
    detected_frames = 0
    seen_frames = 0
    fps_samples = []
    prev_ts = None

    def handle_frame(ts_float, lm, wl, frame):
        nonlocal label_until, detected_frames, seen_frames, prev_ts
        ts_ms = int(ts_float)
        if prev_ts is not None and ts_float > prev_ts:
            fps_samples.append(1000.0 / (ts_float - prev_ts))
        prev_ts = ts_float
        seen_frames += 1
        if lm is not None and wl is not None and len(wl) > R_WR:
            detected_frames += 1

        fired = evaluator.process(lm, wl, ts_ms)
        for ev in fired:
            ev["frame"] = seen_frames
            label_until = ts_ms + 700

        if writer is not None:
            hud = [
                f"t={ts_ms / 1000.0:.2f}s  punches={len(evaluator.events)}",
                f"L spd={evaluator.arms['L'].last_speed:.1f}m/s reach={evaluator.arms['L'].last_reach_n:.2f}x",
                f"R spd={evaluator.arms['R'].last_speed:.1f}m/s reach={evaluator.arms['R'].last_reach_n:.2f}x",
                f"calib={'OK' if evaluator.neutral_ready else '...'}",
            ]
            frame = draw_overlay(frame, lm, fired, hud, label_until, ts_ms)
            writer.write(frame)

    if video_path:
        _require_cv()
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
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_no = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts_float = cap.get(cv2.CAP_PROP_POS_MSEC)
                frame_no += 1
                if ts_float < start_ms or ts_float > end_ms:
                    continue
                if (frame_no - 1) % args.stride != 0:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, int(ts_float))
                lm = result.pose_landmarks[0] if result.pose_landmarks else None
                wl = result.pose_world_landmarks[0] if result.pose_world_landmarks else []
                handle_frame(ts_float, lm, wl, frame)
        cap.release()
    else:
        for ts_ms, lm, wl in iter_landmarks_jsonl(args.landmarks, start_ms, end_ms):
            handle_frame(ts_ms, lm, wl, None)

    if writer is not None:
        writer.release()

    duration_s = min(prev_ts if prev_ts is not None else 0.0, end_ms) / 1000.0 - args.start
    windows_opened = sum(st.windows_opened for st in evaluator.arms.values())
    windows_expired = evaluator.windows_expired
    coverage = detected_frames / seen_frames if seen_frames else 0.0
    fps_mean = statistics.mean(fps_samples) if fps_samples else 0.0

    events = evaluator.events
    summary = summarize(events, coverage, fps_mean, duration_s, windows_opened, windows_expired)

    labels_path = Path(args.labels) if args.labels else None
    if labels_path is None:
        src = Path(args.landmarks) if args.landmarks else video_path
        candidate = src.parent / "labels.json"
        if candidate.exists():
            labels_path = candidate
    scoring = None
    if labels_path is not None and Path(labels_path).exists():
        labels = load_labels_file(labels_path)
        scoring = score_predictions(events, labels)
        summary["scoring"] = scoring

    json_path = out_dir / "report.json"
    try:
        json_path.write_text(json.dumps({"summary": summary, "punches": events}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        tmp_json = out_dir / "report_tmp.json"
        tmp_json.write_text(json.dumps({"summary": summary, "punches": events}, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.move(str(tmp_json), str(json_path))

    csv_path = out_dir / "punches.csv"
    cols = ["t_ms", "frame", "side", "action", "speed_ms", "speed_kmh", "reach_n", "elbow_deg", "vx", "vy", "conf_margin"]
    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            wcsv = csv.DictWriter(f, fieldnames=cols)
            wcsv.writeheader()
            for e in events:
                wcsv.writerow({c: e[c] for c in cols})
    except Exception as err:
        import os
        tmp_csv = out_dir / f"punches_{os.getpid()}.csv"
        with tmp_csv.open("w", newline="", encoding="utf-8-sig") as f:
            wcsv = csv.DictWriter(f, fieldnames=cols)
            wcsv.writeheader()
            for e in events:
                wcsv.writerow({c: e[c] for c in cols})
        try:
            if csv_path.exists():
                csv_path.unlink()
            tmp_csv.rename(csv_path)
        except Exception:
            csv_path = tmp_csv

    print_report(summary, events)
    print("-" * 62)
    print(f"결과 저장: {json_path}")
    print(f"           {csv_path}")
    if annotate_path is not None:
        print(f"           {annotate_path}")


if __name__ == "__main__":
    main()
