"""Run full client action recognition (Punches + Footwork + Guard + Rotation) on a video.

Replicates the complete runtime decision engine of `fighter_client.html` + `punch_core.js`
on an offline video file and outputs a detailed action timeline and summary metrics.

Usage:
  conda activate pjt-4
  python iter4/eval/evaluate_full_actions.py iter4/eval/video/benchmark.mp4
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

cv2 = None
mp = None
BaseOptions = None
vision = None

def _require_cv():
    global cv2, mp, BaseOptions, vision
    if cv2 is not None:
        return
    try:
        import cv2 as _cv2
        import mediapipe as _mp
        from mediapipe.tasks.python import BaseOptions as _BaseOptions
        from mediapipe.tasks.python import vision as _vision
    except ImportError as exc:
        raise SystemExit(f"영상 처리에는 opencv-python과 mediapipe가 필요합니다: {exc}") from exc
    cv2, mp, BaseOptions, vision = _cv2, _mp, _BaseOptions, _vision

# Landmark indices
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16
NODE_IDS = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR]
SKELETON = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6)]

# Tunings synchronized with fighter_client.html & punch_core.js
TUNE = {
    "PUNCH_ARM": 1.0,
    "PUNCH_EXTEND": 0.40,
    "PUNCH_SPEED": 1.6,
    "PUNCH_REACH_N": 0.88,
    "PUNCH_GROW_N": 0.28,
    "PUNCH_WINDOW_MS": 380,
    "PUNCH_CD_MS": 400,
    "PUNCH_CD_ANY_MS": 200,
    "UPPERCUT_VY": 0.55,
    "UPPERCUT_ELBOW": 150,
    "HOOK_VX": 0.56,
    "HOOK_ELBOW": 158,
    "PUNCH_LOCK": 180,
    "PUNCH_LOCK_MAX": 1100,
    "PUNCH_MOVE_DECAY": 1.2,
    "ROLL_ON": 12.0,
    "ROLL_OFF": 8.0,
    "ROLL_FLAT": 8.0,
    "ROLL_RANGE": 14.0,
    "PITCH_ON": 0.16,
    "PITCH_OFF": 0.11,
    "PITCH_RANGE": 0.14,
    "PITCH_BACK_ON": 0.12,
    "PITCH_BACK_OFF": 0.07,
    "PITCH_BACK_RANGE": 0.12,
    "SHIFT_ON": 0.22,
    "SHIFT_OFF": 0.14,
    "SHIFT_RANGE": 0.22,
    "VOTE_WINDOW_MS": 160,
    "GUARD_HOLD_MS": 120,
    "CALIB_MS": 1800,
    "SCALE_TAU_UP": 0.12,
    "SCALE_TAU_DOWN": 0.45,
}

PUNCH_NAME = {
    "L": {"STRAIGHT": "LEFT_JAB", "HOOK": "LEFT_HOOK", "UPPERCUT": "LEFT_UPPERCUT"},
    "R": {"STRAIGHT": "RIGHT_CROSS", "HOOK": "RIGHT_HOOK", "UPPERCUT": "RIGHT_UPPERCUT"},
}

DEFAULT_MODEL = Path(__file__).parent / "models" / "pose_landmarker_full.task"


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


def clamp01(v):
    return max(0.0, min(1.0, v))


class VoteWindow:
    def __init__(self, window_ms=160):
        self.window_ms = window_ms
        self.buf = []

    def vote(self, cand, current, now_ms):
        self.buf.append({"val": cand, "t": now_ms})
        self.buf = [x for x in self.buf if now_ms - x["t"] <= self.window_ms]
        counts = {}
        for x in self.buf:
            counts[x["val"]] = counts.get(x["val"], 0) + 1
        majority = len(self.buf) * 0.55
        for k, c in counts.items():
            if c >= majority and k != "NONE":
                return k
        if counts.get("NONE", 0) >= majority:
            return "NONE"
        return current


class FullActionEvaluator:
    def __init__(self):
        self.arms = {
            "L": {"x": 0, "y": 0, "z": 0, "reach": 0, "t": None, "armed": False, "arm_t": 0, "peak": 0, "reach0": 0, "pvx": 0, "pvy": 0, "pelbow": 180, "last_punch": -1e9, "last_speed": 0, "last_reach_n": 0},
            "R": {"x": 0, "y": 0, "z": 0, "reach": 0, "t": None, "armed": False, "arm_t": 0, "peak": 0, "reach0": 0, "pvx": 0, "pvy": 0, "pelbow": 180, "last_punch": -1e9, "last_speed": 0, "last_reach_n": 0},
        }
        self.last_punch_any = -1e9
        self.s_roll = 0.0
        self.s_pitch = 0.0
        self.s_shift = 0.0
        self.s_scale = 0.0
        self.move_state = "NONE"
        self.rot_state = "NONE"
        self.move_intensity = 0.0
        self.rot_intensity = 0.0
        self.move_vote = VoteWindow(TUNE["VOTE_WINDOW_MS"])
        self.rot_vote = VoteWindow(TUNE["VOTE_WINDOW_MS"])
        self.was_locked = False
        self.locked_move = {"state": "NONE", "intensity": 0.0}
        self.guard_active = False
        self.guard_since = 0

        # Neutral Calibration
        self.neutral_ready = False
        self.calib_t0 = None
        self.calib_n = 0
        self.neutral = {"roll": 0.0, "face_rel": 0.0, "fist_rel": 0.0, "sh_mid_y": 0.0, "scale": 0.0, "lwr_off": 0.0, "rwr_off": 0.0}

        self.last_pose_t = None
        self.punch_events = []
        self.timeline = []

    def is_locked(self, now_ms):
        since_punch = now_ms - self.last_punch_any
        arms_busy = (
            max(self.arms["L"]["last_speed"], self.arms["R"]["last_speed"]) > 0.9
            or max(self.arms["L"]["last_reach_n"], self.arms["R"]["last_reach_n"]) > 1.00
        )
        return since_punch < TUNE["PUNCH_LOCK"] or (since_punch < TUNE["PUNCH_LOCK_MAX"] and arms_busy)

    def process_frame(self, lm, wl, now_ms, frame_idx):
        dt_pose = 0.033 if self.last_pose_t is None else min(max((now_ms - self.last_pose_t) / 1000.0, 0.008), 0.4)
        self.last_pose_t = now_ms

        posture_locked = self.is_locked(now_ms)
        fired_punches = []

        if not lm or len(lm) <= R_WR:
            return {"punches": [], "move": "NONE", "move_intensity": 0, "rot": "NONE", "guard": False, "locked": posture_locked}

        nose, lsh, rsh = lm[NOSE], lm[L_SH], lm[R_SH]
        lwr, rwr = lm[L_WR], lm[R_WR]
        lel, rel = lm[L_EL], lm[R_EL]

        sh_dx, sh_dy = lsh.x - rsh.x, lsh.y - rsh.y
        scale = max(math.hypot(sh_dx, sh_dy), 0.06)
        sh_mid_x = (lsh.x + rsh.x) / 2.0
        sh_mid_y = (lsh.y + rsh.y) / 2.0

        if self.s_scale == 0:
            self.s_scale = scale
        elif not posture_locked:
            tau = TUNE["SCALE_TAU_UP"] if scale > self.s_scale else TUNE["SCALE_TAU_DOWN"]
            self.s_scale += (scale - self.s_scale) * (1 - math.exp(-dt_pose / tau))

        u_scale = max(self.s_scale, 0.06)
        roll_raw = math.atan2(sh_dy, abs(sh_dx)) * 180.0 / math.pi
        wr_mid_y = (lwr.y + rwr.y) / 2.0
        face_rel = (nose.y - sh_mid_y) / u_scale
        fist_rel = (wr_mid_y - sh_mid_y) / u_scale
        lwr_off = (lwr.x - sh_mid_x) / u_scale
        rwr_off = (rwr.x - sh_mid_x) / u_scale

        # Calibration
        if not self.neutral_ready:
            if self.calib_t0 is None:
                self.calib_t0 = now_ms
            a = 1.0 if self.calib_n == 0 else 0.2
            self.neutral["roll"] += (roll_raw - self.neutral["roll"]) * a
            self.neutral["face_rel"] += (face_rel - self.neutral["face_rel"]) * a
            self.neutral["fist_rel"] += (fist_rel - self.neutral["fist_rel"]) * a
            self.neutral["sh_mid_y"] += (sh_mid_y - self.neutral["sh_mid_y"]) * a
            self.neutral["scale"] += (self.s_scale - self.neutral["scale"]) * a
            self.neutral["lwr_off"] += (lwr_off - self.neutral["lwr_off"]) * a
            self.neutral["rwr_off"] += (rwr_off - self.neutral["rwr_off"]) * a
            self.calib_n += 1
            if now_ms - self.calib_t0 >= TUNE["CALIB_MS"] and self.calib_n >= 10:
                self.neutral_ready = True

        # Dual Guard Detection
        dl = math.hypot(lwr.x - nose.x, lwr.y - nose.y) / scale
        dr = math.hypot(rwr.x - nose.x, rwr.y - nose.y) / scale
        guard_now = (lwr.y < sh_mid_y + 0.15 * scale) and (rwr.y < sh_mid_y + 0.15 * scale) and (dl < 1.0) and (dr < 1.0)
        if not guard_now:
            self.guard_since = 0
        elif self.guard_since == 0:
            self.guard_since = now_ms
        self.guard_active = self.guard_since > 0 and (now_ms - self.guard_since) >= TUNE["GUARD_HOLD_MS"]

        # Signal Smoothing (Posture & Pitch & Shift)
        if not posture_locked:
            face_score = face_rel - self.neutral["face_rel"]
            fist_score = fist_rel - self.neutral["fist_rel"]
            w_face = 0.65 if self.guard_active else 0.45
            w_fist = 0.05 if self.guard_active else 0.25
            dir_core = w_face * face_score + w_fist * fist_score

            conf = clamp01(dir_core / 0.08) if dir_core > 0 else 0.0
            near_delta = (self.s_scale - self.neutral["scale"]) / max(self.neutral["scale"], 0.06)
            near_term = conf * 1.20 * max(0.0, near_delta) if dir_core > 0 else 0.80 * min(0.0, near_delta)

            body_drop = (sh_mid_y - self.neutral["sh_mid_y"]) / u_scale
            body_boost = conf * 0.30 * max(0.0, body_drop)
            pitch_raw = dir_core + near_term + body_boost

            shift_l = -(lwr_off - self.neutral["lwr_off"])
            shift_r = -(rwr_off - self.neutral["rwr_off"])
            shift_raw = (shift_l + shift_r) / 2.0

            ka = 1 - math.exp(-dt_pose / 0.12)
            self.s_roll += ((roll_raw - self.neutral["roll"]) - self.s_roll) * ka
            self.s_pitch += (pitch_raw - self.s_pitch) * ka
            self.s_shift += (shift_raw - self.s_shift) * ka

        # Kinematics & Punch
        if wl is not None and len(wl) > R_WR:
            world = lambda i: wl[i]
        else:
            m2 = 0.40 / scale

            class FallbackPt:
                def __init__(self, p):
                    self.x = (p.x - sh_mid_x) * m2
                    self.y = (p.y - sh_mid_y) * m2
                    self.z = p.z * m2

            world = lambda i: FallbackPt(lm[i])

        w_sh = max(dist3(world(L_SH), world(R_SH)), 0.15)
        for side, sh_id, el_id, wr_id in (("L", L_SH, L_EL, L_WR), ("R", R_SH, R_EL, R_WR)):
            st = self.arms[side]
            sh, el, wr = world(sh_id), world(el_id), world(wr_id)
            reach = dist3(wr, sh)
            elbow = angle_deg(sh, el, wr)
            vx = vy = vz = speed = d_reach = 0.0
            dt = None if st["t"] is None else (now_ms - st["t"]) / 1000.0
            if dt is not None and 0.008 < dt < 0.4:
                vx = (wr.x - st["x"]) / dt
                vy = (wr.y - st["y"]) / dt
                vz = (wr.z - st["z"]) / dt
                speed = math.hypot(vx, vy, vz)
                d_reach = (reach - st["reach"]) / dt
            st["x"], st["y"], st["z"] = wr.x, wr.y, wr.z
            st["reach"] = reach
            st["t"] = now_ms
            st["last_speed"] = speed
            st["last_reach_n"] = reach / w_sh

            # Punch trigger logic
            if not st["armed"] and speed > TUNE["PUNCH_ARM"] and d_reach > TUNE["PUNCH_EXTEND"]:
                st["armed"] = True
                st["arm_t"] = now_ms
                st["peak"] = 0.0
                st["reach0"] = reach / w_sh

            if st["armed"]:
                if now_ms - st["arm_t"] > TUNE["PUNCH_WINDOW_MS"]:
                    st["armed"] = False
                else:
                    if speed > st["peak"]:
                        st["peak"] = speed
                        st["pvx"] = vx
                        st["pvy"] = vy
                        st["pelbow"] = elbow
                    reach_n = reach / w_sh
                    if st["peak"] >= TUNE["PUNCH_SPEED"] and (reach_n >= TUNE["PUNCH_REACH_N"] or (reach_n - st["reach0"]) >= TUNE["PUNCH_GROW_N"]):
                        if now_ms - st["last_punch"] >= TUNE["PUNCH_CD_MS"] and now_ms - self.last_punch_any >= TUNE["PUNCH_CD_ANY_MS"]:
                            st["armed"] = False
                            st["last_punch"] = now_ms
                            self.last_punch_any = now_ms

                            s = max(st["peak"], 1e-3)
                            up_r = -st["pvy"] / s
                            hk_r = abs(st["pvx"]) / s
                            if up_r > TUNE["UPPERCUT_VY"] and st["pelbow"] < TUNE["UPPERCUT_ELBOW"]:
                                kind = "UPPERCUT"
                            elif hk_r > TUNE["HOOK_VX"] and st["pelbow"] < TUNE["HOOK_ELBOW"]:
                                kind = "HOOK"
                            else:
                                kind = "STRAIGHT"

                            action = PUNCH_NAME[side][kind]
                            p_info = {
                                "t_ms": now_ms,
                                "frame": frame_idx,
                                "side": side,
                                "kind": kind,
                                "action": action,
                                "speed_kmh": round(st["peak"] * 3.6, 1),
                                "reach_n": round(reach_n, 2),
                                "elbow": round(st["pelbow"], 1),
                            }
                            fired_punches.append(p_info)
                            self.punch_events.append(p_info)

        # Update Movement & Footwork States
        locked = posture_locked
        if locked and not self.was_locked:
            self.locked_move["state"] = self.move_state
            self.locked_move["intensity"] = self.move_intensity
        self.was_locked = locked

        if locked:
            self.move_state = self.locked_move["state"]
            self.locked_move["intensity"] *= math.exp(-dt_pose / TUNE["PUNCH_MOVE_DECAY"])
            self.move_intensity = self.locked_move["intensity"]
            self.rot_state = "NONE"
            self.rot_intensity += (0 - self.rot_intensity) * (1 - math.exp(-dt_pose / 0.13))
        else:
            m_cand = "NONE"
            roll_on = TUNE["ROLL_OFF"] if self.move_state in ("LEFT", "RIGHT") else TUNE["ROLL_ON"]
            if self.s_roll > roll_on:
                m_cand = "LEFT"
            elif self.s_roll < -roll_on:
                m_cand = "RIGHT"
            elif abs(self.s_roll) < TUNE["ROLL_FLAT"]:
                fwd_on = TUNE["PITCH_OFF"] if self.move_state == "FORWARD" else TUNE["PITCH_ON"]
                back_on = TUNE["PITCH_BACK_OFF"] if self.move_state == "BACK" else TUNE["PITCH_BACK_ON"]
                if self.s_pitch > fwd_on:
                    m_cand = "FORWARD"
                elif self.s_pitch < -back_on:
                    m_cand = "BACK"
            else:
                m_cand = self.move_state if self.move_state in ("FORWARD", "BACK") else "NONE"
            self.move_state = self.move_vote.vote(m_cand, self.move_state, now_ms)

            r_cand = "NONE"
            if abs(self.s_roll) < TUNE["ROLL_FLAT"]:
                s_on = TUNE["SHIFT_OFF"] if self.rot_state != "NONE" else TUNE["SHIFT_ON"]
                if self.s_shift > s_on:
                    r_cand = "ROT_LEFT"
                elif self.s_shift < -s_on:
                    r_cand = "ROT_RIGHT"
            self.rot_state = self.rot_vote.vote(r_cand, self.rot_state, now_ms)

            ka = 1 - math.exp(-dt_pose / 0.13)
            mi = 0.0
            if self.move_state in ("LEFT", "RIGHT"):
                mi = clamp01((abs(self.s_roll) - TUNE["ROLL_OFF"]) / TUNE["ROLL_RANGE"])
            elif self.move_state == "FORWARD":
                mi = clamp01((self.s_pitch - TUNE["PITCH_OFF"]) / TUNE["PITCH_RANGE"])
            elif self.move_state == "BACK":
                mi = clamp01((-self.s_pitch - TUNE["PITCH_BACK_OFF"]) / TUNE["PITCH_BACK_RANGE"])
            self.move_intensity += (mi - self.move_intensity) * ka

            ri = 0.0 if self.rot_state == "NONE" else clamp01((abs(self.s_shift) - TUNE["SHIFT_OFF"]) / TUNE["SHIFT_RANGE"])
            self.rot_intensity += (ri - self.rot_intensity) * ka

        frame_result = {
            "t_ms": now_ms,
            "frame": frame_idx,
            "punches": fired_punches,
            "move": self.move_state,
            "move_intensity": round(self.move_intensity, 2),
            "rot": self.rot_state,
            "guard": self.guard_active,
            "locked": posture_locked,
            "roll": round(self.s_roll, 1),
            "pitch": round(self.s_pitch, 2),
        }
        self.timeline.append(frame_result)
        return frame_result


def main():
    ap = argparse.ArgumentParser(description="Evaluate complete runtime motion recognition pipeline on video")
    ap.add_argument("video", default="iter4/eval/video/benchmark.mp4", nargs="?")
    ap.add_argument("--annotate", default="iter4/eval/output/annotated_full_pipeline.mp4")
    ap.add_argument("--report", default="iter4/eval/output/full_pipeline_report.json")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ 오류: 비디오 파일을 찾을 수 없습니다: {video_path}")
        sys.exit(1)

    _require_cv()

    print("=" * 65)
    print("🥊 클라이언트 런타임 전체 동작 인식 평가기 구동")
    print("   (펀치 5종 + 전진/후진/좌우 풋워크 + 듀얼 가드 + 잠금 엔진)")
    print("=" * 65)
    print(f"• 비디오 입력: {video_path.resolve()}")

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps

    print(f"• 비디오 정보: {width}x{height} | {total_frames} 프레임 ({duration_s:.1f}초, {fps:.1f} FPS)")

    # Initialize MediaPipe Pose
    base_options = BaseOptions(model_asset_path=str(DEFAULT_MODEL))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    evaluator = FullActionEvaluator()

    writer = None
    if args.annotate:
        out_path = Path(args.annotate)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    t0 = time.time()
    last_punch_banner = {"text": "READY", "t": -1e9, "color": (150, 150, 150)}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now_ms = int(frame_idx * (1000.0 / fps))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_image, now_ms)

        lm = res.pose_landmarks[0] if res.pose_landmarks else None
        wl = res.pose_world_landmarks[0] if res.pose_world_landmarks else None

        info = evaluator.process_frame(lm, wl, now_ms, frame_idx)

        # Update last punch banner
        if info["punches"]:
            p = info["punches"][-1]
            col = (0, 165, 255) if "HOOK" in p["action"] else ((255, 0, 255) if "UPPER" in p["action"] else (0, 255, 0))
            last_punch_banner = {
                "text": f"{p['action']}  {p['speed_kmh']} km/h",
                "t": now_ms,
                "color": col,
            }

        # Draw HUD & Skeleton
        if writer is not None:
            disp = frame.copy()

            # Skeleton
            if lm:
                for a, b in SKELETON:
                    pa, pb = lm[NODE_IDS[a]], lm[NODE_IDS[b]]
                    pt_a = (int(pa.x * width), int(pa.y * height))
                    pt_b = (int(pb.x * width), int(pb.y * height))
                    cv2.line(disp, pt_a, pt_b, (0, 255, 255), 2, cv2.LINE_AA)
                for nid in NODE_IDS:
                    p = lm[nid]
                    pt = (int(p.x * width), int(p.y * height))
                    cv2.circle(disp, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)

            # Top HUD Bar
            cv2.rectangle(disp, (0, 0), (width, 95), (15, 15, 20), -1)
            cv2.line(disp, (0, 95), (width, 95), (60, 60, 70), 1)

            # Time & FPS
            cv2.putText(disp, f"TIME: {now_ms/1000:04.1f}s / {duration_s:04.1f}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)

            # Footwork Move State
            m_col = (0, 255, 255) if info["move"] != "NONE" else (120, 120, 120)
            cv2.putText(disp, f"MOVE: {info['move']} ({info['move_intensity']*100:.0f}%)", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, m_col, 2, cv2.LINE_AA)

            # Guard State
            g_col = (0, 255, 0) if info["guard"] else (100, 100, 100)
            cv2.putText(disp, f"GUARD: {'ON' if info['guard'] else 'OFF'}", (460, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, g_col, 2, cv2.LINE_AA)

            # Punch Lock State
            l_col = (0, 100, 255) if info["locked"] else (100, 100, 100)
            cv2.putText(disp, f"LOCK: {'LOCKED' if info['locked'] else 'FREE'}", (630, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, l_col, 2, cv2.LINE_AA)

            # Bottom Banner: Last Fired Punch
            punch_active = (now_ms - last_punch_banner["t"]) < 600
            p_color = last_punch_banner["color"] if punch_active else (100, 100, 100)
            p_text = f"PUNCH: {last_punch_banner['text']}" if punch_active else "PUNCH: [READY]"
            cv2.putText(disp, p_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85, p_color, 2, cv2.LINE_AA)

            # Internal Angles (Roll / Pitch)
            cv2.putText(disp, f"Roll:{info['roll']:+04.1f}  Pitch:{info['pitch']:+04.2f}", (width - 270, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

            writer.write(disp)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    elapsed_proc = time.time() - t0
    proc_fps = frame_idx / max(elapsed_proc, 1e-3)

    # Aggregate Statistics
    move_counts = {}
    guard_frames = sum(1 for x in evaluator.timeline if x["guard"])
    lock_frames = sum(1 for x in evaluator.timeline if x["locked"])
    for x in evaluator.timeline:
        m = x["move"]
        move_counts[m] = move_counts.get(m, 0) + 1

    punch_counts = {}
    for p in evaluator.punch_events:
        a = p["action"]
        punch_counts[a] = punch_counts.get(a, 0) + 1

    summary = {
        "video_duration_s": round(duration_s, 2),
        "processed_frames": frame_idx,
        "processing_fps": round(proc_fps, 1),
        "total_punches": len(evaluator.punch_events),
        "punches_by_action": punch_counts,
        "footwork_frames": {k: round(v / frame_idx * 100, 1) for k, v in move_counts.items()},
        "guard_coverage_pct": round(guard_frames / frame_idx * 100, 1),
        "punch_lock_pct": round(lock_frames / frame_idx * 100, 1),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"summary": summary, "punches": evaluator.punch_events}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 65)
    print("✅ 전체 런타임 동작인식 파이프라인 분석 완료!")
    print("=" * 65)
    print(f"• 처리 속도: {proc_fps:.1f} FPS ({frame_idx} 프레임 완료)")
    print(f"• 총 검출 펀치: {len(evaluator.punch_events)} 회")
    for act, cnt in sorted(punch_counts.items(), key=lambda x: -x[1]):
        print(f"   - {act:<16}: {cnt:>2} 회")
    print(f"• 풋워크 상태 분포:")
    for mv, pct in move_counts.items():
        print(f"   - {mv:<10}: {pct:>4} 프레임 ({pct/frame_idx*100:4.1f}%)")
    print(f"• 듀얼 가드 지속: {guard_frames} 프레임 ({guard_frames/frame_idx*100:4.1f}%)")
    print("-" * 65)
    print(f"• 전체 결과 리포트: {report_path.resolve()}")
    if args.annotate:
        print(f"• HUD 시각화 비디오: {Path(args.annotate).resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
