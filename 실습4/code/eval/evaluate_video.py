"""Offline punch evaluation on a video file.

Ports the runtime punch pipeline of iter4/server/templates/fighter_client.html
(MediaPipe Pose upper-body 7 nodes + window-latch punch detector) to Python,
runs it frame-by-frame on an input video, and reports punch counts, types,
speeds and form metrics.

Usage:
  conda activate pjt-4
  python iter4/eval/evaluate_video.py video/benchmark.mp4
  python iter4/eval/evaluate_video.py video/benchmark.mp4 --annotate out.mp4
"""
import argparse
import csv
import json
import math
import statistics
import sys
import shutil
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
TCN_MIN_CONF = 0.35

# v6b: --engine tcn_trigger 전용 (TCNTriggerEvaluator). 룰베이스 키네마틱 임계값과는
# 완전히 독립적인, "모델 출력의 시간적 안정성"만으로 트리거를 결정하는 파라미터.
TCN_TRIGGER_CONFIRM_FRAMES = 2
TCN_TRIGGER_MIN_CONF = 0.5
TCN_TRIGGER_COOLDOWN_MS = 350

# v5c: --engine tcn_hybrid 전용 (TCNHybridTriggerEvaluator). TCNTriggerEvaluator와 동일한
# confirm/edge/cooldown 조건에 "그 순간 해당 side가 실제로 물리적으로 움직이고 있는가"라는
# AND 게이트를 추가한다. 룰베이스가 창을 여는 데 쓰는 PUNCH_ARM/PUNCH_EXTEND와 같은 하한선을
# 그대로 재사용한다 — "확정 피크"(PUNCH_SPEED/PUNCH_REACH_N)까지 요구하면 TCN이 확정되는
# 프레임과 물리적 피크 프레임이 어긋날 때 정당한 펀치까지 게이트에서 막히기 때문이다.
TCN_HYBRID_ARM_GATE = 1.0
TCN_HYBRID_EXTEND_GATE = 0.40

PUNCH_NAME = {
    "L": {"STRAIGHT": "LEFT_JAB", "HOOK": "LEFT_HOOK", "UPPERCUT": "LEFT_UPPERCUT"},
    "R": {"STRAIGHT": "RIGHT_CROSS", "HOOK": "RIGHT_HOOK", "UPPERCUT": "RIGHT_UPPERCUT"},
}

DEFAULT_MODEL = Path(__file__).with_name("models") / "pose_landmarker_full.task"


# ==================== TCN 딥러닝 모션 분류기 ====================
class TCNMotionClassifier:
    """Offline PyTorch Causal TCN Classifier for Punch Kind Recognition."""
    CLASSES = [
        "IDLE", "LEFT_JAB", "RIGHT_JAB", "LEFT_HOOK", "RIGHT_HOOK",
        "LEFT_UPPERCUT", "RIGHT_UPPERCUT", "TWO_HAND_GUARD", "ENERGY_WAVE", "OTHER",
    ]
    PUNCH_CLASSES = {
        "LEFT_JAB", "RIGHT_JAB", "LEFT_HOOK", "RIGHT_HOOK",
        "LEFT_UPPERCUT", "RIGHT_UPPERCUT"
    }

    def __init__(self, model_dir: Path = None):
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "motion_learning"
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.buffer = []  # list of 17-dim arrays
        self.seq_len = 60
        self.dim = 17
        self.prev_l = None
        self.prev_r = None
        self.prev_t = 0
        self._load()

    def _load(self):
        try:
            import torch
            # tcn_model.py (및 그 자신의 `from real_data import CLASSES`)는 항상
            # motion_learning/ 에 있다. --tcn-model-dir 로 실험용 가중치만 든 디렉터리
            # (예: v6b_tcn_trigger/, overfit_hong/) 를 가리켜도 모듈을 찾을 수 있도록
            # 정식 경로를 항상 같이 넣는다.
            canonical_dir = Path(__file__).parent.parent / "motion_learning"
            sys.path.insert(0, str(canonical_dir))
            sys.path.insert(0, str(self.model_dir))
            from tcn_model import CausalMotionTCN
            
            pth_path = self.model_dir / "boxing_tcn.pth"
            scaler_path = self.model_dir / "boxing_tcn_scaler.json"
            
            if not pth_path.exists() or not scaler_path.exists():
                print(f"⚠️ TCN 가중치/스케일러 파일 없음: {pth_path}")
                return

            self.scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
            self.model = CausalMotionTCN(input_dim=17, num_classes=len(self.CLASSES))
            ckpt = torch.load(str(pth_path), map_location="cpu")
            self.model.load_state_dict(ckpt)
            self.model.eval()
            print(f"🧠 [TCN Engine] PyTorch Causal TCN 모델 로드 완료 ({pth_path.name})")
        except Exception as e:
            print(f"⚠️ TCN 로드 실패 (룰베이스로 폴백): {e}")
            self.model = None

    def push(self, lm, now_ms):
        if not lm or len(lm) <= R_WR:
            return
        # 17차원 heuristic_7j_v1 피처 추출
        nose, lsh, rsh = lm[NOSE], lm[L_SH], lm[R_SH]
        lel, rel, lwr, rwr = lm[L_EL], lm[R_EL], lm[L_WR], lm[R_WR]

        sh2d = max(math.hypot(lsh.x - rsh.x, lsh.y - rsh.y), 1e-3)
        l_el_ratio = angle_deg(lsh, lel, lwr) / 180.0
        r_el_ratio = angle_deg(rsh, rel, rwr) / 180.0
        l_reach = dist3(lwr, lsh) / sh2d
        r_reach = dist3(rwr, rsh) / sh2d
        hands_dist = dist3(lwr, rwr) / sh2d
        l_wrist_nose = dist3(lwr, nose) / sh2d
        r_wrist_nose = dist3(rwr, nose) / sh2d
        elbow_dist = dist3(lel, rel) / sh2d
        avg_wrist_z = ((lwr.z + rwr.z) / 2.0) / sh2d

        lvx, lvy, lvz = 0.0, 0.0, 0.0
        rvx, rvy, rvz = 0.0, 0.0, 0.0
        dt = (now_ms - self.prev_t) / 1000.0 if self.prev_t else 0.0
        if self.prev_l and 0.008 < dt < 0.4:
            lvx = (lwr.x - self.prev_l[0]) / dt / sh2d
            lvy = (lwr.y - self.prev_l[1]) / dt / sh2d
            lvz = (lwr.z - self.prev_l[2]) / dt / sh2d
            rvx = (rwr.x - self.prev_r[0]) / dt / sh2d
            rvy = (rwr.y - self.prev_r[1]) / dt / sh2d
            rvz = (rwr.z - self.prev_r[2]) / dt / sh2d

        self.prev_l = (lwr.x, lwr.y, lwr.z)
        self.prev_r = (rwr.x, rwr.y, rwr.z)
        self.prev_t = now_ms

        l_speed = math.hypot(lvx, lvy, lvz)
        r_speed = math.hypot(rvx, rvy, rvz)

        feat17 = [
            l_el_ratio, r_el_ratio, l_reach, r_reach,
            lvx, lvy, lvz, rvx, rvy, rvz,
            l_speed, r_speed, hands_dist, l_wrist_nose, r_wrist_nose, elbow_dist, avg_wrist_z
        ]
        self.buffer.append(feat17)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

    def guess_punch_kind(self, side: str):
        if not self.model or len(self.buffer) == 0:
            return None, 0.0
        import torch
        # Build normalized input tensor
        median = self.scaler["median"]
        scale = self.scaler["scale"]
        clip = self.scaler.get("clip", 5.0)

        n = len(self.buffer)
        data = []
        for t in range(self.seq_len):
            src_idx = max(0, t - (self.seq_len - n))
            src = self.buffer[src_idx]
            norm_f = []
            for f in range(self.dim):
                v = (src[f] - median[f]) / scale[f]
                v = max(-clip, min(clip, v))
                norm_f.append(v)
            data.append(norm_f)

        x = torch.tensor([data], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)[0]
            probs = torch.softmax(logits, dim=-1).numpy()

        best_idx = int(probs.argmax())
        best_label = self.CLASSES[best_idx]
        best_prob = float(probs[best_idx])

        want_side = "LEFT_" if side == "L" else "RIGHT_"
        if best_label in self.PUNCH_CLASSES and best_label.startswith(want_side):
            # Map RIGHT_JAB -> STRAIGHT, LEFT_JAB -> STRAIGHT, HOOK -> HOOK, UPPERCUT -> UPPERCUT
            raw_kind = best_label.replace(want_side, "")
            if raw_kind == "JAB":
                kind = "STRAIGHT"
            elif raw_kind in ("HOOK", "UPPERCUT"):
                kind = raw_kind
            else:
                kind = "STRAIGHT"
            return kind, best_prob
        return None, best_prob


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
    def __init__(self, calib=True, tcn_classifier=None):
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
        self.tcn_classifier = tcn_classifier

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
        if self.posture_locked:
            st.armed = False
            return None
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
        
        # 1) 기본 룰베이스 키네마틱스 분류
        kind, margin = classify_punch(st.peak, st.pvx, st.pvy, st.pelbow)
        
        # 2) TCN 딥러닝 모드 활성화 시 Causal TCN 추론 결과로 재분류
        if self.tcn_classifier:
            tcn_kind, tcn_prob = self.tcn_classifier.guess_punch_kind(k.side)
            if tcn_kind is not None and tcn_prob >= TCN_MIN_CONF:
                kind = tcn_kind
                margin = tcn_prob

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


class TCNTriggerEvaluator:
    """v6b: TCN이 "펀치 종류"뿐 아니라 "펀치가 지금 나가는가(트리거)"까지 직접 결정한다.

    PunchEvaluator/try_punch() 의 PUNCH_ARM·PUNCH_SPEED·PUNCH_REACH_N 같은 물리 임계값을
    **전혀 쓰지 않는다.** 대신 매 프레임 CausalMotionTCN의 10-클래스 softmax 예측만으로:

      1) **확정(confirm)**: 같은 펀치 클래스가 CONFIRM_FRAMES 프레임 연속으로, 매번
         MIN_CONF 이상의 확신도로 나와야 "확정"된다 (한 프레임의 noise로 즉발 트리거되지 않음).
      2) **엣지 트리거(edge-fire)**: 확정되는 바로 그 프레임에서 1회만 이벤트를 발사한다.
         확정된 클래스가 계속 유지돼도 다시 발사하지 않는다 — `already_fired_this_streak` 로
         막는다. 이게 없으면 DEVLOG에 기록된 "같은 라벨을 계속 예측하면 쿨다운마다 무한
         재발동" 버그가 재현된다.
      3) **스트릭 리셋**: 예측이 펀치 클래스가 아니거나 확신도가 MIN_CONF 밑으로 떨어지면
         스트릭이 끝난 것으로 보고, 다음에 다시 확정되면 새 이벤트로 발사를 허용한다.
      4) **전역 쿨다운**: 그래도 최소 COOLDOWN_MS 안에는 새 이벤트를 막아, 스트릭이 아주
         짧게 깨졌다가 같은 스윙으로 바로 재확정되는 경우의 중복 발사를 한 번 더 막는다.

    이 네 가지가 전부 "모델 출력 자체의 시간적 안정성"만으로 트리거를 결정하며, 손목 속도·
    팔꿈치 각도 같은 키네마틱 값은 어디에도 쓰이지 않는다.
    """

    PUNCH_CLASSES = {
        "LEFT_JAB", "RIGHT_JAB", "LEFT_HOOK", "RIGHT_HOOK",
        "LEFT_UPPERCUT", "RIGHT_UPPERCUT",
    }
    KIND_OF = {"JAB": "STRAIGHT", "HOOK": "HOOK", "UPPERCUT": "UPPERCUT"}

    def __init__(self, model, scaler, classes, confirm_frames=2, min_conf=0.5,
                 cooldown_ms=350, seq_len=60, dim=17):
        self.model = model
        self.scaler = scaler
        self.classes = classes
        self.confirm_frames = confirm_frames
        self.min_conf = min_conf
        self.cooldown_ms = cooldown_ms
        self.seq_len = seq_len
        self.dim = dim
        self.buffer = []
        self.prev_l = None
        self.prev_r = None
        self.prev_t = None
        self.confirm_label = None
        self.confirm_count = 0
        self.already_fired_this_streak = False
        self.last_fire_t = -1e9
        self.events = []

    def _physics_gate_ok(self, label, feat):
        """기본(순수 TCN 트리거)은 게이트가 없다 — 항상 통과. v5c가 오버라이드한다."""
        return True

    def _extract_feat17(self, lm, now_ms):
        nose, lsh, rsh = lm[NOSE], lm[L_SH], lm[R_SH]
        lel, rel, lwr, rwr = lm[L_EL], lm[R_EL], lm[L_WR], lm[R_WR]
        sh2d = max(math.hypot(lsh.x - rsh.x, lsh.y - rsh.y), 1e-3)
        l_el_ratio = angle_deg(lsh, lel, lwr) / 180.0
        r_el_ratio = angle_deg(rsh, rel, rwr) / 180.0
        l_reach = dist3(lwr, lsh) / sh2d
        r_reach = dist3(rwr, rsh) / sh2d
        hands_dist = dist3(lwr, rwr) / sh2d
        l_wrist_nose = dist3(lwr, nose) / sh2d
        r_wrist_nose = dist3(rwr, nose) / sh2d
        elbow_dist = dist3(lel, rel) / sh2d
        avg_wrist_z = ((lwr.z + rwr.z) / 2.0) / sh2d
        lvx = lvy = lvz = rvx = rvy = rvz = 0.0
        dt = (now_ms - self.prev_t) / 1000.0 if self.prev_t else 0.0
        if self.prev_l and 0.008 < dt < 0.4:
            lvx = (lwr.x - self.prev_l[0]) / dt / sh2d
            lvy = (lwr.y - self.prev_l[1]) / dt / sh2d
            lvz = (lwr.z - self.prev_l[2]) / dt / sh2d
            rvx = (rwr.x - self.prev_r[0]) / dt / sh2d
            rvy = (rwr.y - self.prev_r[1]) / dt / sh2d
            rvz = (rwr.z - self.prev_r[2]) / dt / sh2d
        self.prev_l = (lwr.x, lwr.y, lwr.z)
        self.prev_r = (rwr.x, rwr.y, rwr.z)
        self.prev_t = now_ms
        l_speed = math.hypot(lvx, lvy, lvz)
        r_speed = math.hypot(rvx, rvy, rvz)
        return [
            l_el_ratio, r_el_ratio, l_reach, r_reach,
            lvx, lvy, lvz, rvx, rvy, rvz,
            l_speed, r_speed, hands_dist, l_wrist_nose, r_wrist_nose, elbow_dist, avg_wrist_z,
        ]

    def process(self, lm, now_ms):
        if lm is None or len(lm) <= R_WR:
            return []
        feat = self._extract_feat17(lm, now_ms)
        self.buffer.append(feat)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

        import torch
        median = self.scaler["median"]
        scale = self.scaler["scale"]
        clip = self.scaler.get("clip", 8.0)
        n = len(self.buffer)
        data = []
        for t in range(self.seq_len):
            src_idx = max(0, t - (self.seq_len - n))
            src = self.buffer[src_idx]
            data.append([max(-clip, min(clip, (src[f] - median[f]) / scale[f])) for f in range(self.dim)])
        x = torch.tensor([data], dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(self.model(x)[0], dim=-1).numpy()
        idx = int(probs.argmax())
        label = self.classes[idx]
        conf = float(probs[idx])

        is_punch_candidate = label in self.PUNCH_CLASSES and conf >= self.min_conf
        if is_punch_candidate and not self._physics_gate_ok(label, feat):
            is_punch_candidate = False  # 서브클래스(v5c)가 물리 게이트를 요구하면 여기서 걸러진다
        if is_punch_candidate and label == self.confirm_label:
            self.confirm_count += 1
        elif is_punch_candidate:
            self.confirm_label = label
            self.confirm_count = 1
        else:
            self.confirm_label = None
            self.confirm_count = 0
            self.already_fired_this_streak = False  # 스트릭이 끝났으니 다음 확정은 새 이벤트

        fired = []
        if (is_punch_candidate and self.confirm_count >= self.confirm_frames
                and not self.already_fired_this_streak
                and (now_ms - self.last_fire_t) >= self.cooldown_ms):
            side = "L" if label.startswith("LEFT_") else "R"
            suffix = label.split("_", 1)[1]
            kind = self.KIND_OF[suffix]
            ev = {
                "t_ms": now_ms, "frame": None, "side": side,
                "action": PUNCH_NAME[side][kind], "kind": kind,
                "speed_ms": 0.0, "speed_kmh": 0.0, "reach_n": 0.0, "elbow_deg": 0.0,
                "vx": 0.0, "vy": 0.0, "conf_margin": round(conf, 3),
            }
            fired.append(ev)
            self.events.append(ev)
            self.already_fired_this_streak = True
            self.last_fire_t = now_ms
        return fired


class TCNHybridTriggerEvaluator(TCNTriggerEvaluator):
    """v5c: TCNTriggerEvaluator(v6b)와 confirm/edge/cooldown 로직은 완전히 동일하되,
    "TCN 확신도" AND "룰베이스 물리 조건"을 둘 다 만족해야 후보로 인정한다.

    v5b(순수 TCN 트리거) 실험 결과, 확신도+시간적 안정성만으로는 90초에 60번을 쏴 Precision이
    0.20까지 무너졌다(F1 0.32, 룰베이스 트리거 v5의 0.379보다 낮음) — 모델이 프레임마다 흔들리는
    구간(footwork, 팔 반동 등)에서도 "펀치처럼 보인다"는 확신을 자주 냈기 때문이다.

    여기서는 그 순간 해당 side의 손목이 **실제로 물리적으로 뻗어나가는 중인가**
    (`TCN_HYBRID_ARM_GATE` 이상의 속도 AND `TCN_HYBRID_EXTEND_GATE` 이상의 뻗음)를 추가로 요구한다.
    TCN 혼자서도, 물리 조건 혼자서도 열 수 없고 **둘 다 있어야 연다.**
    """

    def _physics_gate_ok(self, label, feat):
        side = "L" if label.startswith("LEFT_") else "R"
        speed = feat[10] if side == "L" else feat[11]   # l_speed / r_speed
        reach = feat[2] if side == "L" else feat[3]      # l_reach / r_reach
        return speed > TCN_HYBRID_ARM_GATE and reach > TCN_HYBRID_EXTEND_GATE


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
    global HOOK_VX, HOOK_ELBOW, LOCK_MIN_MS, LOCK_MAX_MS, TCN_MIN_CONF
    global TCN_TRIGGER_CONFIRM_FRAMES, TCN_TRIGGER_MIN_CONF, TCN_TRIGGER_COOLDOWN_MS
    global TCN_HYBRID_ARM_GATE, TCN_HYBRID_EXTEND_GATE
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
    TCN_MIN_CONF = float(tune.get("TCN_MIN_CONF", TCN_MIN_CONF))
    TCN_TRIGGER_CONFIRM_FRAMES = int(tune.get("TCN_TRIGGER_CONFIRM_FRAMES", TCN_TRIGGER_CONFIRM_FRAMES))
    TCN_TRIGGER_MIN_CONF = float(tune.get("TCN_TRIGGER_MIN_CONF", TCN_TRIGGER_MIN_CONF))
    TCN_TRIGGER_COOLDOWN_MS = float(tune.get("TCN_TRIGGER_COOLDOWN_MS", TCN_TRIGGER_COOLDOWN_MS))
    TCN_HYBRID_ARM_GATE = float(tune.get("TCN_HYBRID_ARM_GATE", TCN_HYBRID_ARM_GATE))
    TCN_HYBRID_EXTEND_GATE = float(tune.get("TCN_HYBRID_EXTEND_GATE", TCN_HYBRID_EXTEND_GATE))
    print(f"🔧 [Config Applied] {Path(config_path).name} (SPEED={PUNCH_SPEED}, EXTEND={PUNCH_EXTEND}, TCN_CONF={TCN_MIN_CONF})")


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
    ap.add_argument("--engine", default="rule", choices=["rule", "tcn", "tcn_trigger", "tcn_hybrid"],
                     help="rule: 룰베이스 트리거+분류 / tcn: 룰베이스 트리거 + TCN 분류 / "
                          "tcn_trigger(v6b): TCN이 트리거까지 직접 담당 / "
                          "tcn_hybrid(v5c): TCN 확신도 AND 룰베이스 물리조건")
    ap.add_argument("--tcn-model-dir", default=None,
                     help="boxing_tcn.pth + boxing_tcn_scaler.json 를 담은 디렉터리 오버라이드 "
                          "(기본값: motion_learning/). tcn/tcn_trigger 엔진에 공통 적용")
    ap.add_argument("--annotate", nargs="?", const="AUTO", help="write annotated mp4 (optional path)")
    ap.add_argument("--start", type=float, default=0.0, help="analysis start second")
    ap.add_argument("--end", type=float, default=None, help="analysis end second")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--no-calib", action="store_true", help="skip neutral stance calibration window")
    args = ap.parse_args()

    if args.config:
        apply_tune_config(args.config)

    tcn_model_dir = Path(args.tcn_model_dir) if args.tcn_model_dir else None

    tcn_clf = None
    tcn_trigger = None
    if args.engine == "tcn":
        tcn_clf = TCNMotionClassifier(model_dir=tcn_model_dir)
        # TCN 로드 실패를 조용히 룰베이스로 폴백하면 registry가 오염된다
        # ("v4_tcn_hybrid" 인데 실제로는 rule 결과가 저장되는 상황).
        # --engine tcn 을 명시적으로 요청했는데 모델이 없으면 즉시 중단한다.
        if tcn_clf.model is None:
            raise SystemExit(
                "TCN 엔진 요청됨(--engine tcn) 하지만 모델 로드 실패. "
                "가중치/스케일러 경로와 torch 설치를 확인하세요. "
                "registry 오염을 막기 위해 파이프라인을 중단합니다."
            )
    elif args.engine in ("tcn_trigger", "tcn_hybrid"):
        probe = TCNMotionClassifier(model_dir=tcn_model_dir)
        if probe.model is None:
            raise SystemExit(
                f"TCN 트리거 엔진 요청됨(--engine {args.engine}) 하지만 모델 로드 실패. "
                "가중치/스케일러 경로와 torch 설치를 확인하세요."
            )
        evaluator_cls = TCNHybridTriggerEvaluator if args.engine == "tcn_hybrid" else TCNTriggerEvaluator
        tcn_trigger = evaluator_cls(
            probe.model, probe.scaler, TCNMotionClassifier.CLASSES,
            confirm_frames=TCN_TRIGGER_CONFIRM_FRAMES,
            min_conf=TCN_TRIGGER_MIN_CONF,
            cooldown_ms=TCN_TRIGGER_COOLDOWN_MS,
        )
        if args.engine == "tcn_hybrid":
            print(f"🧠 [TCN Hybrid Trigger Engine] confirm_frames={TCN_TRIGGER_CONFIRM_FRAMES} "
                  f"min_conf={TCN_TRIGGER_MIN_CONF} cooldown_ms={TCN_TRIGGER_COOLDOWN_MS} "
                  f"AND arm_gate={TCN_HYBRID_ARM_GATE} extend_gate={TCN_HYBRID_EXTEND_GATE}")
        else:
            print(f"🧠 [TCN Trigger Engine] confirm_frames={TCN_TRIGGER_CONFIRM_FRAMES} "
                  f"min_conf={TCN_TRIGGER_MIN_CONF} cooldown_ms={TCN_TRIGGER_COOLDOWN_MS}")

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

    evaluator = None if tcn_trigger else PunchEvaluator(calib=not args.no_calib, tcn_classifier=tcn_clf)
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

        if tcn_trigger is not None:
            # v6b: 룰베이스 키네마틱스를 전혀 거치지 않는다 — wl(월드 랜드마크)도 안 쓴다.
            fired = tcn_trigger.process(lm, ts_ms)
            active_events = tcn_trigger.events
        else:
            if tcn_clf and lm is not None:
                tcn_clf.push(lm, ts_ms)
            fired = evaluator.process(lm, wl, ts_ms)
            active_events = evaluator.events

        for ev in fired:
            ev["frame"] = seen_frames
            label_until = ts_ms + 700

        if writer is not None:
            if tcn_trigger is not None:
                hud = [f"t={ts_ms / 1000.0:.2f}s  punches={len(active_events)} (tcn_trigger)"]
            else:
                hud = [
                    f"t={ts_ms / 1000.0:.2f}s  punches={len(active_events)}",
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
    if tcn_trigger is not None:
        # v6b 는 "arm/window" 개념이 없다 (물리 임계값을 안 쓰므로) — 0으로 둔다.
        windows_opened = 0
        windows_expired = 0
        events = tcn_trigger.events
    else:
        windows_opened = sum(st.windows_opened for st in evaluator.arms.values())
        windows_expired = evaluator.windows_expired
        events = evaluator.events
    coverage = detected_frames / seen_frames if seen_frames else 0.0
    fps_mean = statistics.mean(fps_samples) if fps_samples else 0.0
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
    cols = ["t_ms", "frame", "side", "action", "kind", "speed_ms", "speed_kmh", "reach_n", "elbow_deg", "vx", "vy", "conf_margin"]
    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            wcsv = csv.DictWriter(f, fieldnames=cols)
            wcsv.writeheader()
            for e in events:
                wcsv.writerow({c: e.get(c, "") for c in cols})
    except Exception as err:
        import os
        tmp_csv = out_dir / f"punches_{os.getpid()}.csv"
        with tmp_csv.open("w", newline="", encoding="utf-8-sig") as f:
            wcsv = csv.DictWriter(f, fieldnames=cols)
            wcsv.writeheader()
            for e in events:
                wcsv.writerow({c: e.get(c, "") for c in cols})
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
