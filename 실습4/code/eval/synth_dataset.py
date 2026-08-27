"""Generate synthetic boxing punch sequences as landmarks JSONL + labels.json.

Each case is a parameterized motion program (side, punch kind, peak speed,
reach fraction) rendered into MediaPipe-style world landmarks (meters,
pelvis origin, y-down) plus matching normalized upper-body landmarks.
The arm is built as a two-link triangle (shoulder-elbow-wrist) with fixed
upper-arm/forearm lengths, so elbow angles always follow from wrist reach
via the law of cosines - exactly what the runtime kinematics measure.
Labels come from the program itself, so auto-grading needs no manual work.

Usage:
  python iter3/eval/synth_dataset.py                 # generate all cases
  python iter3/eval/synth_dataset.py --cases clean_hook_L,too_slow
"""
import argparse
import json
import math
import random
from pathlib import Path

FPS = 30
DT_MS = round(1000 / FPS)
UPPER_ARM = 0.30
FOREARM = 0.28
MAX_REACH = UPPER_ARM + FOREARM - 0.01
SH_HALF = 0.20
SH_Y = -0.22

SIDE_SIGN = {"L": -1.0, "R": 1.0}
GUARD_WRIST = {"L": (-0.12, -0.06, -0.28), "R": (0.12, -0.06, -0.28)}
NORMAL_BY_KIND = {
    "STRAIGHT": (0.0, 1.0, 0.0),
    "HOOK": (0.0, 1.0, 0.0),
    "UPPERCUT": (1.0, 0.0, 0.0),
}

TARGETS = {
    "STRAIGHT": lambda sx: (sx * 0.06, -0.16, -0.50),
    "HOOK": lambda sx: (-sx * 0.16, -0.10, -0.42),
    "UPPERCUT": lambda sx: (sx * 0.10, -0.32, -0.38),
}

NOISE_M = 0.003
PRE_ROLL_MS = 2300
POST_ROLL_MS = 1000
MIN_GAP_MS = 700
HOLD_MS = 80


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(v, k):
    return (v[0] * k, v[1] * k, v[2] * k)


def norm(v):
    return math.sqrt(sum(c * c for c in v))


def unit(v):
    n = norm(v)
    return (v[0] / n, v[1] / n, v[2] / n)


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def rotate_about(v, axis, ang):
    k = unit(axis)
    c, s = math.cos(ang), math.sin(ang)
    return tuple(v[i] * c + cross(k, v)[i] * s + k[i] * dot(k, v) * (1 - c) for i in range(3))


def smoothstep(u):
    u = clamp(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def shoulder(side):
    return (SIDE_SIGN[side] * SH_HALF, SH_Y, 0.0)


def elbow_for(side, wrist):
    sh = shoulder(side)
    dvec = sub(wrist, sh)
    d = clamp(norm(dvec), 0.05, MAX_REACH)
    cos_a = (UPPER_ARM**2 + d**2 - FOREARM**2) / (2 * UPPER_ARM * d)
    alpha = math.acos(clamp(cos_a, -1.0, 1.0))
    direction = rotate_about(unit(dvec), (0, 1, 0), alpha)
    return add(sh, scale(direction, UPPER_ARM))


class Punch:
    def __init__(self, spec, t_start):
        self.side = spec["side"]
        self.kind = spec["kind"]
        self.v_peak = spec.get("v_peak", 3.0)
        self.reach_scale = spec.get("reach_scale", 1.0)
        self.gap_after_ms = spec.get("gap_ms", MIN_GAP_MS)
        guard = GUARD_WRIST[self.side]
        target = TARGETS[self.kind](SIDE_SIGN[self.side])
        delta = scale(sub(target, guard), self.reach_scale)
        self.w_start = guard
        self.w_end = add(guard, delta)
        self.t_start = t_start
        self.t_strike = max(int(1.5 * norm(delta) / self.v_peak * 1000), 40)
        self.t_hold = HOLD_MS
        self.t_recover = int(self.t_strike * 1.6 + 150)

    def covers(self, t_ms):
        return self.t_start <= t_ms < self.t_start + self.t_strike + self.t_hold + self.t_recover

    def wrist_at(self, t_ms):
        t = t_ms - self.t_start
        if t < self.t_strike:
            u = smoothstep(t / self.t_strike)
            return add(self.w_start, scale(sub(self.w_end, self.w_start), u))
        if t < self.t_strike + self.t_hold:
            return self.w_end
        u = smoothstep((t - self.t_strike - self.t_hold) / self.t_recover)
        return add(self.w_end, scale(sub(self.w_start, self.w_end), u))

    def label(self):
        return {"t_ms": self.t_start + int(self.t_strike * 0.75), "side": self.side, "kind": self.kind}


def build_schedule(specs):
    punches = []
    t = PRE_ROLL_MS
    for spec in specs:
        p = Punch(spec, t)
        punches.append(p)
        t += p.t_strike + p.t_hold + p.t_recover + p.gap_after_ms
    duration = t - MIN_GAP_MS + POST_ROLL_MS
    return punches, duration


def build_sweep():
    punches = []
    t = PRE_ROLL_MS
    for _ in range(4):
        strike = max(int(1.5 * 0.35 / 2.2 * 1000), 40)
        recover = int(strike * 1.6 + 150)
        for side in ("L", "R"):
            p = Punch({"side": side, "kind": "STRAIGHT"}, t)
            p.w_end = add(GUARD_WRIST[side], (0.35, 0.05, 0.0))
            p.t_strike = strike
            p.t_recover = recover
            p.v_peak = 2.2
            punches.append(p)
        t += strike + HOLD_MS + recover + 600
    return punches, t - 600 + POST_ROLL_MS


def render_frames(punches, duration_ms, rng):
    frames = []
    t = 0
    while t <= duration_ms:
        world = [[0.0, -0.5, 0.0] for _ in range(33)]
        lm = [[0.5, 0.5, 0.0, 0.9] for _ in range(33)]

        def jitter(pt):
            return (
                pt[0] + rng.gauss(0, NOISE_M),
                pt[1] + rng.gauss(0, NOISE_M),
                pt[2] + rng.gauss(0, NOISE_M),
            )

        for idx, sh in ((11, shoulder("L")), (12, shoulder("R"))):
            world[idx] = [round(c, 4) for c in sh]
        world[0] = [round(rng.gauss(0, NOISE_M), 4), round(SH_Y - 0.18 + rng.gauss(0, NOISE_M), 4), round(rng.gauss(0, NOISE_M), 4)]

        for side, ids in (("L", (11, 13, 15)), ("R", (12, 14, 16))):
            active = next((p for p in punches if p.side == side and p.covers(t)), None)
            wx = active.wrist_at(t) if active else GUARD_WRIST[side]
            wrist = jitter(wx)
            elbow = jitter(elbow_for(side, wrist))
            world[ids[1]] = [round(elbow[0], 4), round(elbow[1], 4), round(elbow[2], 4)]
            world[ids[2]] = [round(wrist[0], 4), round(wrist[1], 4), round(wrist[2], 4)]

            sign = SIDE_SIGN[side]
            lm[ids[2]] = [round(clamp(0.5 + wx[0], 0.02, 0.98), 4),
                          round(clamp(0.55 + (wx[1] - SH_Y), 0.02, 0.98), 4), 0.0, 0.9]
            lm[ids[1]] = [round(clamp(0.5 + sign * 0.10, 0.02, 0.98), 4), 0.62, 0.0, 0.9]

        lm[0] = [0.5, 0.35, 0.0, 0.95]
        lm[11] = [0.30, 0.55, 0.0, 0.95]
        lm[12] = [0.70, 0.55, 0.0, 0.95]

        frames.append({
            "t_ms": t,
            "lm": lm,
            "wl": world,
        })
        t += DT_MS
    return frames


CASES = {
    "clean_straight_L": [{"side": "L", "kind": "STRAIGHT"} for _ in range(8)],
    "clean_straight_R": [{"side": "R", "kind": "STRAIGHT"} for _ in range(8)],
    "clean_hook_L": [{"side": "L", "kind": "HOOK"} for _ in range(8)],
    "clean_hook_R": [{"side": "R", "kind": "HOOK"} for _ in range(8)],
    "clean_uppercut_L": [{"side": "L", "kind": "UPPERCUT"} for _ in range(8)],
    "clean_uppercut_R": [{"side": "R", "kind": "UPPERCUT"} for _ in range(8)],
    "combo_mixed": [
        {"side": "L", "kind": "STRAIGHT"}, {"side": "R", "kind": "STRAIGHT"},
        {"side": "L", "kind": "HOOK"}, {"side": "R", "kind": "UPPERCUT"},
        {"side": "R", "kind": "HOOK"}, {"side": "L", "kind": "UPPERCUT"},
    ] * 2,
    "rapid_jab_combo": [
        {"side": "L", "kind": "STRAIGHT", "gap_after_ms": 320},
        {"side": "R", "kind": "STRAIGHT", "gap_after_ms": 320},
        {"side": "L", "kind": "STRAIGHT", "gap_after_ms": 320},
        {"side": "R", "kind": "STRAIGHT", "gap_after_ms": 320},
    ] * 2,
    "too_slow": [{"side": "L", "kind": "STRAIGHT", "v_peak": 1.1} for _ in range(6)],
    "short_reach": [{"side": "L", "kind": "STRAIGHT", "reach_scale": 0.15} for _ in range(6)],
}
NEGATIVE_CASES = {"too_slow", "short_reach", "sweep_rotation"}
ALL_CASES = list(CASES.keys()) + ["sweep_rotation"]


def generate_case(name, out_root, seed_base):
    rng = random.Random(seed_base + abs(hash(name)) % 1000)
    if name == "sweep_rotation":
        punches, duration = build_sweep()
    else:
        punches, duration = build_schedule(CASES[name])

    frames = render_frames(punches, duration, rng)

    case_dir = Path(out_root) / name
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "landmarks.jsonl").open("w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr) + "\n")

    expected = [] if name in NEGATIVE_CASES else sorted(
        (p.label() for p in punches), key=lambda x: x["t_ms"]
    )
    labels = {
        "case": name,
        "source": "synthetic",
        "tolerance_ms": 350,
        "punches": expected,
    }
    (case_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{name:<20} {len(frames):5d}프레임 ({duration / 1000:5.1f}s) · 정답 {len(expected)}발 → {case_dir}")


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic punch evaluation cases")
    ap.add_argument("--out", default=str(Path(__file__).parent / "datasets"))
    ap.add_argument("--cases", default=None, help="comma-separated subset (default: all)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    names = ALL_CASES
    if args.cases:
        names = [c.strip() for c in args.cases.split(",")]
        unknown = [c for c in names if c not in ALL_CASES]
        if unknown:
            print(f"알 수 없는 케이스: {unknown} (가능: {ALL_CASES})")
            raise SystemExit(1)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    for name in names:
        generate_case(name, args.out, args.seed)


if __name__ == "__main__":
    main()
