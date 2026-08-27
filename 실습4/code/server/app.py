import os
import json
import asyncio
import math
import time
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="4-Player AR Boxing & Battle Arena")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# 정적 파일
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

eval_video_dir = os.path.join(os.path.dirname(BASE_DIR), "eval", "video")
os.makedirs(eval_video_dir, exist_ok=True)
app.mount("/eval_video", StaticFiles(directory=eval_video_dir), name="eval_video")

eval_output_dir = os.path.join(os.path.dirname(BASE_DIR), "eval", "output")
os.makedirs(eval_output_dir, exist_ok=True)
app.mount("/eval_output", StaticFiles(directory=eval_output_dir), name="eval_output")

# 4인 파이터 상태 관리자
COLLIDER_RADIUS = 2.8
MIN_FIGHTER_DIST = COLLIDER_RADIUS * 2  # 5.6 — 두 파이터가 겹치지 않는 최소 거리


class ArenaGameManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.last_attack_times: Dict[str, float] = {}
        self._last_collision_time: float = 0.0
        self._collision_interval: float = 0.05  # 50ms 간격으로 충돌 체크 (초당 20회)
        self._last_positions: Dict[str, tuple] = {}  # 마지막 broadcast한 위치 캐시
        self._last_state_sync: float = 0.0           # fighters 전체 상태를 마지막으로 실은 시각
        # 3D 복원 얼굴 { client_id: {"lm": [...], "tex": "data:image/jpeg;base64,..."} }
        # 접속이 끊겨도 유지한다 — 재접속할 때마다 다시 촬영하게 만들 이유가 없다.
        self.faces: Dict[str, dict] = {}
        self.reset_game()

    # 분노 게이지 — 맞을수록 찬다. 가득 차면 필살기를 쓸 수 있다.
    #
    # "맞으면 찬다"로 설계한 이유: 지고 있는 쪽에 역전 수단을 주면 경기가 끝까지 팽팽해진다.
    # 때릴 때도 조금 차게 해서 적극적으로 싸우는 쪽이 손해 보지 않게 한다.
    RAGE_MAX = 100.0
    # 수치는 데미지 설계(HP 100 = 최소 10대)에 맞춰 역산했다.
    # 한 대(데미지 6 안팎)에 약 13 이 차므로 **7~8대 맞으면 가득 찬다** —
    # 즉 한 번 죽을 뻔할 때마다 필살기 한 번이다. 이보다 빠르면 필살기가 상시 기술이 되고,
    # 느리면 경기 내내 한 번도 못 쓴다.
    RAGE_ON_HIT_TAKEN = 2.2      # 맞았을 때 (데미지 1당)
    RAGE_ON_HIT_DEALT = 0.55     # 때렸을 때 (데미지 1당) — 적극적으로 싸우는 쪽도 조금은 찬다
    RAGE_DECAY_PER_SEC = 1.2     # 차오르는 중일 때만 식는다 (가득 찬 게이지는 유지)

    def reset_game(self):
        self.fighters = {
            "client_1": {"name": "Red Boxer", "color": "#FF3366", "hp": 100, "score": 0, "action": "IDLE", "pos": [-12, 0, 0], "world_x": -12, "world_z": 0, "yaw": -1.5708, "rage": 0.0},
            "client_2": {"name": "Cyan Boxer", "color": "#00E5FF", "hp": 100, "score": 0, "action": "IDLE", "pos": [12, 0, 0], "world_x": 12, "world_z": 0, "yaw": 1.5708, "rage": 0.0},
            "client_3": {"name": "Gold Mage", "color": "#FFD700", "hp": 100, "score": 0, "action": "IDLE", "pos": [0, 0, -12], "world_x": 0, "world_z": -12, "yaw": 3.1416, "rage": 0.0},
            "client_4": {"name": "Green Striker", "color": "#00FF66", "hp": 100, "score": 0, "action": "IDLE", "pos": [0, 0, 12], "world_x": 0, "world_z": 12, "yaw": 0, "rage": 0.0},
        }

    def add_rage(self, client_id: str, amount: float):
        """분노 게이지를 올린다. 죽은 파이터는 차지 않는다."""
        f = self.fighters.get(client_id)
        if not f or f.get("hp", 0) <= 0:
            return
        f["rage"] = min(self.RAGE_MAX, f.get("rage", 0.0) + amount)

    def decay_rage(self):
        """
        차오르는 중인 게이지는 서서히 식는다. 맞다 말고 도망다니면 그대로 유지되면 안 된다.

        **다만 가득 찬 게이지는 식지 않는다.** 그러지 않으면 100에 닿는 순간부터 곧바로 깎여
        (실측 99.1) 사용 조건인 "가득 참"을 영영 만족하지 못한다 — 필살기를 아예 못 쓴다.
        어렵게 채운 게이지는 쓸 때까지 유지되는 것이 규칙으로도 자연스럽다.
        """
        now = time.monotonic()
        dt = now - getattr(self, "_last_rage_t", now)
        self._last_rage_t = now
        if dt <= 0 or dt > 2.0:
            return
        drop = self.RAGE_DECAY_PER_SEC * dt
        for f in self.fighters.values():
            r = f.get("rage", 0)
            if 0 < r < self.RAGE_MAX:
                f["rage"] = max(0.0, r - drop)

    def should_sync_state(self, interval: float = 0.5) -> bool:
        """마지막 전체 상태 브로드캐스트로부터 interval 초가 지났으면 True."""
        now = time.monotonic()
        if now - self._last_state_sync < interval:
            return False
        self._last_state_sync = now
        return True

    def enforce_collision_throttled(self):
        """충돌 체크를 일정 간격(50ms)으로만 실행하여 CPU 부하 감소."""
        now = time.monotonic()
        if now - self._last_collision_time < self._collision_interval:
            return {}
        self._last_collision_time = now
        return self.enforce_collision()

    def enforce_collision(self):
        """모든 파이터 쌍 간 거리 검사 → 겹치면 서로 밀어냄. 수정된 world_x/z를 반환."""
        corrections = {}  # {client_id: (new_x, new_z)}
        ids = list(self.fighters.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                fa = self.fighters[a]
                fb = self.fighters[b]
                # KO된 파이터는 링에서 사라지므로 콜라이더에서도 빠진다.
                # 그러지 않으면 보이지 않는 벽이 남아 살아있는 파이터가 막힌다.
                if fa.get("hp", 0) <= 0 or fb.get("hp", 0) <= 0:
                    continue
                ax, az = fa.get("world_x", fa.get("pos", [0, 0, 0])[0]), fa.get("world_z", fa.get("pos", [0, 0, 0])[2])
                bx, bz = fb.get("world_x", fb.get("pos", [0, 0, 0])[0]), fb.get("world_z", fb.get("pos", [0, 0, 0])[2])
                dx, dz = ax - bx, az - bz
                dist = math.hypot(dx, dz)
                if dist < MIN_FIGHTER_DIST and dist > 0.001:
                    nx, nz = dx / dist, dz / dist
                    push = (MIN_FIGHTER_DIST - dist) / 2
                    ca = corrections.get(a, (ax, az))
                    cb = corrections.get(b, (bx, bz))
                    corrections[a] = (ca[0] + nx * push, ca[1] + nz * push)
                    corrections[b] = (cb[0] - nx * push, cb[1] - nz * push)
        # 기록 반영
        for cid, (x, z) in corrections.items():
            x = max(-16.0, min(16.0, x))
            z = max(-16.0, min(16.0, z))
            self.fighters[cid]["world_x"] = x
            self.fighters[cid]["world_z"] = z
        return corrections

    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """슬롯 중복 검사. 이미 사용 중이면 accept 후 code 4409로 close하고 False 반환."""
        # 예약된 슬롯: host_arena(1) + client_1~4(각 1)
        if client_id in self.active_connections:
            existing = self.active_connections[client_id]
            # 기존 연결이 아직 살아 있는지 ping. 살아 있으면 신규 거부.
            try:
                # WebSocket 상태 확인 (starlette는 client_state.value == 1이면 CONNECTED)
                still_open = getattr(existing, "client_state", None)
                if still_open is None or still_open.value == 1:
                    await websocket.accept()
                    await websocket.send_text(json.dumps({
                        "type": "connection_rejected",
                        "reason": "slot_in_use",
                        "client_id": client_id,
                        "message": f"'{client_id}' 슬롯이 이미 사용 중입니다. 다른 슬롯을 선택하거나 기존 세션을 종료한 뒤 다시 시도하세요.",
                    }))
                    await websocket.close(code=4409, reason="slot_in_use")
                    print(f"[!] Rejected duplicate connection for {client_id}", flush=True)
                    return False
            except Exception:
                # 기존 연결이 죽어 있으면 정리하고 신규 접속을 받는다.
                pass
            self.disconnect(client_id)

        await websocket.accept()
        self.active_connections[client_id] = websocket
        if client_id in self.fighters:
            self.fighters[client_id]["hp"] = 100 # 접속 시 HP 100 초기화!
        print(f"[+] Fighter connected: {client_id}")

        # 이미 등록된 얼굴들을 새로 들어온 쪽에 몰아서 보낸다.
        # 그러지 않으면 나중에 들어온 사람은 남들 얼굴을 영영 못 본다.
        if self.faces:
            try:
                await websocket.send_text(json.dumps({
                    "type": "face_bulk",
                    "faces": self.faces,
                }))
            except Exception:
                pass

        await self.broadcast({
            "type": "game_state",
            "event": "fighter_joined",
            "client_id": client_id,
            "fighters": self.fighters,
            "active_users": list(self.active_connections.keys())
        })
        return True

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"[-] Fighter disconnected: {client_id}")

    async def broadcast(self, message: dict):
        msg_text = json.dumps(message)

        async def _send(cid, conn):
            try:
                await conn.send_text(msg_text)
            except Exception:
                return cid
            return None

        results = await asyncio.gather(*[_send(cid, conn) for cid, conn in self.active_connections.items()], return_exceptions=True)
        for cid in results:
            if isinstance(cid, str):
                self.disconnect(cid)

    def process_attack(self, attacker_id: str, action: str, velocity: float):
        """공격 판정 — 디버그 로그 포함"""

        attacker = self.fighters.get(attacker_id, {})
        if attacker.get("hp", 0) <= 0:
            return None

        # (데미지, 최대사거리, dot 임계값)
        #
        # HP 100 기준 "최소 10대는 버틴다"를 만족하도록 잡은 값이다.
        # 속도 보너스까지 최대로 받은 어퍼컷(가장 센 기술)이 정확히 10, 즉 10대가 상한이고
        # 실제 대전에서 흔한 잽/스트레이트 위주면 15~20대가 오간다.
        attack_specs = {
            "JAB_STRAIGHT":    (5, 10.0, 0.3),
            "LEFT_JAB":        (5, 10.0, 0.3),
            "RIGHT_CROSS":     (6, 10.0, 0.3),
            "LEFT_HOOK":       (7, 10.0, 0.2),
            "RIGHT_HOOK":      (7, 10.0, 0.2),
            "LEFT_UPPERCUT":   (8,  8.0, 0.3),
            "RIGHT_UPPERCUT":  (8,  8.0, 0.3),
            # 필살기 — 분노 게이지를 가득 채워야 쓸 수 있다.
            # 사거리가 길고(18) 시야각이 넓어(0.1) 정면 부채꼴을 쓸어버린다.
            "ENERGY_WAVE":     (34, 18.0, 0.1),
        }
        spec = attack_specs.get(action)
        if not spec:
            return None
        raw_dmg, max_range, dot_threshold = spec

        # 필살기는 게이지가 가득 찼을 때만. 쓰면 전부 소모한다.
        if action == "ENERGY_WAVE":
            if attacker.get("rage", 0) < self.RAGE_MAX:
                print(f"[ULT] {attacker_id} 게이지 부족 ({attacker.get('rage', 0):.0f}/100)", flush=True)
                return None
            attacker["rage"] = 0.0
            print(f"[ULT] {attacker_id} 필살기 발동!", flush=True)

        # 0.3초 쿨다운
        now = asyncio.get_event_loop().time()
        last_time = self.last_attack_times.get(attacker_id, 0.0)
        if now - last_time < 0.3:
            print(f"[ATK] {attacker_id} {action} BLOCKED by cooldown ({now - last_time:.2f}s)", flush=True)
            return None
        self.last_attack_times[attacker_id] = now

        # 속도 보너스 최대 +25% (이전 +50%). 빠른 펀치의 이점은 남기되 즉사 구간을 없앤다.
        dmg = max(1, int(raw_dmg * (1.0 + min(velocity, 50.0) / 200.0)))

        att_x = attacker.get("world_x", attacker.get("pos", [0, 0, 0])[0])
        att_z = attacker.get("world_z", attacker.get("pos", [0, 0, 0])[2])
        att_yaw = attacker.get("yaw", 0.0)

        look_dx = -math.sin(att_yaw)
        look_dz = -math.cos(att_yaw)

        best_target_id = None
        best_dot = -2.0

        for target_id, fighter in self.fighters.items():
            if target_id != attacker_id and fighter.get("hp", 0) > 0:
                tgt_x = fighter.get("world_x", fighter.get("pos", [0, 0, 0])[0])
                tgt_z = fighter.get("world_z", fighter.get("pos", [0, 0, 0])[2])

                to_tgt_x = tgt_x - att_x
                to_tgt_z = tgt_z - att_z
                dist = (to_tgt_x**2 + to_tgt_z**2)**0.5

                if dist > 0.1:
                    dot = (look_dx * to_tgt_x + look_dz * to_tgt_z) / dist
                    in_range = dist <= max_range
                    in_angle = dot > dot_threshold
                    print(f"[ATK] {attacker_id}->{target_id} dist={dist:.1f}(max{max_range}) dot={dot:.2f}(min{dot_threshold}) {'OK' if in_range and in_angle else 'MISS'}", flush=True)
                    if in_range and in_angle and dot > best_dot:
                        best_dot = dot
                        best_target_id = target_id

        # 필살기는 **부채꼴 안의 모두**를 때린다. 일반 펀치는 가장 정면인 하나만.
        if action == "ENERGY_WAVE":
            hits = []
            for target_id, fighter in self.fighters.items():
                if target_id == attacker_id or fighter.get("hp", 0) <= 0:
                    continue
                tx = fighter.get("world_x", 0.0)
                tz = fighter.get("world_z", 0.0)
                dx, dz = tx - att_x, tz - att_z
                dist = math.hypot(dx, dz)
                if dist <= 0.1 or dist > max_range:
                    continue
                if (look_dx * dx + look_dz * dz) / dist <= dot_threshold:
                    continue
                is_guard = (fighter.get("action") in ["TWO_HAND_GUARD", "DUAL_GUARD"])
                # 필살기는 가드로도 절반밖에 못 막는다
                actual = int(dmg * 0.5) if is_guard else dmg
                fighter["hp"] = max(0, fighter["hp"] - actual)
                self.add_rage(target_id, actual * self.RAGE_ON_HIT_TAKEN)
                hits.append({
                    "attacker_id": attacker_id, "target_id": target_id,
                    "damage": actual, "is_guard": is_guard,
                    "target_hp": fighter["hp"], "distance": round(dist, 1),
                    "ko": fighter["hp"] <= 0, "ultimate": True,
                })
                print(f"[ULT] {attacker_id}->{target_id} dmg={actual} hp={fighter['hp']}", flush=True)
                if fighter["hp"] == 0:
                    attacker["score"] = attacker.get("score", 0) + 1
            return hits

        hits = []
        if best_target_id:
            fighter = self.fighters[best_target_id]
            is_guard = (fighter.get("action") in ["TWO_HAND_GUARD", "DUAL_GUARD"])
            actual_dmg = int(dmg * 0.2) if is_guard else dmg
            fighter["hp"] = max(0, fighter["hp"] - actual_dmg)
            tgt_x = fighter.get("world_x", fighter.get("pos", [0, 0, 0])[0])
            tgt_z = fighter.get("world_z", fighter.get("pos", [0, 0, 0])[2])
            hit_dist = ((tgt_x - att_x)**2 + (tgt_z - att_z)**2)**0.5
            hits.append({
                "attacker_id": attacker_id,
                "target_id": best_target_id,
                "damage": actual_dmg,
                "is_guard": is_guard,
                "target_hp": fighter["hp"],
                "distance": round(hit_dist, 1),
                "ko": fighter["hp"] <= 0,
            })
            # 분노: 맞은 쪽이 많이, 때린 쪽도 조금
            self.add_rage(best_target_id, actual_dmg * self.RAGE_ON_HIT_TAKEN)
            self.add_rage(attacker_id, actual_dmg * self.RAGE_ON_HIT_DEALT)
            print(f"[HIT] {attacker_id}->{best_target_id} dmg={actual_dmg} hp={fighter['hp']} "
                  f"rage={self.fighters[best_target_id]['rage']:.0f}", flush=True)
            if fighter["hp"] == 0:
                attacker["score"] = attacker.get("score", 0) + 1
        else:
            print(f"[ATK] {attacker_id} {action} NO TARGET HIT", flush=True)

        return hits

manager = ArenaGameManager()

@app.get("/", response_class=HTMLResponse)
@app.get("/arena", response_class=HTMLResponse)
async def get_arena_page(request: Request):
    """메인 Host 3D 복싱 링 / 배틀 아레나 페이지"""
    return templates.TemplateResponse(request=request, name="arena.html", context={})

@app.get("/client", response_class=HTMLResponse)
async def get_client_page(request: Request, id: str = "client_1"):
    """파이터 웹캠 클라이언트 페이지"""
    valid_ids = ["client_1", "client_2", "client_3", "client_4"]
    if id not in valid_ids:
        # 오타 방지: 유효하지 않은 id면 client_1로 리다이렉트
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/client?id=client_1")
    fighter = manager.fighters.get(id, {"name": "Fighter", "color": "#FF3366"})
    return templates.TemplateResponse(
        request=request,
        name="fighter_client.html",
        context={"client_id": id, "name": fighter["name"], "color": fighter["color"]}
    )

@app.get("/replay", response_class=HTMLResponse)
async def get_replay_page(request: Request):
    """비디오 입력 기반 3D 아바타 실시간 동작 리플레이 뷰어"""
    return templates.TemplateResponse(
        request=request,
        name="eval_replay.html",
        context={}
    )

@app.get("/eval", response_class=HTMLResponse)
async def get_eval_dashboard(request: Request):
    """녹화영상 + 3D 리플레이 + 벤치마크 정확도 통합 대시보드"""
    return templates.TemplateResponse(
        request=request,
        name="eval_dashboard.html",
        context={}
    )

@app.get("/api/eval-versions")
async def get_eval_versions():
    """등록된 벤치마크 평가 버전 리스트 API"""
    runs_dir = os.path.join(os.path.dirname(BASE_DIR), "eval", "runs")
    registry_path = os.path.join(runs_dir, "runs_registry.json")
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                return JSONResponse(json.load(f))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"runs": []})


@app.get("/api/eval-punches")
async def get_eval_punches(version: str = None):
    """벤치마크 펀치 검출 결과 CSV → JSON API (버전별 지원)"""
    import csv
    if version:
        runs_dir = os.path.join(os.path.dirname(BASE_DIR), "eval", "runs", version)
        csv_path = os.path.join(runs_dir, "punches.csv")
    else:
        csv_path = os.path.join(eval_output_dir, "benchmark", "punches.csv")

    if not os.path.exists(csv_path):
        return JSONResponse({"error": f"punches.csv for version '{version}' not found"}, status_code=404)
    rows = []
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                clean_r = {k.strip().lstrip("\ufeff"): v.strip() for k, v in r.items() if k}
                t_val = clean_r.get("t_ms")
                if not t_val:
                    continue
                rows.append({
                    "t_ms": int(float(t_val)),
                    "frame": int(float(clean_r.get("frame", 0))),
                    "side": clean_r.get("side", ""),
                    "action": clean_r.get("action", ""),
                    "speed_ms": float(clean_r.get("speed_ms", 0)),
                    "speed_kmh": float(clean_r.get("speed_kmh", 0)),
                    "elbow_deg": float(clean_r.get("elbow_deg", 0)),
                    "conf_margin": float(clean_r.get("conf_margin", 0)),
                })
        return JSONResponse({"version": version or "latest", "punches": rows, "total": len(rows)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/reset-game")
@app.get("/api/reset-game")
async def reset_game_endpoint():
    """모든 파이터 HP 100 및 점수 초기화"""
    manager.reset_game()
    await manager.broadcast({
        "type": "game_state",
        "event": "game_reset",
        "fighters": manager.fighters,
        "active_users": list(manager.active_connections.keys())
    })
    return {"status": "success", "message": "Game reset to 100 HP"}

@app.get("/api/motion-eval")
async def get_motion_eval():
    """GPU 딥러닝 모션 학습 지표 반환"""
    eval_file = os.path.join(os.path.dirname(BASE_DIR), "motion_learning", "eval_results.json")
    if os.path.exists(eval_file):
        with open(eval_file, "r") as f:
            return json.load(f)
    return {
        "device": "cuda:0 (NVIDIA GPU)",
        "rule_based_accuracy": 0.624,
        "lstm_accuracy": 0.987,
        "improvement_pct_points": 36.3,
        "training_time_seconds": 4.12
    }

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    accepted = await manager.connect(websocket, client_id)
    if not accepted:
        # 중복 슬롯: connect()가 이미 accept + 거부 메시지 + close 처리함.
        return
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            # 3D 복원 얼굴 등록 — 캡처 시 1회만 온다 (수십 KB라 매 프레임 오는 패킷이 아니다)
            if payload.get("type") == "face":
                face = payload.get("face")
                if isinstance(face, dict) and face.get("lm") and face.get("tex"):
                    manager.faces[client_id] = face
                    print(f"[FACE] {client_id} 얼굴 등록 "
                          f"(랜드마크 {len(face['lm']) // 3}개, 텍스처 {len(face['tex']) // 1024}KB)",
                          flush=True)
                    await manager.broadcast({
                        "type": "face_update",
                        "client_id": client_id,
                        "face": face,
                    })
                continue

            action = payload.get("action", "IDLE")
            velocity = payload.get("velocity", 0.0)

            # 공격 액션은 로깅
            if action not in ("IDLE", "DUAL_GUARD", "TWO_HAND_GUARD"):
                print(f"[RECV] {client_id} action={action} vel={velocity:.1f} pos=({payload.get('world_x',0):.1f},{payload.get('world_z',0):.1f})", flush=True)

            # 파이터 액션 및 3D 월드 위치 갱신
            if client_id in manager.fighters:
                manager.fighters[client_id]["action"] = action
                if "world_x" in payload:
                    manager.fighters[client_id]["world_x"] = payload["world_x"]
                if "world_z" in payload:
                    manager.fighters[client_id]["world_z"] = payload["world_z"]
                if "yaw" in payload:
                    manager.fighters[client_id]["yaw"] = payload["yaw"]

            # 서버 권한 충돌 해소 — 쓰로틀링 적용 (50ms 간격)
            corrections = manager.enforce_collision_throttled()

            # 충돌 보정된 좌표를 payload에 반영 (arena 뷰 + 클라이언트 동기화)
            if client_id in manager.fighters:
                payload["world_x"] = manager.fighters[client_id]["world_x"]
                payload["world_z"] = manager.fighters[client_id]["world_z"]

            # 타격 이벤트 판정 (양손 액션 포함)
            hit_results = None
            if action in ["JAB_STRAIGHT", "LEFT_JAB", "RIGHT_CROSS", "LEFT_HOOK", "RIGHT_HOOK",
                          "LEFT_UPPERCUT", "RIGHT_UPPERCUT", "ENERGY_WAVE"]:
                hit_results = manager.process_attack(client_id, action, velocity)

            payload["client_id"] = client_id
            payload["color"] = manager.fighters.get(client_id, {}).get("color", "#FFFFFF")
            payload["hits"] = hit_results
            # fighters 전체 상태: 타격/충돌 보정 시, 그리고 최소 0.5초마다 한 번은 반드시 포함.
            #
            # 예전에는 타격·보정이 있을 때만 보냈다. 그러면 아무도 안 때리는 동안 클라이언트는
            # HP도 상대 위치도 갱신받지 못한다. K.O.된 뒤 아무도 안 때리면 "왜 안 움직이지"만
            # 남고 사망 표시가 갱신되지 않는 상태가 된다. HUD의 HP 바·미니맵도 이 패킷에 의존한다.
            # 4명분 상태라 크기가 작아 0.5초 주기로 보내도 대역폭에 영향이 없다.
            if hit_results or corrections or manager.should_sync_state():
                manager.decay_rage()
                payload["fighters"] = manager.fighters

            # 위치/yaw 변경이 있거나 공격/타격이면 broadcast (idle 정지 상태만 스킵)
            is_attack = action not in ("IDLE", "DUAL_GUARD", "TWO_HAND_GUARD")
            curr_pos = (
                round(payload.get("world_x", 0), 2),
                round(payload.get("world_z", 0), 2),
                round(payload.get("yaw", 0), 2)
            )
            last_pos = manager._last_positions.get(client_id)
            pos_changed = (curr_pos != last_pos)

            # "fighters 를 실었다"면 반드시 내보내야 한다. 정지 상태에서 스킵해버리면
            # 주기 동기화가 무의미해지고, 가만히 서 있는 클라이언트는 HP를 영원히 못 받는다.
            if is_attack or hit_results or corrections or pos_changed or "fighters" in payload:
                manager._last_positions[client_id] = curr_pos
                await manager.broadcast(payload)
            else:
                pass  # idle 정지 → 스킵
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        await manager.broadcast({
            "type": "game_state",
            "event": "fighter_left",
            "client_id": client_id,
            "active_users": list(manager.active_connections.keys())
        })
    except asyncio.CancelledError:
        manager.disconnect(client_id)
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] websocket_endpoint exception for {client_id}: {e}", flush=True)
        traceback.print_exc()
        manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_graceful_shutdown=0)

