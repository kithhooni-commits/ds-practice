"""Boxing motion dataset collector v2: record → detect → review → relabel."""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, model_validator

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "collector_v2_templates"
OUTPUT_DIR = BASE_DIR / "collected_review_v2"
PUNCH_CORE_PATH = BASE_DIR.parent / "server" / "static" / "punch_core.js"
MAX_VIDEO_BYTES = 100 * 1024 * 1024
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
UPLOAD_ID = re.compile(r"^[a-f0-9]{32}$")
MANIFEST_LOCK = threading.Lock()

LABELS = {
    "IDLE": "대기",
    "OTHER": "기타 / 오판정",
    "LEFT_JAB": "왼손 잽",
    "RIGHT_JAB": "오른손 스트레이트",
    "LEFT_HOOK": "왼손 훅",
    "RIGHT_HOOK": "오른손 훅",
    "LEFT_UPPERCUT": "왼손 어퍼컷",
    "RIGHT_UPPERCUT": "오른손 어퍼컷",
    "TWO_HAND_GUARD": "양손 가드",
    "ENERGY_WAVE": "에너지 웨이브",
}


class DetectionEvent(BaseModel):
    event_id: str = Field(pattern=r"^evt_[a-f0-9]{8}$")
    timestamp_ms: float = Field(ge=0)
    predicted_label: str
    corrected_label: str
    runtime_action: str = Field(min_length=1, max_length=32)
    confidence: float | None = Field(default=None, ge=0, le=1)
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_labels(self):
        if self.predicted_label not in LABELS or self.corrected_label not in LABELS:
            raise ValueError("unsupported event label")
        return self


class RecordingUpload(BaseModel):
    upload_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    participant_id: str = Field(min_length=1, max_length=32)
    session_id: str = Field(min_length=1, max_length=32)
    capture: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = Field(gt=0, le=30 * 60 * 1000)
    pose_frame_count: int = Field(ge=0)
    # Raw upper-body MediaPipe landmarks are kept with the video so the same
    # reviewed recording can later train rule tuning, Bi-LSTM, or TCN models.
    pose_frames: list[dict[str, Any]] = Field(default_factory=list, max_length=100_000)
    events: list[DetectionEvent] = Field(default_factory=list, max_length=1000)


def _safe_id(value: str, name: str) -> str:
    value = value.strip()
    if not SAFE_ID.fullmatch(value):
        raise HTTPException(422, detail=f"{name}는 영문, 숫자, _, -만 1~32자로 입력하세요.")
    return value


def create_app(output_dir: Path | str = OUTPUT_DIR) -> FastAPI:
    root = Path(output_dir)
    staging = root / ".video_staging"
    root.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
    app = FastAPI(title="Boxing Motion Dataset Collector v2")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="collector_v2.html",
            context={"labels": LABELS},
        )

    @app.get("/assets/punch_core.js")
    async def punch_core():
        if not PUNCH_CORE_PATH.exists():
            raise HTTPException(500, detail="게임 판정 로직(punch_core.js)을 찾을 수 없습니다.")
        return FileResponse(PUNCH_CORE_PATH, media_type="application/javascript")

    @app.get("/api/config")
    async def config():
        return {
            "labels": LABELS,
            "detector": "iter4_punch_core_rule_base",
            "video_required": True,
            "max_video_bytes": MAX_VIDEO_BYTES,
            "review_policy": "click event label to seek recorded video; save corrected_label",
        }

    @app.put("/api/video-staging/{upload_id}")
    async def stage_video(upload_id: str, request: Request):
        if not UPLOAD_ID.fullmatch(upload_id):
            raise HTTPException(422, detail="유효하지 않은 업로드 ID입니다.")
        content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type != "video/webm":
            raise HTTPException(415, detail="WebM 영상만 저장할 수 있습니다.")
        data = await request.body()
        if not data or len(data) > MAX_VIDEO_BYTES:
            raise HTTPException(413, detail="영상은 1 byte 이상 100MB 이하여야 합니다.")
        if not data.startswith(b"\x1aE\xdf\xa3"):
            raise HTTPException(422, detail="유효한 WebM 헤더가 아닙니다.")
        temp_path = staging / f".{upload_id}.tmp"
        final_path = staging / f"{upload_id}.webm"
        temp_path.write_bytes(data)
        os.replace(temp_path, final_path)
        return {"status": "staged", "bytes": len(data)}

    @app.post("/api/recordings")
    async def save_recording(recording: RecordingUpload):
        participant = _safe_id(recording.participant_id, "participant_id")
        session = _safe_id(recording.session_id, "session_id")
        staged = staging / f"{recording.upload_id}.webm"
        if not staged.is_file():
            raise HTTPException(422, detail="저장할 임시 영상을 찾을 수 없습니다.")

        now = datetime.now(timezone.utc)
        recording_id = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
        target_dir = root / participant / session / "reviewed_recordings"
        target_dir.mkdir(parents=True, exist_ok=True)
        video_path = target_dir / f"{recording_id}.webm"
        json_path = target_dir / f"{recording_id}.json"
        video_bytes = staged.stat().st_size
        document = {
            "schema_version": 2,
            "recording_id": recording_id,
            "participant_id": participant,
            "session_id": session,
            "created_at": now.isoformat(),
            "source": "boxing_motion_collector_v2",
            "detector": "iter4_punch_core_rule_base",
            "capture": recording.capture,
            "video": {
                "path": str(video_path.relative_to(root)).replace("\\", "/"),
                "mime_type": "video/webm",
                "bytes": video_bytes,
                "duration_ms": round(recording.duration_ms, 1),
            },
            "pose_frame_count": recording.pose_frame_count,
            "pose_frames": recording.pose_frames,
            "events": [event.model_dump() for event in recording.events],
            "review": {
                "status": "reviewed",
                "event_count": len(recording.events),
                "corrected_count": sum(e.predicted_label != e.corrected_label for e in recording.events),
                "labels": LABELS,
            },
        }
        temp_json = json_path.with_suffix(".tmp")
        temp_json.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            os.replace(staged, video_path)
            os.replace(temp_json, json_path)
        except Exception:
            temp_json.unlink(missing_ok=True)
            if video_path.exists() and not staged.exists():
                os.replace(video_path, staged)
            raise

        manifest_entry = {
            "recording_id": recording_id,
            "participant_id": participant,
            "session_id": session,
            "created_at": document["created_at"],
            "path": str(json_path.relative_to(root)).replace("\\", "/"),
            "video_path": document["video"]["path"],
            "duration_ms": document["video"]["duration_ms"],
            "event_count": len(recording.events),
            "corrected_count": document["review"]["corrected_count"],
        }
        with MANIFEST_LOCK:
            with (root / "manifest.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
        return {"status": "saved", **manifest_entry}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8012)
