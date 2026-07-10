import asyncio
import json
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .controller import RouletteController
from .settings import settings

app = FastAPI(title="Roulette Vision Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = RouletteController()


class TargetRequest(BaseModel):
    target_number: int


class RoiRequest(BaseModel):
    x: int
    y: int
    w: int
    h: int


class StartSessionRequest(BaseModel):
    mode: str = "real"
    targetNumber: int
    cameraEnabled: bool = True
    soundEnabled: bool = True




class RobotSettingsRequest(BaseModel):
    cameraEnabled: bool = True
    soundEnabled: bool = True
    armEnabled: bool = True
    soundDelaySeconds: float = 0.0
    armDelaySeconds: float = 0.0
    armExtensionPercent: int = 100

class TestReadingRequest(BaseModel):
    targetNumber: int
    forceMatch: bool = False


@app.on_event("startup")
def on_startup():
    if settings.run_on_start:
        controller.start()


@app.on_event("shutdown")
def on_shutdown():
    controller.stop()


@app.get("/")
def root():
    return {
        "name": "Roulette Vision Backend",
        "docs": "/docs",
        "legacy_status": "/api/status",
        "system_status": "/api/v1/system/status",
        "video": "/api/video",
        "live": "/api/v1/live",
    }


# ---------------------------------------------------------------------------
# Legacy API kept for compatibility with the original backend.
# ---------------------------------------------------------------------------
@app.get("/api/status")
def status():
    return controller.status()


@app.post("/api/start")
def start():
    controller.start()
    return {"ok": True, "status": controller.status()}


@app.post("/api/stop")
def stop():
    controller.stop()
    return {"ok": True, "status": controller.status()}


@app.post("/api/target")
def set_target(payload: TargetRequest):
    try:
        controller.set_target_number(payload.target_number)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "target_number": payload.target_number}


@app.post("/api/roi")
def set_roi(payload: RoiRequest):
    controller.set_roi(payload.x, payload.y, payload.w, payload.h)
    return {"ok": True, "roi": payload.model_dump()}


@app.post("/api/trigger")
def manual_trigger():
    ok = controller.manual_trigger()
    if not ok:
        raise HTTPException(status_code=500, detail=controller.status().get("last_trigger_error"))
    return {"ok": True}


@app.get("/api/frame.jpg")
def frame_jpg():
    jpeg = controller.get_jpeg()
    if jpeg is None:
        return Response(status_code=404, content=b"No frame available yet")
    return Response(content=jpeg, media_type="image/jpeg")


def mjpeg_generator():
    while True:
        jpeg = controller.get_jpeg()
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(0.1)


@app.get("/api/video")
def video():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Frontend v1 API.
# ---------------------------------------------------------------------------
@app.get("/api/v1/system/status")
def v1_system_status():
    return controller.system_status()


@app.post("/api/v1/sessions")
def v1_start_session(payload: StartSessionRequest):
    try:
        return controller.start_session(
            mode=payload.mode,
            target_number=payload.targetNumber,
            camera_enabled=payload.cameraEnabled,
            sound_enabled=payload.soundEnabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/sessions/{session_id}/stop")
def v1_stop_session(session_id: str):
    return controller.stop_session(session_id)


@app.get("/api/v1/readings")
def v1_readings(
    sessionId: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=2000),
):
    return controller.list_readings(session_id=sessionId, limit=limit)


@app.get("/api/v1/live/latest")
def v1_latest_reading():
    latest = controller.get_latest_reading()
    if latest is None:
        return {"reading": None, "status": controller.system_status()}
    return {"reading": latest, "status": controller.system_status()}


@app.post("/api/v1/admin/readings/clear")
def v1_clear_readings():
    controller.clear_readings()
    return {"ok": True}


@app.post("/api/v1/admin/test-reading")
def v1_test_reading(payload: TestReadingRequest):
    try:
        return controller.create_test_reading(
            target_number=payload.targetNumber,
            force_match=payload.forceMatch,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/api/v1/live")
async def v1_live(websocket: WebSocket, sessionId: str | None = None):
    await websocket.accept()
    last_reading_id: str | None = None
    last_status_payload: dict[str, Any] | None = None

    try:
        while True:
            latest = controller.get_latest_reading()
            if latest and latest.get("id") != last_reading_id:
                if sessionId is None or latest.get("sessionId") in {None, sessionId}:
                    await websocket.send_text(
                        json.dumps({"type": "reading.validated", "payload": latest})
                    )
                    last_reading_id = latest.get("id")

            status_payload = controller.system_status()
            if status_payload != last_status_payload:
                await websocket.send_text(
                    json.dumps({"type": "system.status.changed", "payload": status_payload})
                )
                last_status_payload = status_payload

            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return




@app.get("/api/v1/settings")
def v1_get_settings():
    return controller.get_robot_settings()


@app.post("/api/v1/settings")
def v1_set_settings(payload: RobotSettingsRequest):
    return controller.set_robot_settings(payload.model_dump())


@app.post("/api/v1/pico/test")
def v1_pico_test():
    ok = controller.send_pico_test()
    if not ok:
        raise HTTPException(status_code=500, detail=controller.status().get("last_trigger_error"))
    return {"ok": True}


# Minimal analysis endpoints so the existing Analysis page does not fail when
# backend mode is enabled. They can be expanded later with richer statistics.
@app.post("/api/v1/analysis/run")
def v1_analysis_run(payload: dict[str, Any]):
    target = int(payload.get("targetNumber", controller.status().get("target_number", 0)))
    readings = controller.list_readings(limit=2000)
    total = len(readings)
    counts = {number: 0 for number in range(37)}
    for reading in readings:
        counts[int(reading["number"])] += 1

    most = max(counts, key=counts.get) if total else None
    non_zero_counts = {number: count for number, count in counts.items() if count > 0}
    least = min(non_zero_counts, key=non_zero_counts.get) if non_zero_counts else None

    frequencies = []
    for number in range(37):
        count = counts[number]
        frequencies.append(
            {
                "number": number,
                "count": count,
                "percentage": (count / total * 100.0) if total else 0.0,
                "currentGap": 0,
                "averageGap": 0,
                "maxGap": 0,
                "trend": "stable",
            }
        )

    match_count = sum(1 for item in readings if item.get("isMatch"))
    average_confidence = (
        sum(float(item.get("confidence", 0.0)) for item in readings) / total if total else 0.0
    )

    return {
        "summary": {
            "totalReadings": total,
            "uniqueNumbers": len(non_zero_counts),
            "mostFrequentNumber": most,
            "leastFrequentNumber": least,
            "matchCount": match_count,
            "averageConfidence": average_confidence,
            "from": payload.get("range", {}).get("from", ""),
            "to": payload.get("range", {}).get("to", ""),
        },
        "frequencies": frequencies,
        "target": {
            "targetNumber": target,
            "count": counts.get(target, 0),
            "percentage": (counts.get(target, 0) / total * 100.0) if total else 0.0,
            "currentGap": 0,
            "averageGap": 0,
            "maxGap": 0,
            "recentAppearances": counts.get(target, 0),
            "interpretation": "Analyse basée sur les lectures réelles sauvegardées.",
        },
        "distribution": {
            "red": 0,
            "black": 0,
            "green": counts.get(0, 0),
            "even": sum(counts[n] for n in range(2, 37, 2)),
            "odd": sum(counts[n] for n in range(1, 37, 2)),
            "low": sum(counts[n] for n in range(1, 19)),
            "high": sum(counts[n] for n in range(19, 37)),
            "dozen1": sum(counts[n] for n in range(1, 13)),
            "dozen2": sum(counts[n] for n in range(13, 25)),
            "dozen3": sum(counts[n] for n in range(25, 37)),
            "column1": 0,
            "column2": 0,
            "column3": 0,
        },
        "trends": [],
        "quality": {
            "averageConfidence": average_confidence,
            "lowConfidenceCount": 0,
            "validatedCount": total,
            "rejectedCount": 0,
            "averageConsensusFrames": settings.stable_frames,
            "qualityLabel": "bonne" if total else "faible",
        },
        "sequences": {
            "repeatedNumberRuns": 0,
            "longestSameColorRun": 0,
            "longestNoZeroRun": 0,
            "recentClusterNumber": most,
            "recentClusterCount": counts.get(most, 0) if most is not None else 0,
        },
    }


@app.post("/api/v1/analysis/import")
def v1_analysis_import():
    raise HTTPException(status_code=501, detail="File analysis import is not implemented yet")


@app.post("/api/v1/export/analysis")
def v1_export_analysis():
    content = "Export is not implemented yet. Use /api/v1/readings for raw readings.\n"
    return Response(content=content, media_type="text/plain")
