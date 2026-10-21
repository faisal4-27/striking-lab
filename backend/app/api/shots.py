from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlmodel import select

from scoring import model as learned

from ..db import Label, Shot, now, session
from ..worker import worker

router = APIRouter(prefix="/api")

ALLOWED = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi", ".ogv", ".3gp"}
MAX_BYTES = 300 * 1024 * 1024


def _shot_or_404(s, shot_id: str) -> Shot:
    shot = s.get(Shot, shot_id)
    if shot is None:
        raise HTTPException(404, "shot not found")
    return shot


def _label_dict(label: Label | None) -> dict | None:
    if label is None:
        return None
    return {"overall": label.overall, "aspects": json.loads(label.aspects_json), "on_target": label.on_target,
            "notes": label.notes, "updated_at": label.updated_at}


def _shot_dict(shot: Shot, label: Label | None = None) -> dict:
    d = shot.model_dump()
    d["target"] = shot.target()
    d["manual"] = shot.manual()
    d.pop("target_json"), d.pop("manual_json")
    d["label"] = _label_dict(label)
    return d


@router.post("/shots")
async def create_shot(file: UploadFile = File(...), player: str | None = Form(None),
                      foot_hint: str | None = Form(None), player_height_m: float = Form(1.75)):
    ext = Path(file.filename or "clip.webm").suffix.lower() or ".webm"
    if ext not in ALLOWED:
        raise HTTPException(400, f"unsupported file type {ext}")
    if foot_hint not in (None, "", "left", "right"):
        raise HTTPException(400, "foot_hint must be left or right")
    shot = Shot(id=uuid.uuid4().hex[:12], player=(player or None), original_name=file.filename,
                foot_hint=foot_hint or None, player_height_m=float(min(max(player_height_m, 1.0), 2.3)))
    shot.dir.mkdir(parents=True, exist_ok=True)
    dest = shot.dir / f"original{ext}"
    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                f.close()
                shutil.rmtree(shot.dir, ignore_errors=True)
                raise HTTPException(413, "video too large (max 300 MB)")
            f.write(chunk)
    with session() as s:
        s.add(shot)
        s.commit()
        s.refresh(shot)
        out = _shot_dict(shot)
    worker.submit(shot.id)
    return out


@router.get("/shots")
def list_shots():
    with session() as s:
        shots = s.exec(select(Shot).order_by(Shot.created_at.desc())).all()
        labels = {l.shot_id: l for l in s.exec(select(Label)).all()}
        return [_shot_dict(sh, labels.get(sh.id)) for sh in shots]


@router.get("/shots/{shot_id}")
def get_shot(shot_id: str):
    with session() as s:
        shot = _shot_or_404(s, shot_id)
        return _shot_dict(shot, s.get(Label, shot_id))


@router.get("/shots/{shot_id}/analysis")
def get_analysis(shot_id: str):
    with session() as s:
        shot = _shot_or_404(s, shot_id)
    path = shot.dir / "analysis.json"
    if not path.exists():
        raise HTTPException(404, "analysis not ready")
    return Response(path.read_bytes(), media_type="application/json")


@router.get("/shots/{shot_id}/video")
def get_video(shot_id: str):
    with session() as s:
        shot = _shot_or_404(s, shot_id)
    path = shot.dir / "playback.mp4"
    if not path.exists():
        raise HTTPException(404, "video not ready")
    return FileResponse(path, media_type="video/mp4")


@router.get("/shots/{shot_id}/thumb")
def get_thumb(shot_id: str):
    with session() as s:
        shot = _shot_or_404(s, shot_id)
    path = shot.dir / "thumb.jpg"
    if not path.exists():
        raise HTTPException(404, "thumbnail not ready")
    return FileResponse(path, media_type="image/jpeg")


class TargetIn(BaseModel):
    frame: int = Field(ge=0)
    points: list[list[float]] = Field(min_length=3, max_length=8)


@router.put("/shots/{shot_id}/target")
def set_target(shot_id: str, target: TargetIn | None = None):
    with session() as s:
        shot = _shot_or_404(s, shot_id)
        shot.target_json = target.model_dump_json() if target else None
        s.add(shot)
        s.commit()
    worker.submit(shot_id)
    return {"ok": True}


class CorrectionsIn(BaseModel):
    contact: int | None = Field(None, ge=0)
    plant: int | None = Field(None, ge=0)
    foot_hint: str | None = None
    player_height_m: float | None = Field(None, ge=1.0, le=2.3)


@router.put("/shots/{shot_id}/corrections")
def set_corrections(shot_id: str, c: CorrectionsIn):
    if c.foot_hint not in (None, "left", "right"):
        raise HTTPException(400, "foot_hint must be left or right")
    with session() as s:
        shot = _shot_or_404(s, shot_id)
        manual = {k: v for k, v in {"contact": c.contact, "plant": c.plant}.items() if v is not None}
        shot.manual_json = json.dumps(manual) if manual else None
        shot.foot_hint = c.foot_hint
        if c.player_height_m:
            shot.player_height_m = c.player_height_m
        s.add(shot)
        s.commit()
    worker.submit(shot_id)
    return {"ok": True}


@router.post("/shots/{shot_id}/reanalyze")
def reanalyze(shot_id: str, fresh: bool = False):
    with session() as s:
        _shot_or_404(s, shot_id)
    worker.submit(shot_id, fresh=fresh)
    return {"ok": True}


class LabelIn(BaseModel):
    overall: int = Field(ge=0, le=100)
    aspects: dict[str, int] = {}
    on_target: bool | None = None
    notes: str | None = None


@router.put("/shots/{shot_id}/label")
def set_label(shot_id: str, body: LabelIn):
    if any(not 1 <= v <= 5 for v in body.aspects.values()):
        raise HTTPException(400, "aspect ratings must be 1-5")
    with session() as s:
        _shot_or_404(s, shot_id)
        label = s.get(Label, shot_id) or Label(shot_id=shot_id, overall=body.overall)
        label.overall, label.on_target, label.notes = body.overall, body.on_target, body.notes
        label.aspects_json = json.dumps(body.aspects)
        label.updated_at = now()
        s.add(label)
        s.commit()
        s.refresh(label)
        return _label_dict(label)


@router.delete("/shots/{shot_id}")
def delete_shot(shot_id: str):
    with session() as s:
        shot = _shot_or_404(s, shot_id)
        if (label := s.get(Label, shot_id)) is not None:
            s.delete(label)
        s.delete(shot)
        s.commit()
    shutil.rmtree(shot.dir, ignore_errors=True)
    return {"ok": True}


@router.get("/model")
def model_status():
    with session() as s:
        n_labels = len(s.exec(select(Label.shot_id)).all())
    return {**learned.status(), "n_labels": n_labels}


@router.post("/model/reload")
def model_reload():
    from scoring import seq
    learned.reload()
    seq.reload()
    return model_status()
