from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Field, Session, SQLModel, create_engine

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SHOTS_DIR = DATA_DIR / "shots"
SHOTS_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DATA_DIR / 'striking_lab.db'}", connect_args={"check_same_thread": False})


def now() -> datetime:
    return datetime.now(timezone.utc)


class Shot(SQLModel, table=True):
    id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=now)
    player: str | None = None
    original_name: str | None = None
    status: str = "queued"            # queued | processing | done | failed
    progress: float = 0.0
    stage: str | None = None
    error: str | None = None
    # summary of the latest analysis
    overall: float | None = None
    technique: float | None = None
    outcome: float | None = None
    confidence: float | None = None
    kicking_foot: str | None = None
    # analysis inputs the user can change
    foot_hint: str | None = None
    player_height_m: float = 1.75
    target_json: str | None = None    # {"frame": k, "points": [[x,y] x4]} normalised
    manual_json: str | None = None    # {"contact": i, "plant": j}

    @property
    def dir(self) -> Path:
        return SHOTS_DIR / self.id

    def target(self) -> dict | None:
        return json.loads(self.target_json) if self.target_json else None

    def manual(self) -> dict | None:
        return json.loads(self.manual_json) if self.manual_json else None


class Label(SQLModel, table=True):
    """Your rating of a shot: the training target for the learned scorer."""
    shot_id: str = Field(primary_key=True, foreign_key="shot.id")
    overall: int                       # 0..100
    aspects_json: str = "{}"           # {"approach": 1-5, "plant": ..., ...}
    on_target: bool | None = None
    notes: str | None = None
    updated_at: datetime = Field(default_factory=now)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def session() -> Session:
    return Session(engine)
