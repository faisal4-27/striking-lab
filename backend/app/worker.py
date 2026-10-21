"""Single background worker: one GPU, one job at a time, FIFO."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import traceback

from sqlmodel import select

from pipeline.analyze import analyze
from pipeline.video import normalize

from .db import Shot, session

log = logging.getLogger(__name__)


class Worker:
    def __init__(self):
        self.q: queue.Queue[tuple[str, bool]] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="analysis-worker", daemon=True)

    def start(self) -> None:
        # resume anything interrupted by a restart
        with session() as s:
            for shot in s.exec(select(Shot).where(Shot.status.in_(["queued", "processing"]))):
                self.q.put((shot.id, False))
        self.thread.start()

    def submit(self, shot_id: str, fresh: bool = False) -> None:
        with session() as s:
            shot = s.get(Shot, shot_id)
            shot.status, shot.progress, shot.stage, shot.error = "queued", 0.0, "Waiting in queue", None
            s.add(shot)
            s.commit()
        self.q.put((shot_id, fresh))

    def _update(self, shot_id: str, **fields) -> None:
        with session() as s:
            shot = s.get(Shot, shot_id)
            if shot is None:
                return
            for k, v in fields.items():
                setattr(shot, k, v)
            s.add(shot)
            s.commit()

    def _run(self) -> None:
        while True:
            shot_id, fresh = self.q.get()
            try:
                self._process(shot_id, fresh)
            except Exception as e:  # keep the worker alive
                log.exception("analysis failed for %s", shot_id)
                self._update(shot_id, status="failed", error=f"{type(e).__name__}: {e}",
                             stage=traceback.format_exc(limit=1).strip().splitlines()[-1][:200])
            finally:
                self.q.task_done()

    def _process(self, shot_id: str, fresh: bool) -> None:
        with session() as s:
            shot = s.get(Shot, shot_id)
            if shot is None:
                return
            d, target, manual = shot.dir, shot.target(), shot.manual()
            height, foot = shot.player_height_m, shot.foot_hint
        self._update(shot_id, status="processing", progress=0.01, stage="Preparing video")

        playback, info_path = d / "playback.mp4", d / "video.json"
        if fresh or not playback.exists():
            src = next(p for p in d.iterdir() if p.stem == "original")
            info_path.write_text(json.dumps(normalize(src, playback)))
        info = json.loads(info_path.read_text())

        last = [0.0]

        def progress(p: float, msg: str) -> None:
            if time.time() - last[0] > 0.5 or p >= 1.0:
                last[0] = time.time()
                self._update(shot_id, progress=round(p, 3), stage=msg)

        res = analyze(playback, info, d, target=target, manual=manual, player_height_m=height,
                      foot_hint=foot, progress=progress, reuse=not fresh)
        save_thumb(playback, res["events"]["contact"], d / "thumb.jpg")
        rep = res["report"]
        g = rep["rules"]["groups"]
        self._update(shot_id, status="done", progress=1.0, stage="Done", overall=rep["overall"],
                     technique=g.get("technique"), outcome=g.get("outcome"), confidence=rep["confidence"],
                     kicking_foot=res["events"]["kicking_foot"])


def save_thumb(video, frame_idx: int, dest, width: int = 360) -> None:
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if ok:
        h, w = frame.shape[:2]
        cv2.imwrite(str(dest), cv2.resize(frame, (width, int(h * width / w))), [cv2.IMWRITE_JPEG_QUALITY, 80])


worker = Worker()
