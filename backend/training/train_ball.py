"""Fine-tune the ball detector on frames from your own footage.

    python -m training.export_dataset ball      # then review labels!
    python -m training.train_ball [--epochs 80] [--model yolo11m.pt]

Starts from COCO-pretrained YOLO11 (which already knows "sports ball") and
adapts it to your ball, pitch, lighting and motion blur. The best weights are
copied to models/ball.pt, which the pipeline picks up automatically.
Sized for a 6 GB GPU: imgsz 960, batch 8.
"""
from __future__ import annotations

import argparse
import shutil

from scoring.model import MODELS_DIR

from .export_dataset import DATASETS


def main() -> None:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--model", default=str(MODELS_DIR / "yolo11m.pt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    data = DATASETS / "ball" / "data.yaml"
    if not data.exists():
        raise SystemExit("No dataset. Run: python -m training.export_dataset ball")
    model = YOLO(args.model)
    res = model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                      project=str(DATASETS / "ball_runs"), name="ft", exist_ok=True,
                      patience=20, cos_lr=True, close_mosaic=10,
                      # blur/scale augmentation mimics fast shots; no vertical flips (gravity matters)
                      degrees=5, scale=0.5, fliplr=0.5, flipud=0.0, mosaic=1.0)
    best = res.save_dir / "weights" / "best.pt"
    shutil.copy(best, MODELS_DIR / "ball.pt")
    print(f"Saved {MODELS_DIR / 'ball.pt'}. Restart the server; re-run analyses with 'Re-run from scratch'.")


if __name__ == "__main__":
    main()
