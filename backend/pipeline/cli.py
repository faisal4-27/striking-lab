"""Analyse a clip from the command line.

    python -m pipeline.cli analyze path/to/clip.mp4 [--out data/cli/<name>] [--fresh]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .analyze import analyze
from .video import normalize


def main() -> None:
    ap = argparse.ArgumentParser(prog="pipeline.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("video", type=Path)
    a.add_argument("--out", type=Path)
    a.add_argument("--fresh", action="store_true", help="ignore cached tracks")
    a.add_argument("--height", type=float, default=1.75, help="player height (m)")
    a.add_argument("--foot", choices=["left", "right"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    out = args.out or Path("data/cli") / args.video.stem
    out.mkdir(parents=True, exist_ok=True)
    norm = out / "playback.mp4"
    info_path = out / "video.json"
    if args.fresh or not norm.exists():
        info = normalize(args.video, norm)
        info_path.write_text(json.dumps(info))
    else:
        info = json.loads(info_path.read_text())

    res = analyze(norm, info, out, player_height_m=args.height, foot_hint=args.foot, reuse=not args.fresh,
                  progress=lambda p, m: print(f"\r{p * 100:5.1f}%  {m:<40}", end="", flush=True))
    print()
    rep = res["report"]
    print(f"Overall {rep['overall']}%  (technique {rep['rules']['groups'].get('technique')}, "
          f"outcome {rep['rules']['groups'].get('outcome')})  confidence {rep['confidence']}")
    print("Events:", res["events"])
    print("Flight:", res["flight"])
    for s in rep["rules"]["subscores"]:
        print(f"  {s['label']:<26} {s['value']!s:>8}{s['unit']:<6} -> {s['score']:5.1f} {s['status']}")
    for i in rep["top_fixes"]:
        print(f"FIX: {i['issue']}\n     {i['tip']}")
    print(f"({res['elapsed_s']} s)")


if __name__ == "__main__":
    main()
