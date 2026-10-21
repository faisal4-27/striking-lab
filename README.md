# Striking Lab

Record or upload a soccer shot (instep drive) and get a score from 0 to 100%, a breakdown of your technique and the ball's flight, and the top things to fix, each with a coaching tip.

It's a machine-learning project in two layers:

1. **Perception (pretrained deep nets):** body pose in 2D and 3D, ball detection, and camera-motion tracking. These turn a video into numbers.
2. **Judgement (learns from you):** a biomechanics rule engine scores the shot on day one. As you rate your own shots, a LightGBM model, and later an ST-GCN sequence network, learns what *you* consider a good shot and blends into the score.

```
phone browser ──record/upload──▶ FastAPI ──▶ GPU worker
                                              │ 1  ffmpeg: constant-fps H.264 (same file the browser plays → overlay in sync)
                                              │ 2  RTMW-x wholebody pose, every frame (23 body+feet keypoints)
                                              │ 3  YOLO11 ball candidates → tracklets → stitched trajectory
                                              │ 4  ORB + RANSAC camera motion (handheld phone) → stabilised coords
                                              │ 5  events: approach · plant · backswing · contact · follow-through
                                              │ 6  MediaPipe 3D landmarks around contact → view-invariant joint angles
                                              │ 7  17 features: biomechanics + ball speed/launch/accuracy
                                              │ 8  rules.yaml ⊕ LightGBM ⊕ ST-GCN → score, ranked fixes, tips
                                              ▼
                                   data/shots/<id>/ (video, tracks.npz cache, analysis.json)
```

## Setup (Windows, NVIDIA GPU)

Already done on this PC; listed here for a fresh machine:

```powershell
winget install astral-sh.uv Gyan.FFmpeg      # Python package manager + ffmpeg
cd backend;  uv sync                         # Python 3.12 venv, CUDA 13 PyTorch, ONNX Runtime GPU, ...
cd ..\frontend; npm install
```

Model weights (RTMW, YOLO11m, MediaPipe pose) download automatically on first analysis, about 400 MB.

## Run

```powershell
.\scripts\dev.ps1
```

- On the PC, open **https://localhost:5173**.
- On your phone, connect to the same Wi-Fi and open **https://\<PC-IP\>:5173**. The dev server prints the IP.
- The certificate is self-signed, so accept the warning once. HTTPS is required because phone browsers only allow camera access on secure pages.
- If Windows Firewall asks, allow Node.js on private networks.

From the command line:

```powershell
cd backend
.venv\Scripts\python -m pipeline.cli analyze path\to\clip.mp4      # prints the full report
.venv\Scripts\python -m pipeline.debug_render data\cli\clip --sheet  # overlay video + contact sheet
.venv\Scripts\python -m pytest                                      # 14 tests on synthetic kicks
```

## Filming for good results

- **Side view**, 5–10 m away, camera at about ball height, whole body including the feet in frame.
- Start before the run-up and stop after the ball reaches the target.
- **60 fps** if your phone offers it, since contact lasts only about 10 ms. Hold the phone steady in landscape.
- Keep the goal or target in frame, then use **Mark target** on the result page to score accuracy.

The app detects its own limitations. If you film from behind it detects that view (from the 3D pose) and skips side-only metrics rather than reporting wrong ones. Each result also shows a confidence level and filming tips.

## Training it on your shots

| Step | When | Command (in `backend/`) |
|---|---|---|
| Rate shots on the **Label & train** page (0–100 plus optional aspect ratings). The app's score stays hidden until you rate, so it can't bias you. | always | — |
| **Shot scorer** (LightGBM on the 17 features plus rule scores). | ≥ 30 ratings; good from ~150 | `.venv\Scripts\python -m training.train_scorer` |
| **Ball detector** fine-tune (biggest accuracy win for fast, blurry shots). | ~20+ clips | `python -m training.export_dataset ball`, review the pseudo-labels in Label Studio or CVAT, then `python -m training.train_ball` |
| **Sequence model** (ST-GCN on the joint trajectories). | ~300+ ratings | `.venv\Scripts\python -m training.train_seq` |

Training reports cross-validated error next to the rule engine's error on the same shots. The model's share of the final score grows with your data and with how much it beats the rules, up to 85%. If you rate shots from several players, validation holds out whole players, so the reported error reflects people the model hasn't seen.

After fine-tuning the ball detector, use **Re-run from scratch** on old shots to re-track them.

## Code map

| Path | What |
|---|---|
| `backend/pipeline/` | perception and measurement: `video`, `pose`, `ball`, `stabilize`, `events`, `calibration`, `features`, `analyze` (orchestrator) |
| `backend/scoring/rules.yaml` | ideal ranges, weights and coaching text. Tune the coaching here. |
| `backend/scoring/` | `rules.py` rule engine, `model.py` LightGBM blend, `seq.py` ST-GCN, `feedback.py` ranked report |
| `backend/training/` | dataset export and the three trainers |
| `backend/app/` | FastAPI API, SQLite (SQLModel), single-GPU job worker |
| `backend/tests/synth.py` | procedural kick generator with known ground truth, used by the tests |
| `frontend/src/` | React app: Record, Upload, Result (video + skeleton/ball overlay), History, Label |

## Known limitations

- Single camera, so plant-foot *lateral* distance (into the screen) isn't measured; plant position is measured fore/aft.
- Ball speed comes from the ball's apparent size and position (pinhole model). It's most accurate in side views, and when the depth fit is poor it falls back to in-plane speed (a lower bound).
- On-target detection needs the ball tracked until it reaches the marked target.
- Analysis takes about 30–40 s for a 5 s clip on a laptop RTX 4050. Plug the laptop in and use a performance power mode; on battery the GPU throttles hard.
