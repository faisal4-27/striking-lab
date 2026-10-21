import { useCallback, useEffect, useImperativeHandle, useRef, useState, forwardRef } from "react";
import type { Analysis, Target } from "../api";

// keypoint indices (pipeline/skeleton.py)
const LEFT_LEG = new Set([11, 13, 15, 17, 18, 19]);
const RIGHT_LEG = new Set([12, 14, 16, 20, 21, 22]);

const EVENT_LABELS: [keyof Analysis["events"], string][] = [
  ["approach_start", "Approach"],
  ["plant", "Plant"],
  ["backswing", "Backswing"],
  ["contact", "Contact"],
  ["follow_through", "Follow-through"],
];

export interface ShotPlayerHandle {
  seekFrame: (f: number) => void;
  frame: () => number;
}

interface Props {
  videoUrl: string;
  analysis: Analysis | null;
  target?: Target | null;
  markingTarget?: boolean;
  onTargetDone?: (t: Target) => void;
  onFrame?: (f: number) => void;
}

function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Video + canvas overlay of the tracked skeleton, ball trail and kick phases. */
export const ShotPlayer = forwardRef<ShotPlayerHandle, Props>(function ShotPlayer(
  { videoUrl, analysis, target, markingTarget, onTargetDone, onFrame }, ref) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [slow, setSlow] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [pending, setPending] = useState<[number, number][]>([]);
  const fps = analysis?.video.fps ?? 30;
  const nFrames = analysis?.overlay.kp.length ?? 0;

  const frameAt = (t: number) => Math.max(0, Math.min(Math.round(t * fps), Math.max(nFrames - 1, 0)));

  const seekFrame = useCallback((f: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.pause();
    // aim for the middle of the frame interval so decoders land on the right frame
    v.currentTime = (f + 0.5) / fps;
    setFrame(f);
  }, [fps]);

  useImperativeHandle(ref, () => ({ seekFrame, frame: () => frame }), [seekFrame, frame]);

  // Displayed video rect inside the element (object-fit: contain).
  const contentRect = () => {
    const v = videoRef.current!;
    const W = v.clientWidth, H = v.clientHeight;
    const vw = v.videoWidth || analysis?.video.width || 16, vh = v.videoHeight || analysis?.video.height || 9;
    const s = Math.min(W / vw, H / vh);
    return { x: (W - vw * s) / 2, y: (H - vh * s) / 2, s, vw, vh };
  };

  const draw = useCallback((f: number) => {
    const c = canvasRef.current, v = videoRef.current;
    if (!c || !v) return;
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== v.clientWidth * dpr || c.height !== v.clientHeight * dpr) {
      c.width = v.clientWidth * dpr;
      c.height = v.clientHeight * dpr;
    }
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, c.width, c.height);
    const { x: ox, y: oy, s } = contentRect();
    const P = (x: number, y: number): [number, number] => [ox + x * s, oy + y * s];

    // target polygon (stored normalised)
    const poly = markingTarget ? pending : target?.points;
    if (poly?.length) {
      const { vw, vh } = contentRect();
      ctx.strokeStyle = cssVar("--accent");
      ctx.fillStyle = "rgba(61,220,132,0.15)";
      ctx.lineWidth = 2;
      ctx.setLineDash(markingTarget ? [6, 4] : []);
      ctx.beginPath();
      poly.forEach(([nx, ny], i) => {
        const [px, py] = P(nx * vw, ny * vh);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      if (!markingTarget || poly.length === 4) ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
      poly.forEach(([nx, ny]) => {
        const [px, py] = P(nx * vw, ny * vh);
        ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.fillStyle = cssVar("--accent"); ctx.fill();
      });
    }

    if (!analysis || !showOverlay) return;
    const ov = analysis.overlay;
    const kick = analysis.events.kicking_foot;
    const kickSet = kick === "left" ? LEFT_LEG : RIGHT_LEG;
    const kp = ov.kp[f];
    if (kp) {
      ctx.lineWidth = 3;
      ctx.lineCap = "round";
      for (const [a, b] of ov.edges) {
        const pa = kp[a], pb = kp[b];
        if (pa?.[0] == null || pb?.[0] == null) continue;
        const isKick = kickSet.has(a) && kickSet.has(b);
        const isLeg = (LEFT_LEG.has(a) || RIGHT_LEG.has(a)) && (LEFT_LEG.has(b) || RIGHT_LEG.has(b));
        ctx.strokeStyle = isKick ? cssVar("--kick") : isLeg ? cssVar("--support") : "rgba(255,255,255,0.75)";
        ctx.beginPath();
        ctx.moveTo(...P(pa[0]!, pa[1]!));
        ctx.lineTo(...P(pb[0]!, pb[1]!));
        ctx.stroke();
      }
      for (const p of kp) {
        if (p[0] == null) continue;
        ctx.beginPath(); ctx.arc(...P(p[0], p[1]!), 2.5, 0, Math.PI * 2);
        ctx.fillStyle = "#fff"; ctx.fill();
      }
    }
    // ball trail
    const ballCol = cssVar("--ball");
    for (let i = Math.max(0, f - 12); i <= f; i++) {
      const b = ov.ball[i];
      if (!b || b[0] == null) continue;
      const alpha = (i - (f - 12)) / 13;
      ctx.beginPath();
      const r = i === f ? Math.max((b[2] ?? 8) * s, 5) : 3;
      ctx.arc(...P(b[0], b[1]!), r, 0, Math.PI * 2);
      ctx.strokeStyle = ballCol;
      ctx.globalAlpha = i === f ? 1 : alpha * 0.8;
      ctx.lineWidth = i === f ? 2.5 : 2;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }, [analysis, showOverlay, target, markingTarget, pending]);

  // Frame-accurate redraw while playing.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    let handle = 0;
    const hasRVFC = "requestVideoFrameCallback" in v;
    const tick = (_now: number, meta?: { mediaTime: number }) => {
      const f = frameAt(meta?.mediaTime ?? v.currentTime);
      setFrame(f);
      draw(f);
      handle = hasRVFC ? (v as any).requestVideoFrameCallback(tick) : requestAnimationFrame((n) => tick(n));
    };
    handle = hasRVFC ? (v as any).requestVideoFrameCallback(tick) : requestAnimationFrame((n) => tick(n));
    return () => (hasRVFC ? (v as any).cancelVideoFrameCallback(handle) : cancelAnimationFrame(handle));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draw, fps, nFrames]);

  useEffect(() => { draw(frame); onFrame?.(frame); }, [frame, draw]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const onResize = () => draw(frame);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [draw, frame]);
  useEffect(() => { if (videoRef.current) videoRef.current.playbackRate = slow ? 0.25 : 1; }, [slow]);
  useEffect(() => { if (!markingTarget) setPending([]); }, [markingTarget]);

  const onCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!markingTarget) {
      togglePlay();
      return;
    }
    const rect = e.currentTarget.getBoundingClientRect();
    const { x, y, s, vw, vh } = contentRect();
    const nx = (e.clientX - rect.left - x) / s / vw;
    const ny = (e.clientY - rect.top - y) / s / vh;
    if (nx < 0 || ny < 0 || nx > 1 || ny > 1) return;
    const next = [...pending, [nx, ny] as [number, number]];
    setPending(next);
    if (next.length === 4) onTargetDone?.({ frame, points: next });
  };

  const togglePlay = () => {
    const v = videoRef.current!;
    if (v.paused) { v.play(); } else { v.pause(); }
  };

  const ev = analysis?.events;
  return (
    <div className="flex flex-col gap-3">
      <div className="relative w-full overflow-hidden rounded-2xl bg-black" style={{ aspectRatio: analysis ? `${analysis.video.width}/${analysis.video.height}` : "16/9", maxHeight: "70vh" }}>
        <video
          ref={videoRef}
          src={videoUrl}
          className="absolute inset-0 h-full w-full object-contain"
          playsInline
          muted
          preload="auto"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onLoadedData={() => draw(frame)}
        />
        <canvas ref={canvasRef} onClick={onCanvasClick}
          className={`absolute inset-0 h-full w-full ${markingTarget ? "cursor-crosshair" : "cursor-pointer"}`} />
        {markingTarget && (
          <div className="absolute left-3 top-3 chip" style={{ background: "var(--accent)", color: "var(--accent-ink)" }}>
            Tap the 4 corners of the target ({pending.length}/4)
          </div>
        )}
        <div className="absolute right-3 top-3 chip tabular-nums">frame {frame}</div>
      </div>

      {/* timeline with phase ticks */}
      {nFrames > 0 && (
        <div className="relative h-8 select-none">
          <input type="range" min={0} max={nFrames - 1} value={frame} aria-label="Scrub frames"
            onChange={(e) => seekFrame(Number(e.target.value))}
            className="absolute inset-x-0 top-1 w-full accent-[var(--accent)]" />
          {ev && EVENT_LABELS.map(([k]) => (
            <div key={k} className="absolute top-5 h-2 w-0.5 rounded"
              style={{ left: `${((ev[k] as number) / Math.max(nFrames - 1, 1)) * 100}%`, background: k === "contact" ? "var(--kick)" : "var(--muted)" }} />
          ))}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <button className="btn btn-primary" onClick={togglePlay}>{playing ? "Pause" : "Play"}</button>
        <button className="btn" onClick={() => seekFrame(Math.max(0, frame - 1))} aria-label="Previous frame">−1</button>
        <button className="btn" onClick={() => seekFrame(Math.min(nFrames - 1, frame + 1))} aria-label="Next frame">+1</button>
        <button className={`btn ${slow ? "btn-primary" : ""}`} onClick={() => setSlow(!slow)}>¼× slow-mo</button>
        <button className="btn btn-ghost" onClick={() => setShowOverlay(!showOverlay)}>{showOverlay ? "Hide" : "Show"} overlay</button>
      </div>
      {ev && (
        <div className="flex flex-wrap gap-2">
          {EVENT_LABELS.map(([k, label]) => (
            <button key={k} className="chip hover:opacity-80" onClick={() => seekFrame(ev[k] as number)}
              style={k === "contact" ? { borderColor: "var(--kick)" } : undefined}>
              {label} · {ev[k] as number}
            </button>
          ))}
        </div>
      )}
    </div>
  );
});
