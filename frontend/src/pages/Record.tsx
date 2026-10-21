import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { Progress, ShotOptionsForm, useShotOptions } from "../components/ui";

const MAX_SECONDS = 10;

function pickMime(): string {
  const cands = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  return cands.find((m) => typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(m)) ?? "";
}

type Phase = "idle" | "countdown" | "recording" | "review" | "uploading";

export default function Record() {
  const nav = useNavigate();
  const liveRef = useRef<HTMLVideoElement>(null);
  const recRef = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [count, setCount] = useState(3);
  const [elapsed, setElapsed] = useState(0);
  const [clip, setClip] = useState<{ blob: Blob; url: string; ext: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [facing, setFacing] = useState<"environment" | "user">("environment");
  const [info, setInfo] = useState<string>("");
  const [opts, setOpts] = useShotOptions();
  const [upload, setUpload] = useState(0);

  const secure = typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;

  useEffect(() => {
    if (!secure || phase === "review" || phase === "uploading") return;
    let cancelled = false;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: facing, frameRate: { ideal: 60 }, width: { ideal: 1920 }, height: { ideal: 1080 } },
        });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = stream;
        if (liveRef.current) liveRef.current.srcObject = stream;
        const s = stream.getVideoTracks()[0].getSettings();
        setInfo(`${s.width}×${s.height} @ ${Math.round(s.frameRate ?? 0)} fps`);
        setErr(null);
      } catch (e) {
        setErr(`Camera unavailable: ${(e as Error).message}`);
      }
    })();
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, [facing, secure, phase === "review" || phase === "uploading"]); // eslint-disable-line react-hooks/exhaustive-deps

  const start = () => {
    setPhase("countdown");
    setCount(3);
    let n = 3;
    const iv = setInterval(() => {
      n -= 1;
      setCount(n);
      if (n === 0) { clearInterval(iv); beginRecording(); }
    }, 1000);
  };

  const beginRecording = () => {
    const stream = streamRef.current;
    if (!stream) return;
    const mime = pickMime();
    const rec = new MediaRecorder(stream, { ...(mime ? { mimeType: mime } : {}), videoBitsPerSecond: 12_000_000 });
    chunks.current = [];
    rec.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
    rec.onstop = () => {
      const type = rec.mimeType || mime || "video/webm";
      const blob = new Blob(chunks.current, { type });
      setClip({ blob, url: URL.createObjectURL(blob), ext: type.includes("mp4") ? ".mp4" : ".webm" });
      setPhase("review");
    };
    rec.start(250);
    recRef.current = rec;
    setPhase("recording");
    const t0 = Date.now();
    const iv = setInterval(() => {
      const s = (Date.now() - t0) / 1000;
      setElapsed(s);
      if (s >= MAX_SECONDS || rec.state !== "recording") {
        clearInterval(iv);
        if (rec.state === "recording") rec.stop();
      }
    }, 100);
  };

  const stop = () => recRef.current?.state === "recording" && recRef.current.stop();

  const retake = () => {
    if (clip) URL.revokeObjectURL(clip.url);
    setClip(null);
    setElapsed(0);
    setPhase("idle");
  };

  const analyze = async () => {
    if (!clip) return;
    setPhase("uploading");
    try {
      const shot = await api.upload(clip.blob, `recording${clip.ext}`, {
        player: opts.player, foot_hint: opts.foot_hint, player_height_m: opts.player_height_m,
      }, setUpload);
      nav(`/shots/${shot.id}`);
    } catch (e) {
      setErr((e as Error).message);
      setPhase("review");
    }
  };

  if (!secure) {
    return (
      <div className="card p-6">
        <h1 className="display text-3xl font-bold">Camera needs HTTPS</h1>
        <p className="mt-2 text-muted">
          Browsers only allow camera access on secure pages. Open this app with <code>https://</code> (the dev server
          runs HTTPS on port 5173), accept the certificate warning once, and come back. Or use{" "}
          <a className="text-accent underline" href="/upload">Upload</a> instead.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="relative overflow-hidden rounded-2xl bg-black" style={{ aspectRatio: "16/9", maxHeight: "70vh" }}>
        {phase === "review" || phase === "uploading" ? (
          <video src={clip?.url} className="h-full w-full object-contain" controls playsInline autoPlay muted loop />
        ) : (
          <video ref={liveRef} className="h-full w-full object-contain" playsInline autoPlay muted />
        )}
        {(phase === "idle" || phase === "countdown") && (
          // framing guide: player should fit inside, side-on, ball and target in view
          <div className="pointer-events-none absolute inset-[8%] rounded-xl border-2 border-dashed border-white/40">
            <span className="absolute left-2 top-2 chip bg-black/50 text-white">Side-on · whole body · ball in frame</span>
          </div>
        )}
        {phase === "countdown" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/30">
            <span className="display text-9xl font-bold text-white">{count}</span>
          </div>
        )}
        {phase === "recording" && (
          <div className="absolute left-3 top-3 chip" style={{ background: "var(--bad)", color: "#fff" }}>
            ● REC {elapsed.toFixed(1)}s / {MAX_SECONDS}s
          </div>
        )}
        {info && phase !== "review" && <div className="absolute bottom-3 right-3 chip bg-black/50 text-white">{info}</div>}
      </div>

      {err && <div className="card border-bad p-3 text-sm" style={{ color: "var(--bad)" }}>{err}</div>}

      {phase === "idle" && (
        <div className="flex flex-wrap gap-2">
          <button className="btn btn-primary px-8 text-lg" onClick={start} disabled={!!err}>Record shot</button>
          <button className="btn" onClick={() => setFacing(facing === "environment" ? "user" : "environment")}>Flip camera</button>
        </div>
      )}
      {phase === "countdown" && <p className="text-muted">Get ready…</p>}
      {phase === "recording" && <button className="btn btn-primary px-8 text-lg" onClick={stop}>Stop</button>}
      {(phase === "review" || phase === "uploading") && (
        <div className="card flex flex-col gap-4 p-4">
          <ShotOptionsForm value={opts} onChange={setOpts} />
          {phase === "uploading" ? <Progress value={upload} label="Uploading" /> : (
            <div className="flex gap-2">
              <button className="btn btn-primary px-8" onClick={analyze}>Analyze shot</button>
              <button className="btn" onClick={retake}>Retake</button>
            </div>
          )}
        </div>
      )}

      <details className="card p-4 text-sm">
        <summary className="cursor-pointer font-semibold">Filming tips for the best analysis</summary>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-muted">
          <li>Film from the <b>side</b>, 5–10 m away, level with the ball. Whole body and feet visible.</li>
          <li>Start recording before the run-up and keep filming until the ball reaches the target.</li>
          <li>Hold the phone steady with both hands, in landscape. Good light helps the ball tracker.</li>
          <li>Include the goal or target in frame if you want accuracy scored. You can mark it afterwards.</li>
        </ul>
      </details>
    </div>
  );
}
