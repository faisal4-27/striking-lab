import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { Progress, ShotOptionsForm, useShotOptions } from "../components/ui";

export default function Upload() {
  const nav = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [opts, setOpts] = useShotOptions();
  const [progress, setProgress] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const submit = async () => {
    if (!file) return;
    setErr(null);
    setProgress(0);
    try {
      const shot = await api.upload(file, file.name, opts, setProgress);
      nav(`/shots/${shot.id}`);
    } catch (e) {
      setErr((e as Error).message);
      setProgress(null);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <h1 className="display text-3xl font-bold">Upload a shot</h1>
      <label
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); }}
        className="card flex cursor-pointer flex-col items-center justify-center gap-2 p-10 text-center"
        style={drag ? { borderColor: "var(--accent)" } : undefined}>
        <input type="file" accept="video/*" className="hidden" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" aria-hidden>
          <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 20h16" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="font-semibold">{file ? file.name : "Choose a video or drop it here"}</span>
        <span className="text-sm text-muted">
          {file ? `${(file.size / 1e6).toFixed(1)} MB` : "MP4, MOV, WebM. Trim to the shot if you can; the first 12 s are analysed."}
        </span>
      </label>
      <div className="card p-4"><ShotOptionsForm value={opts} onChange={setOpts} /></div>
      {err && <div className="card p-3 text-sm" style={{ color: "var(--bad)" }}>{err}</div>}
      {progress != null ? <Progress value={progress} label="Uploading" /> : (
        <button className="btn btn-primary self-start px-8" disabled={!file} onClick={submit}>Analyze shot</button>
      )}
    </div>
  );
}
