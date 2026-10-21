import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type Shot } from "../api";
import { scoreColor } from "../components/ui";

export default function History() {
  const [shots, setShots] = useState<Shot[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () => api.listShots().then((s) => alive && setShots(s)).catch((e) => alive && setErr(e.message));
    load();
    const iv = setInterval(load, 4000);
    return () => { alive = false; clearInterval(iv); };
  }, []);

  if (err) return <div className="card p-4" style={{ color: "var(--bad)" }}>Can't reach the server: {err}</div>;
  if (!shots) return <div className="text-muted">Loading…</div>;

  const done = shots.filter((s) => s.status === "done" && s.overall != null);
  const avg = done.length ? done.reduce((a, s) => a + (s.overall ?? 0), 0) / done.length : null;
  const best = done.length ? Math.max(...done.map((s) => s.overall ?? 0)) : null;

  if (!shots.length) {
    return (
      <div className="card flex flex-col items-start gap-3 p-8">
        <h1 className="display text-4xl font-bold">Analyse your first shot</h1>
        <p className="max-w-prose text-muted">
          Record or upload an instep drive. Striking Lab tracks your body and the ball, scores your technique and the
          shot, and tells you the one or two things to fix first.
        </p>
        <div className="flex gap-2">
          <Link to="/record" className="btn btn-primary">Record</Link>
          <Link to="/upload" className="btn">Upload</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-3 gap-3">
        {[["Shots", String(shots.length)], ["Average", avg == null ? "–" : `${Math.round(avg)}%`], ["Best", best == null ? "–" : `${Math.round(best)}%`]].map(([k, v]) => (
          <div key={k} className="card p-4">
            <div className="text-xs uppercase tracking-wider text-muted">{k}</div>
            <div className="display text-3xl font-bold tabular-nums">{v}</div>
          </div>
        ))}
      </div>
      <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {shots.map((s) => (
          <li key={s.id}>
            <Link to={`/shots/${s.id}`} className="card flex gap-3 overflow-hidden p-2 hover:opacity-90">
              <div className="h-20 w-32 shrink-0 overflow-hidden rounded-xl bg-surface-2">
                {s.status === "done" && <img src={api.thumbUrl(s.id)} alt="" className="h-full w-full object-cover" loading="lazy" />}
              </div>
              <div className="flex min-w-0 flex-1 flex-col justify-center">
                <div className="truncate font-semibold">{s.player || "Shot"} <span className="font-normal text-muted">· {new Date(s.created_at).toLocaleString()}</span></div>
                <div className="mt-1 flex flex-wrap items-center gap-2 text-sm">
                  {s.status === "done" ? (
                    <span className="display text-2xl font-bold tabular-nums" style={{ color: scoreColor(s.overall) }}>{s.overall == null ? "–" : `${Math.round(s.overall)}%`}</span>
                  ) : s.status === "failed" ? (
                    <span className="chip" style={{ color: "var(--bad)" }}>Failed</span>
                  ) : (
                    <span className="chip">{s.stage ?? "Queued"} · {Math.round(s.progress * 100)}%</span>
                  )}
                  {s.label && <span className="chip">Rated {s.label.overall}</span>}
                </div>
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
