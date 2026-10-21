import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, ASPECTS, type Analysis, type ModelStatus, type Shot } from "../api";
import { ShotPlayer } from "../components/ShotPlayer";
import { scoreColor } from "../components/ui";

function ModelCard({ status }: { status: ModelStatus | null }) {
  if (!status) return null;
  const goal = 150;
  return (
    <section className="card flex flex-col gap-3 p-4">
      <div className="flex items-baseline justify-between">
        <h2 className="display text-2xl font-bold">Your AI coach</h2>
        <span className="text-sm text-muted">{status.n_labels} rated shots</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2">
        <div className="h-full rounded-full bg-accent" style={{ width: `${Math.min(100, (status.n_labels / goal) * 100)}%` }} />
      </div>
      {status.trained ? (
        <p className="text-sm text-muted">
          Trained on {status.n_samples} shots. Average error {status.cv_mae?.toFixed(1)} points vs {status.rules_mae?.toFixed(1)} for
          the rule engine. It now contributes <b className="text-ink">{Math.round(status.weight * 100)}%</b> of each score.
        </p>
      ) : (
        <p className="text-sm text-muted">
          Not trained yet. Rate shots below; from about 30 the model can start learning your idea of a good shot, and
          around {goal} it becomes reliable. Then run <code className="rounded bg-surface-2 px-1">python -m training.train_scorer</code> in <code>backend/</code>.
        </p>
      )}
    </section>
  );
}

export default function Label() {
  const [params, setParams] = useSearchParams();
  const [shots, setShots] = useState<Shot[]>([]);
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [overall, setOverall] = useState(60);
  const [aspects, setAspects] = useState<Record<string, number>>({});
  const [onTarget, setOnTarget] = useState<boolean | null>(null);
  const [notes, setNotes] = useState("");
  const [reveal, setReveal] = useState(false);
  const [saving, setSaving] = useState(false);

  const load = () => {
    api.listShots().then(setShots);
    api.modelStatus().then(setStatus);
  };
  useEffect(load, []);

  const done = shots.filter((s) => s.status === "done");
  const unlabeled = done.filter((s) => !s.label);
  const currentId = params.get("shot") ?? unlabeled[0]?.id ?? null;
  const current = useMemo(() => shots.find((s) => s.id === currentId) ?? null, [shots, currentId]);

  useEffect(() => {
    if (!currentId) return;
    setAnalysis(null);
    api.getAnalysis(currentId).then(setAnalysis).catch(() => setAnalysis(null));
  }, [currentId]);

  useEffect(() => {
    const l = current?.label;
    setOverall(l?.overall ?? 60);
    setAspects(l?.aspects ?? {});
    setOnTarget(l?.on_target ?? null);
    setNotes(l?.notes ?? "");
    setReveal(!!l);
  }, [current?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const save = async (next: boolean) => {
    if (!current) return;
    setSaving(true);
    await api.setLabel(current.id, { overall, aspects, on_target: onTarget, notes: notes || null });
    setSaving(false);
    const remaining = unlabeled.filter((s) => s.id !== current.id);
    if (next && remaining[0]) setParams({ shot: remaining[0].id });
    else setReveal(true);
    load();
  };

  return (
    <div className="flex flex-col gap-5">
      <ModelCard status={status} />

      {!current ? (
        <div className="card p-6 text-muted">
          {done.length ? "Every analysed shot is rated. Record more shots to keep training." : "No analysed shots yet."}{" "}
          <Link to="/record" className="text-accent underline">Record a shot</Link>
        </div>
      ) : (
        <>
          <div className="flex items-baseline justify-between">
            <h1 className="display text-2xl font-bold">Rate this shot</h1>
            <span className="text-sm text-muted">{unlabeled.length} unrated</span>
          </div>
          <ShotPlayer videoUrl={api.videoUrl(current.id)} analysis={analysis} target={current.target} />

          <section className="card flex flex-col gap-5 p-4">
            <div>
              <div className="mb-2 flex items-baseline justify-between">
                <span className="font-semibold">Overall quality</span>
                <span className="display text-4xl font-bold tabular-nums" style={{ color: scoreColor(overall) }}>{overall}</span>
              </div>
              <input type="range" min={0} max={100} value={overall} onChange={(e) => setOverall(Number(e.target.value))}
                className="w-full accent-[var(--accent)]" aria-label="Overall rating" />
              <div className="flex justify-between text-xs text-muted"><span>Shank</span><span>Decent</span><span>Pro-level</span></div>
            </div>

            <div className="flex flex-col gap-2">
              <span className="font-semibold">Aspects <span className="font-normal text-muted">(optional, 1–5)</span></span>
              {ASPECTS.map((a) => (
                <div key={a.key} className="flex items-center justify-between gap-3">
                  <span className="text-sm">{a.label}</span>
                  <div className="flex gap-1">
                    {[1, 2, 3, 4, 5].map((n) => (
                      <button key={n} aria-label={`${a.label} ${n}`}
                        onClick={() => setAspects((x) => (x[a.key] === n ? Object.fromEntries(Object.entries(x).filter(([k]) => k !== a.key)) : { ...x, [a.key]: n }))}
                        className="h-8 w-8 rounded-full text-sm font-semibold"
                        style={{ background: (aspects[a.key] ?? 0) >= n ? "var(--accent)" : "var(--surface-2)", color: (aspects[a.key] ?? 0) >= n ? "var(--accent-ink)" : "var(--muted)" }}>
                        {n}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-semibold">On target?</span>
              {([[true, "Yes"], [false, "No"], [null, "Unknown"]] as const).map(([v, l]) => (
                <button key={l} className={`btn ${onTarget === v ? "btn-primary" : ""}`} onClick={() => setOnTarget(v)}>{l}</button>
              ))}
            </div>

            <textarea className="input" rows={2} placeholder="Notes (optional)" value={notes} onChange={(e) => setNotes(e.target.value)} />

            <div className="flex flex-wrap items-center gap-2">
              <button className="btn btn-primary" disabled={saving} onClick={() => save(true)}>Save & next</button>
              <button className="btn" disabled={saving} onClick={() => save(false)}>Save</button>
              <Link to={`/shots/${current.id}`} className="btn btn-ghost">Open analysis</Link>
            </div>
            <div className="text-sm text-muted">
              {reveal ? (
                <>App's score: <b style={{ color: scoreColor(current.overall) }}>{current.overall == null ? "–" : Math.round(current.overall)}</b></>
              ) : (
                <button className="underline" onClick={() => setReveal(true)}>Reveal the app's score</button>
              )}
              <span className="ml-1">(hidden until you rate, so it doesn't bias your rating)</span>
            </div>
          </section>

          {done.length > 1 && (
            <section className="flex flex-wrap gap-2">
              {done.map((s) => (
                <button key={s.id} onClick={() => setParams({ shot: s.id })}
                  className="chip" style={s.id === current.id ? { borderColor: "var(--accent)" } : undefined}>
                  {s.player || s.id.slice(0, 6)} {s.label ? `· ${s.label.overall}` : "· unrated"}
                </button>
              ))}
            </section>
          )}
        </>
      )}
    </div>
  );
}
