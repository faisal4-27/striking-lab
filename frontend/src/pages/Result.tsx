import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, type Analysis, type Shot, type Subscore, type Target } from "../api";
import { ShotPlayer, type ShotPlayerHandle } from "../components/ShotPlayer";
import { Bar, Progress, ScoreRing, scoreColor } from "../components/ui";

const fmt = (v: number, unit: string) => {
  const n = Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2);
  return `${n}${unit}`;
};

function IdealText({ s }: { s: Subscore }) {
  const [lo, hi] = s.ideal;
  if (s.feature === "on_target") return <>target hit</>;
  if (hi >= 100 && s.unit.includes("km/h")) return <>≥ {lo}{s.unit}</>;
  return <>{fmt(lo, s.unit)} to {fmt(hi, s.unit)}</>;
}

function Fix({ s, rank, onShow }: { s: Subscore; rank: number; onShow: () => void }) {
  return (
    <div className="card flex flex-col gap-2 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="display flex h-7 w-7 items-center justify-center rounded-full text-lg font-bold"
            style={{ background: scoreColor(s.score), color: "var(--bg)" }}>{rank}</span>
          <span className="font-semibold">{s.label}</span>
        </div>
        <button className="chip shrink-0" onClick={onShow}>Show me</button>
      </div>
      <p className="font-medium">{s.issue}</p>
      <p className="text-sm text-muted">{s.tip}</p>
      <div className="text-xs text-muted">
        Measured <b className="text-ink">{s.feature === "on_target" ? (s.value ? "hit" : "miss") : fmt(s.value, s.unit)}</b> · ideal <IdealText s={s} />
      </div>
    </div>
  );
}

// Which phase to jump to when you tap "Show me" on a metric.
const FEATURE_PHASE: Record<string, keyof Analysis["events"]> = {
  approach_angle_deg: "plant", plant_fore_aft: "contact", plant_knee_flex_deg: "contact",
  backswing_knee_flex_deg: "backswing", backswing_thigh_deg: "backswing", trunk_lean_deg: "contact",
  ankle_lock_deg: "contact", foot_speed_mps: "contact", arm_balance_deg: "contact",
  follow_through_height: "follow_through", momentum_forward: "follow_through", head_over_ball: "contact",
  knee_over_ball: "contact", ball_speed_kmh: "contact", launch_angle_deg: "contact",
  direction_error_deg: "contact", on_target: "contact",
};

export default function Result() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const player = useRef<ShotPlayerHandle>(null);
  const [shot, setShot] = useState<Shot | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [marking, setMarking] = useState(false);
  const [frame, setFrame] = useState(0);
  const [busy, setBusy] = useState(false);
  const [version, setVersion] = useState(0);   // bump to refetch after changes

  useEffect(() => {
    let alive = true, timer: number | undefined;
    const poll = async () => {
      try {
        const s = await api.getShot(id);
        if (!alive) return;
        setShot(s);
        if (s.status === "done") {
          const a = await api.getAnalysis(id);
          if (alive) setAnalysis(a);
          setBusy(false);
        } else if (s.status !== "failed") {
          timer = window.setTimeout(poll, 1200);
        } else setBusy(false);
      } catch (e) {
        if (alive) setErr((e as Error).message);
      }
    };
    poll();
    return () => { alive = false; clearTimeout(timer); };
  }, [id, version]);

  const rerun = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    await fn();
    setVersion((v) => v + 1);
  };

  if (err) return <div className="card p-4" style={{ color: "var(--bad)" }}>{err}</div>;
  if (!shot) return <div className="text-muted">Loading…</div>;

  const processing = shot.status === "queued" || shot.status === "processing";
  const rep = analysis?.report;
  const flight = analysis?.flight ?? {};
  const show = (feature: string) => {
    const ev = analysis?.events;
    if (ev) player.current?.seekFrame(ev[FEATURE_PHASE[feature] ?? "contact"] as number);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="flex flex-col gap-5">
      {shot.status === "done" || analysis ? (
        <ShotPlayer ref={player} videoUrl={`${api.videoUrl(id)}?v=${version}`} analysis={analysis} target={shot.target}
          markingTarget={marking} onFrame={setFrame}
          onTargetDone={(t: Target) => { setMarking(false); rerun(() => api.setTarget(id, t)); }} />
      ) : null}

      {processing && (
        <div className="card flex flex-col gap-3 p-5">
          <h1 className="display text-2xl font-bold">Analysing your shot</h1>
          <Progress value={shot.progress} label={shot.stage} />
          <p className="text-sm text-muted">Tracking 23 body points and the ball frame by frame, then measuring the kick in 3D. Usually 20–60 seconds.</p>
        </div>
      )}
      {shot.status === "failed" && (
        <div className="card flex flex-col gap-3 p-5">
          <h1 className="display text-2xl font-bold" style={{ color: "var(--bad)" }}>Analysis failed</h1>
          <p className="text-sm text-muted">{shot.error}</p>
          <button className="btn self-start" onClick={() => rerun(() => api.reanalyze(id, true))}>Try again</button>
        </div>
      )}

      {rep && analysis && (
        <>
          <section className="card flex flex-col items-center gap-5 p-5 sm:flex-row">
            <ScoreRing value={rep.overall} label="overall" />
            <div className="flex w-full flex-col gap-3">
              {(["technique", "outcome"] as const).map((g) => (
                <div key={g}>
                  <div className="mb-1 flex justify-between text-sm">
                    <span className="capitalize">{g}</span>
                    <span className="tabular-nums text-muted">{rep.rules.groups[g] == null ? "not measured" : `${Math.round(rep.rules.groups[g]!)}%`}</span>
                  </div>
                  <Bar value={rep.rules.groups[g]} />
                </div>
              ))}
              <div className="flex flex-wrap gap-2 pt-1">
                <span className="chip">Confidence {Math.round((rep.confidence ?? 0) * 100)}%</span>
                <span className="chip">{analysis.events.kicking_foot} foot</span>
                {flight.ball_speed_kmh != null && <span className="chip">{String(flight.ball_speed_kmh)} km/h</span>}
                {flight.launch_angle_deg != null && <span className="chip">launch {String(flight.launch_angle_deg)}°</span>}
                {analysis.target.on_target != null && <span className="chip" style={{ color: analysis.target.on_target ? "var(--good)" : "var(--bad)" }}>{analysis.target.on_target ? "On target" : "Missed target"}</span>}
                <span className="chip" title="How the score was produced">{rep.source === "blend" ? `AI model ${Math.round((rep.model?.weight ?? 0) * 100)}% + rules` : "Coaching rules"}</span>
              </div>
            </div>
          </section>

          {rep.top_fixes.length > 0 ? (
            <section className="flex flex-col gap-3">
              <h2 className="display text-2xl font-bold">Fix these first</h2>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
                {rep.top_fixes.map((s, i) => <Fix key={s.feature} s={s} rank={i + 1} onShow={() => show(s.feature)} />)}
              </div>
            </section>
          ) : (
            <section className="card p-4"><b>Nothing major to fix.</b> <span className="text-muted">Every measured part of the kick is in the ideal range.</span></section>
          )}

          {rep.strengths.length > 0 && (
            <section className="flex flex-wrap items-center gap-2">
              <span className="text-sm text-muted">Doing well:</span>
              {rep.strengths.map((s) => <span key={s.label} className="chip" style={{ color: "var(--good)" }}>✓ {s.label}</span>)}
            </section>
          )}

          <section className="card p-4">
            <h2 className="display mb-3 text-2xl font-bold">Full breakdown</h2>
            <div className="flex flex-col divide-y divide-line">
              {rep.rules.subscores.map((s) => (
                <button key={s.feature} onClick={() => show(s.feature)} className="grid grid-cols-[1fr_auto] items-center gap-x-4 gap-y-1 py-2.5 text-left sm:grid-cols-[14rem_1fr_7rem]">
                  <span className="font-medium">{s.label}</span>
                  <span className="text-right text-sm tabular-nums sm:order-last">{s.feature === "on_target" ? (s.value ? "hit" : "miss") : fmt(s.value, s.unit)}</span>
                  <div className="col-span-2 sm:col-span-1"><Bar value={s.score} height={6} /></div>
                </button>
              ))}
            </div>
            {rep.rules.subscores.length < 17 && (
              <p className="mt-3 text-xs text-muted">
                Some metrics weren't measurable in this clip. Ball metrics need the ball tracked after contact, side-on
                metrics need a side view, and accuracy needs a target (mark one below).
              </p>
            )}
          </section>

          {rep.filming_tips.length > 0 && (
            <section className="card p-4">
              <h2 className="mb-2 font-semibold">For a more accurate analysis next time</h2>
              <ul className="list-disc space-y-1 pl-5 text-sm text-muted">{rep.filming_tips.map((t) => <li key={t}>{t}</li>)}</ul>
            </section>
          )}
        </>
      )}

      {shot.status === "done" && (
        <section className="card flex flex-col gap-3 p-4">
          <h2 className="font-semibold">Adjust the analysis</h2>
          <div className="flex flex-wrap gap-2">
            <button className={`btn ${marking ? "btn-primary" : ""}`} disabled={busy} onClick={() => { setMarking(!marking); player.current?.seekFrame(frame); }}>
              {marking ? "Cancel marking" : shot.target ? "Re-mark target" : "Mark target / goal"}
            </button>
            {shot.target && <button className="btn" disabled={busy} onClick={() => rerun(() => api.setTarget(id, null))}>Clear target</button>}
            <button className="btn" disabled={busy}
              onClick={() => rerun(() => api.setCorrections(id, { ...shot.manual, contact: frame, foot_hint: shot.foot_hint }))}>
              Set contact = frame {frame}
            </button>
            <button className="btn" disabled={busy}
              onClick={() => rerun(() => api.setCorrections(id, { ...shot.manual, foot_hint: analysis?.events.kicking_foot === "left" ? "right" : "left" }))}>
              Kicked with {analysis?.events.kicking_foot === "left" ? "right" : "left"} foot
            </button>
            {shot.manual && <button className="btn" disabled={busy} onClick={() => rerun(() => api.setCorrections(id, { foot_hint: shot.foot_hint }))}>Reset corrections</button>}
          </div>
          <div className="flex flex-wrap gap-2 border-t border-line pt-3">
            <Link to={`/label?shot=${id}`} className="btn btn-primary">{shot.label ? `Your rating: ${shot.label.overall}` : "Rate this shot (trains the AI)"}</Link>
            <button className="btn btn-ghost" disabled={busy} onClick={() => rerun(() => api.reanalyze(id, true))}>Re-run from scratch</button>
            <button className="btn btn-ghost" style={{ color: "var(--bad)" }}
              onClick={async () => { if (confirm("Delete this shot and its video?")) { await api.deleteShot(id); nav("/"); } }}>Delete</button>
          </div>
        </section>
      )}
    </div>
  );
}
