import { useState } from "react";
import { Link, NavLink, Outlet } from "react-router-dom";

export function scoreColor(s: number | null | undefined) {
  if (s == null) return "var(--muted)";
  return s >= 75 ? "var(--good)" : s >= 50 ? "var(--warn)" : "var(--bad)";
}

export function ScoreRing({ value, size = 148, label }: { value: number | null; size?: number; label?: string }) {
  const r = size / 2 - 10, c = 2 * Math.PI * r;
  const pct = value == null ? 0 : Math.max(0, Math.min(100, value)) / 100;
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--surface-2)" strokeWidth={10} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={scoreColor(value)} strokeWidth={10}
          strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - pct)}
          style={{ transition: "stroke-dashoffset .8s ease" }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="display text-5xl font-bold tabular-nums leading-none">{value == null ? "–" : Math.round(value)}<span className="text-2xl text-muted">%</span></span>
        {label && <span className="mt-1 text-xs uppercase tracking-wider text-muted">{label}</span>}
      </div>
    </div>
  );
}

export function Bar({ value, height = 8 }: { value: number | null | undefined; height?: number }) {
  return (
    <div className="w-full overflow-hidden rounded-full bg-surface-2" style={{ height }}>
      <div className="h-full rounded-full" style={{ width: `${Math.max(0, Math.min(100, value ?? 0))}%`, background: scoreColor(value), transition: "width .6s ease" }} />
    </div>
  );
}

export function Progress({ value, label }: { value: number; label?: string | null }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex justify-between text-sm"><span>{label ?? "Working"}</span><span className="tabular-nums text-muted">{Math.round(value * 100)}%</span></div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2">
        <div className="h-full rounded-full bg-accent" style={{ width: `${value * 100}%`, transition: "width .4s ease" }} />
      </div>
    </div>
  );
}

export interface ShotOptions { player: string; foot_hint: string; player_height_m: number }

const OPTS_KEY = "striking-lab:shot-options";
export function useShotOptions(): [ShotOptions, (o: ShotOptions) => void] {
  const [opts, setOpts] = useState<ShotOptions>(() => {
    try {
      const raw = localStorage.getItem(OPTS_KEY);
      if (raw) return JSON.parse(raw);
    } catch { /* storage unavailable */ }
    return { player: "", foot_hint: "", player_height_m: 1.75 };
  });
  const save = (o: ShotOptions) => {
    setOpts(o);
    try { localStorage.setItem(OPTS_KEY, JSON.stringify(o)); } catch { /* ignore */ }
  };
  return [opts, save];
}

export function ShotOptionsForm({ value, onChange }: { value: ShotOptions; onChange: (o: ShotOptions) => void }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-muted">Player</span>
        <input className="input" placeholder="Name (optional)" value={value.player}
          onChange={(e) => onChange({ ...value, player: e.target.value })} />
      </label>
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-muted">Kicking foot</span>
        <select className="input" value={value.foot_hint} onChange={(e) => onChange({ ...value, foot_hint: e.target.value })}>
          <option value="">Auto-detect</option>
          <option value="right">Right</option>
          <option value="left">Left</option>
        </select>
      </label>
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-muted">Player height (m)</span>
        <input className="input" type="number" step={0.01} min={1} max={2.3} value={value.player_height_m}
          onChange={(e) => onChange({ ...value, player_height_m: Number(e.target.value) })} />
      </label>
    </div>
  );
}

export function Layout() {
  const link = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-2 rounded-full text-sm font-semibold ${isActive ? "bg-surface-2 text-ink" : "text-muted hover:text-ink"}`;
  return (
    <div className="mx-auto flex min-h-full max-w-5xl flex-col px-4 pb-24 sm:pb-10">
      <header className="flex items-center justify-between py-4">
        <Link to="/" className="flex items-center gap-2">
          <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden>
            <circle cx="16" cy="16" r="13" fill="none" stroke="var(--accent)" strokeWidth="2.5" />
            <path d="M16 9l5 3.6-1.9 5.9h-6.2L11 12.6z" fill="var(--accent)" />
          </svg>
          <span className="display text-2xl font-bold tracking-tight">STRIKING LAB</span>
        </Link>
        <nav className="hidden gap-1 sm:flex">
          <NavLink to="/record" className={link}>Record</NavLink>
          <NavLink to="/upload" className={link}>Upload</NavLink>
          <NavLink to="/" end className={link}>History</NavLink>
          <NavLink to="/label" className={link}>Label & train</NavLink>
        </nav>
      </header>
      <main className="flex-1"><Outlet /></main>
      {/* phone tab bar */}
      <nav className="fixed inset-x-0 bottom-0 z-10 grid grid-cols-4 border-t border-line bg-surface sm:hidden"
        style={{ paddingBottom: "env(safe-area-inset-bottom)" }}>
        {[["/record", "Record"], ["/upload", "Upload"], ["/", "History"], ["/label", "Label"]].map(([to, label]) => (
          <NavLink key={to} to={to} end={to === "/"}
            className={({ isActive }) => `py-3 text-center text-sm font-semibold ${isActive ? "text-accent" : "text-muted"}`}>
            {label}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
