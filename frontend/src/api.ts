export type Status = "queued" | "processing" | "done" | "failed";

export interface LabelData {
  overall: number;
  aspects: Record<string, number>;
  on_target: boolean | null;
  notes: string | null;
  updated_at?: string;
}

export interface Shot {
  id: string;
  created_at: string;
  player: string | null;
  original_name: string | null;
  status: Status;
  progress: number;
  stage: string | null;
  error: string | null;
  overall: number | null;
  technique: number | null;
  outcome: number | null;
  confidence: number | null;
  kicking_foot: string | null;
  foot_hint: string | null;
  player_height_m: number;
  target: Target | null;
  manual: { contact?: number; plant?: number } | null;
  label: LabelData | null;
}

export interface Target {
  frame: number;
  points: [number, number][];
}

export interface Subscore {
  feature: string;
  label: string;
  group: "technique" | "outcome";
  aspect: string;
  value: number;
  unit: string;
  ideal: [number, number];
  score: number;
  weight: number;
  status: "good" | "low" | "high";
  issue: string | null;
  tip: string | null;
  impact?: number;
}

export interface Events {
  kicking_foot: "left" | "right";
  contact: number;
  plant: number;
  backswing: number;
  follow_through: number;
  approach_start: number;
  contact_source: string;
}

export interface Analysis {
  events: Events;
  flight: Record<string, number | string | number[] | null>;
  target: { on_target?: boolean | null; direction_error_deg?: number };
  features: Record<string, number | string | null>;
  quality: Record<string, number | string | null>;
  report: {
    overall: number | null;
    source: "rules" | "blend";
    rules: { overall: number | null; groups: { technique?: number; outcome?: number }; aspects: Record<string, number>; subscores: Subscore[] };
    model: { score: number; weight: number; n_samples: number; cv_mae: number } | null;
    issues: Subscore[];
    top_fixes: Subscore[];
    strengths: { label: string; score: number }[];
    confidence: number;
    filming_tips: string[];
  };
  video: { fps: number; width: number; height: number; frames: number };
  overlay: {
    fps: number;
    width: number;
    height: number;
    edges: [number, number][];
    kp: ([number | null, number | null])[][];
    ball: [number | null, number | null, number | null][];
  };
}

export interface ModelStatus {
  trained: boolean;
  weight: number;
  n_labels: number;
  n_samples?: number;
  cv_mae?: number;
  rules_mae?: number;
  trained_at?: string;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (body.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  listShots: () => req<Shot[]>("/api/shots"),
  getShot: (id: string) => req<Shot>(`/api/shots/${id}`),
  getAnalysis: (id: string) => req<Analysis>(`/api/shots/${id}/analysis`),
  videoUrl: (id: string) => `/api/shots/${id}/video`,
  thumbUrl: (id: string) => `/api/shots/${id}/thumb`,
  upload: (file: Blob, filename: string, opts: { player?: string; foot_hint?: string; player_height_m?: number },
           onProgress?: (p: number) => void) =>
    new Promise<Shot>((resolve, reject) => {
      // XHR (not fetch) so we can show upload progress on slow phone connections
      const fd = new FormData();
      fd.append("file", file, filename);
      if (opts.player) fd.append("player", opts.player);
      if (opts.foot_hint) fd.append("foot_hint", opts.foot_hint);
      if (opts.player_height_m) fd.append("player_height_m", String(opts.player_height_m));
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/shots");
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
        else {
          let msg = `Upload failed (${xhr.status})`;
          try { msg = JSON.parse(xhr.responseText).detail ?? msg; } catch { /* ignore */ }
          reject(new Error(msg));
        }
      };
      xhr.onerror = () => reject(new Error("Network error during upload"));
      xhr.send(fd);
    }),
  setTarget: (id: string, target: Target | null) => req(`/api/shots/${id}/target`, json("PUT", target)),
  setCorrections: (id: string, c: { contact?: number | null; plant?: number | null; foot_hint?: string | null; player_height_m?: number }) =>
    req(`/api/shots/${id}/corrections`, json("PUT", c)),
  reanalyze: (id: string, fresh = false) => req(`/api/shots/${id}/reanalyze?fresh=${fresh}`, { method: "POST" }),
  setLabel: (id: string, label: Omit<LabelData, "updated_at">) => req<LabelData>(`/api/shots/${id}/label`, json("PUT", label)),
  deleteShot: (id: string) => req(`/api/shots/${id}`, { method: "DELETE" }),
  modelStatus: () => req<ModelStatus>("/api/model"),
};

export const ASPECTS: { key: string; label: string }[] = [
  { key: "approach", label: "Approach" },
  { key: "plant", label: "Plant foot" },
  { key: "strike", label: "Strike & ankle" },
  { key: "body_shape", label: "Body shape" },
  { key: "follow_through", label: "Follow-through" },
  { key: "power", label: "Power" },
  { key: "accuracy", label: "Accuracy" },
];
