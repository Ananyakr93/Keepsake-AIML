import PrivateImage from "../components/PrivateImage";
import { useEffect, useState } from "react";
import { fmtDuration, fmtTime } from "../api";

const INTENTS = ["research", "learning", "purchase", "planning", "entertainment", "casual", "task"] as const;
type Intent = (typeof INTENTS)[number];
type Label = { intent: Intent; memory_worthy: boolean; memory_note?: string | null };
type Session = {
  id: number;
  evidence_id: string;
  started_at: string;
  ended_at: string;
  needs_review: boolean;
  label: Label | null;
  features: { dwell_sec: number; visit_count: number; domain_count: number };
  pages: { domain: string; title: string | null; dwell_sec: number }[];
  screenshots: { id: string; captured_at: string; embedded: boolean }[];
};
type Analysis = {
  prediction: {
    intent: Intent;
    intent_probabilities: Record<Intent, number>;
    selected: string;
    admission: Record<string, { probability: number; threshold: number; remember: boolean }>;
  };
  interests: { topic: string; baseline: number; learned: number | null }[];
};

type Queue = {
  collection_enabled: boolean;
  total: number;
  reviewed: number;
  sessions: Session[];
};

const PAGE_SIZE = 20;
const buttonStyle = "rounded-lg surface px-3 py-2 text-xs font-medium hover:bg-emphasis disabled:opacity-40";
const inputStyle = "rounded-lg border border-rule bg-sheet px-3 py-2 text-sm";

function dataset<T>(action: string, input?: Parameters<NonNullable<Window["desktop"]>["dataset"]>[1]) {
  if (!window.desktop) return Promise.reject(new Error("Open the Electron app to review the dataset."));
  return window.desktop.dataset<T>(action, input);
}

export default function LabelDataset() {
  const [queue, setQueue] = useState<Queue>();
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<number>();
  const [intent, setIntent] = useState<Intent | "">("");
  const [worthy, setWorthy] = useState("");
  const [memoryNote, setMemoryNote] = useState("");
  const [includeText, setIncludeText] = useState(false);
  const [includeVisual, setIncludeVisual] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [preview, setPreview] = useState<string>();
  const [analysis, setAnalysis] = useState<Analysis>();
  const session = queue?.sessions.find((entry) => entry.id === selected);

  function select(entry?: Session) {
    setSelected(entry?.id);
    setIntent(entry?.label?.intent ?? "");
    setWorthy(entry?.label ? String(entry.label.memory_worthy) : "");
    setMemoryNote(entry?.label?.memory_note ?? "");
    setPreview(undefined);
    setAnalysis(undefined);
  }

  async function load(at: number, keep?: number) {
    let result = await dataset<Queue>("list", { offset: at });
    if (at > 0 && !result.sessions.length) {
      at = Math.max(0, Math.floor((result.total - 1) / PAGE_SIZE) * PAGE_SIZE);
      result = await dataset<Queue>("list", { offset: at });
    }
    setQueue(result);
    setOffset(at);
    select(result.sessions.find((entry) => entry.id === keep) ?? result.sessions[0]);
  }

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await action();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void run(() => load(0));
  }, []);

  return (
    <div className="mx-auto max-w-6xl space-y-5 p-7">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="page-title">Label dataset</h1>
          {queue && <p className="mt-1 text-sm text-slate-400">{queue.reviewed} of {queue.total} sessions labeled</p>}
        </div>
        <button className={buttonStyle} disabled={busy} onClick={() => void run(() => load(offset, selected))}>Refresh</button>
      </header>

      <section className="surface rounded-xl p-4">
        <label className="flex items-center gap-3 text-sm font-medium">
          <input type="checkbox" className="accent-lime-300" checked={queue?.collection_enabled ?? false}
            disabled={busy || !queue}
            onChange={(event) => {
              const enabled = event.target.checked;
              void run(async () => {
                await dataset("collection", { payload: { enabled } });
                await load(0);
              });
            }} />
          I consent to include new activity in my local AIML dataset
        </label>
        <p className="mt-2 text-xs leading-relaxed text-slate-400">
          Enable browsing tracking in Settings and the extension. Sessions appear after 30 minutes of inactivity and a short upload grace period.
          Screenshots are optional. Close sensitive content before enabling capture.
        </p>
      </section>

      {error && <p role="alert" className="text-sm text-red-300">{error}</p>}
      {notice && <p role="status" className="text-sm text-lime-300">{notice}</p>}

      {queue && queue.total === 0 && (
        <p className="rounded-xl surface p-5 text-sm text-slate-400">No completed consented sessions yet.</p>
      )}

      {queue && queue.total > 0 && (
        <div className="grid items-start gap-4 lg:grid-cols-[16rem_1fr]">
          <section aria-label="Sessions" className="surface rounded-xl p-3">
            <ul className="space-y-1">
              {queue.sessions.map((entry) => (
                <li key={entry.id}>
                  <button className={`w-full rounded-lg px-3 py-2 text-left text-sm ${entry.id === selected ? "bg-emphasis text-ink" : "text-slate-400 hover:bg-wash"}`}
                    aria-pressed={entry.id === selected} disabled={busy} onClick={() => select(entry)}>
                    <span className="block truncate">{entry.pages[0]?.title || entry.pages[0]?.domain || "Browsing session"}</span>
                    <span className="text-xs text-slate-400">{fmtTime(entry.started_at)} · {entry.label ? "Labeled" : entry.needs_review ? "Review again" : "Unlabeled"}</span>
                  </button>
                </li>
              ))}
            </ul>
            <div className="mt-3 flex justify-between gap-2">
              <button className={buttonStyle} disabled={busy || offset === 0} onClick={() => void run(() => load(offset - PAGE_SIZE))}>Previous</button>
              <button className={buttonStyle} disabled={busy || offset + PAGE_SIZE >= queue.total} onClick={() => void run(() => load(offset + PAGE_SIZE))}>Next</button>
            </div>
          </section>

          {session && (
            <section className="surface space-y-4 rounded-xl p-5">
              <div>
                <h2 className="text-base font-semibold">{fmtTime(session.started_at)} – {fmtTime(session.ended_at)}</h2>
                <p className="mt-1 text-xs text-slate-400">
                  {fmtDuration(session.features.dwell_sec)} active · {session.features.visit_count} visits · {session.features.domain_count} sites
                </p>
                {session.needs_review && <p className="mt-2 text-sm text-amber-300">Evidence changed. Review this session again before exporting it.</p>}
              </div>
              <ul className="max-h-72 divide-y divide-rule overflow-auto">
                {session.pages.map((page, index) => (
                  <li key={index} className="flex justify-between gap-3 py-2 text-sm">
                    <div className="min-w-0"><p className="break-words">{page.title || page.domain}</p><p className="text-xs text-slate-400">{page.domain}</p></div>
                    <span className="shrink-0 text-xs text-slate-400">{fmtDuration(page.dwell_sec)}</span>
                  </li>
                ))}
              </ul>
              {session.screenshots.length > 0 && (
                <div>
                  <h3 className="mb-2 text-sm font-medium">Matched screenshots</h3>
                  <div className="flex flex-wrap gap-2">
                    {session.screenshots.map((shot) => (
                      <button key={shot.id} className={buttonStyle} aria-pressed={preview === shot.id}
                        onClick={() => setPreview(preview === shot.id ? undefined : shot.id)}>
                        {fmtTime(shot.captured_at)}{shot.embedded ? "" : " · embedding pending"}
                      </button>
                    ))}
                  </div>
                  {preview && <PrivateImage className="mt-3 max-h-96 w-full rounded-lg object-contain" id={preview} alt="Selected session screenshot" />}
                </div>
              )}

              <div className="space-y-2 border-t border-rule pt-4">
                <button className={buttonStyle} disabled={busy} onClick={() => void run(async () => {
                  setAnalysis(await dataset<Analysis>("analysis", { anchorId: session.id }));
                })}>Analyze with trained models</button>
                {analysis && (
                  <div className="space-y-2 text-sm text-slate-300">
                    <p>Predicted intent <strong>{analysis.prediction.intent}</strong> · {
                      Math.round(100 * analysis.prediction.intent_probabilities[analysis.prediction.intent])}% confidence</p>
                    <ul className="space-y-1">
                      {Object.entries(analysis.prediction.admission).map(([name, decision]) => (
                        <li key={name}>{name.replaceAll("_", " ")}{name === analysis.prediction.selected ? " (selected)" : ""}
                          {" · "}{Math.round(100 * decision.probability)}% · {decision.remember ? "Remember" : "Skip"}</li>
                      ))}
                    </ul>
                    {analysis.interests.length > 0 && <p>Current interests · {analysis.interests.slice(0, 5)
                      .map((interest) => `${interest.topic} ${Math.round(100 * (interest.learned ?? interest.baseline))}%`).join(", ")}</p>}
                  </div>
                )}
              </div>

              <form className="flex flex-wrap items-end gap-3 border-t border-rule pt-4"
                onSubmit={(event) => {
                  event.preventDefault();
                  void run(async () => {
                    await dataset("label", { anchorId: session.id, payload: {
                      evidence_id: session.evidence_id, intent, memory_worthy: worthy === "true", memory_note: memoryNote || null,
                    } });
                    await load(offset, session.id);
                    setNotice("Label saved.");
                  });
                }}>
                <label className="flex flex-col gap-1 text-xs text-slate-400">Primary intent
                  <select value={intent} onChange={(event) => setIntent(event.target.value as Intent | "")} className={inputStyle} required>
                    <option value="">Select intent</option>
                    {INTENTS.map((value) => <option key={value} value={value}>{value[0].toUpperCase() + value.slice(1)}</option>)}
                  </select>
                </label>
                <label className="flex flex-col gap-1 text-xs text-slate-400">Worth remembering?
                  <select value={worthy} onChange={(event) => setWorthy(event.target.value)} className={inputStyle} required>
                    <option value="">Select decision</option>
                    <option value="true">Yes</option><option value="false">No</option>
                  </select>
                </label>
                {worthy === "true" && <label className="flex w-full flex-col gap-1 text-xs text-slate-400">
                  Optional memory note from the participant
                  <textarea maxLength={1000} value={memoryNote} onChange={(event) => setMemoryNote(event.target.value)}
                    className="input" placeholder="What should the assistant remember from this session?" />
                </label>}
                <button type="submit" className="btn-accent rounded-lg px-4 py-2 text-sm disabled:opacity-40" disabled={busy || !intent || !worthy}>Save label</button>
                {(session.label || session.needs_review) && <button type="button" className={buttonStyle} disabled={busy} onClick={() => void run(async () => {
                  await dataset("remove", { anchorId: session.id });
                  await load(offset, session.id);
                  setNotice("Label removed.");
                })}>Remove label</button>}
              </form>
            </section>
          )}
        </div>
      )}

      <section className="surface space-y-3 rounded-xl p-4">
        <h2 className="text-sm font-semibold">Export reviewed sessions</h2>
        <label className="flex gap-2 text-xs text-slate-400">
          <input type="checkbox" checked={includeText} onChange={(event) => setIncludeText(event.target.checked)} />
          Include page titles approved for export
        </label>
        <label className="flex gap-2 text-xs text-slate-400">
          <input type="checkbox" checked={includeVisual} onChange={(event) => setIncludeVisual(event.target.checked)} />
          Include locally computed screenshot embeddings
        </label>
        <button className={buttonStyle} disabled={busy || !queue?.reviewed}
          onClick={() => void run(async () => {
            const result = await dataset<{ saved: boolean; rows: number }>("export", {
              payload: { include_text: includeText, include_visual: includeVisual },
            });
            if (result.saved) setNotice(`Exported ${result.rows} labeled sessions.`);
          })}>Save CSV</button>
      </section>
    </div>
  );
}
