import { useEffect, useState } from "react";
import { get, post, del, request } from "../api";
import Icon from "../components/Icon";

interface ShownMemory {
  id: string;
  content: string;
  similarity: number;
  importance: number;
  vote: number;
}
interface Shown {
  id: string;
  text: string;
  gate_open: boolean;
  memories: ShownMemory[];
}
interface PromptRow {
  id: string;
  text: string;
  up: number;
  down: number;
}

export default function TrainMemories() {
  const [text, setText] = useState("");
  const [prompts, setPrompts] = useState<PromptRow[]>([]);
  const [shown, setShown] = useState<Shown | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [trainMsg, setTrainMsg] = useState("");

  const retrainNow = async () => {
    setTrainMsg("Updating the ranking model…");
    try {
      const r = await post<{ retrained: boolean; votes: number; reason?: string; validation?: { baseline: number; candidate: number } }>("/api/training/retrain");
      const quality = r.validation ? ` Held-out NDCG@5 ${r.validation.baseline.toFixed(3)} → ${r.validation.candidate.toFixed(3)}.` : "";
      setTrainMsg((r.retrained ? `Ranking model updated from ${r.votes} votes.` : r.reason ?? "") + quality);
    } catch (e) {
      setTrainMsg(`Update failed: ${e}`);
    }
  };

  const loadPrompts = () =>
    get<{ prompts: PromptRow[] }>("/api/training/prompts").then((r) => setPrompts(r.prompts));
  useEffect(() => { loadPrompts().catch(() => {}); }, []);

  const submit = async () => {
    if (!text.trim()) return;
    setBusy(true);
    setError("");
    try {
      setShown(await post<Shown>("/api/training/prompts", { text }));
      setText("");
      await loadPrompts();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const open = (id: string) => get<Shown>(`/api/training/prompts/${id}`).then(setShown);

  const vote = async (m: ShownMemory, v: 1 | -1) => {
    if (!shown) return;
    const next = m.vote === v ? 0 : v; // clicking the active vote clears it
    try {
      await request("PUT", `/api/training/prompts/${shown.id}/votes`, { memory_id: m.id, vote: next });
    } catch (error) {
      setError(String(error));
      return;
    }
    setTrainMsg("");
    setShown({ ...shown, memories: shown.memories.map((x) => (x.id === m.id ? { ...x, vote: next } : x)) });
    loadPrompts();
  };

  const remove = async (id: string) => {
    await del(`/api/training/prompts/${id}`);
    if (shown?.id === id) setShown(null);
    loadPrompts();
  };

  const totalVotes = prompts.reduce((n, p) => n + p.up + p.down, 0);

  return (
    <div className="mx-auto max-w-5xl p-8">
      <h1 className="page-title">Train memories</h1>
      <p className="mb-4 text-xs text-slate-500">
        Type a message you might send to a chatbot. The memories the app would pick are listed below.
        Give each one a thumbs up if it should be used for that message, or a thumbs down if not. Votes are saved
        ({totalVotes} so far). Click Finish when you are done, to evaluate a ranking update on held-out prompts.
      </p>
      <button className="surface mb-3 rounded-lg px-3 py-2 text-xs" onClick={() =>
        get<{ saved: boolean }>("/api/training/judgments")
          .then((result) => { if (result.saved) setTrainMsg("Judgments saved."); })
          .catch((error) => setError(String(error)))}>Export private relevance judgments</button>
      <div className="flex gap-2">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="e.g. Which laptop should I buy?"
          className="flex-1 rounded-xl surface px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:border-blue-500/50 focus:outline-none"
        />
        <button onClick={submit} disabled={busy || !text.trim()} className="btn-accent rounded-xl px-5 text-sm font-semibold disabled:opacity-40">
          {busy ? "Searching…" : "Find memories"}
        </button>
      </div>
      {error && <div className="mt-2 text-xs text-red-400">{error}</div>}
      <div className="mt-3 flex items-center gap-3">
        <button onClick={retrainNow} className="rounded-xl surface px-4 py-2 text-xs font-semibold text-slate-200 hover:border-blue-400/60">
          Finish and update model
        </button>
        {trainMsg && <span className="text-xs text-slate-400">{trainMsg}</span>}
      </div>

      <div className="mt-6 grid gap-6 md:grid-cols-[1fr_260px]">
        <div>
          {shown ? (
            <>
              <div className="mb-2 text-sm font-medium text-slate-200">“{shown.text}”</div>
              {!shown.gate_open && (
                <div className="mb-3 text-[11px] text-amber-400">
                  The gate would not inject any memory for this message, so none would be sent to the chatbot.
                  They are shown here so you can still train on them.
                </div>
              )}
              {shown.memories.length === 0 && <div className="text-xs text-slate-600">No memories yet.</div>}
              <div className="space-y-2">
                {shown.memories.map((m) => (
                  <div key={m.id} className="surface flex items-start gap-3 rounded-xl p-3">
                    <div className="min-w-0 flex-1">
                      <div className="text-[13px] leading-5 text-slate-200">{m.content}</div>
                      <div className="mt-1 text-[10px] text-slate-600">
                        similarity {(m.similarity * 100).toFixed(0)}% · importance {(m.importance * 100).toFixed(0)}%
                      </div>
                    </div>
                    <button
                      onClick={() => vote(m, 1)}
                      title="Relevant"
                      className={`rounded-lg p-2 ${m.vote === 1 ? "bg-emerald-500/25 text-emerald-300" : "text-slate-500 hover:bg-emphasis hover:text-ink"}`}
                    >
                      <Icon name="thumbUp" className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => vote(m, -1)}
                      title="Not relevant"
                      className={`rounded-lg p-2 ${m.vote === -1 ? "bg-red-500/25 text-red-300" : "text-slate-500 hover:bg-emphasis hover:text-ink"}`}
                    >
                      <Icon name="thumbDown" className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="mt-10 text-center text-xs text-slate-600">Enter a prompt to see its memories.</div>
          )}
        </div>

        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Your prompts</div>
          <div className="space-y-1.5">
            {prompts.map((p) => (
              <div key={p.id} className={`surface flex items-center gap-2 rounded-lg px-3 py-2 ${shown?.id === p.id ? "border-blue-400/60" : ""}`}>
                <button onClick={() => open(p.id)} className="min-w-0 flex-1 text-left">
                  <div className="truncate text-[12px] text-slate-300">{p.text}</div>
                  <div className="text-[10px] text-slate-600">{p.up} up · {p.down} down</div>
                </button>
                <button onClick={() => remove(p.id)} title="Delete prompt" className="text-slate-600 hover:text-red-400">
                  <Icon name="trash" className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
            {prompts.length === 0 && <div className="text-xs text-slate-600">None yet.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
