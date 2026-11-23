export const API = "http://localhost:8000";

export async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  if (!window.desktop) throw new Error("Open AIML-EL in the desktop app.");
  try {
    return await window.desktop.request<T>(method, path, body);
  } catch (error) {
    if (!path.startsWith("/api/auth/") && /401 /.test(String(error))) {
      window.dispatchEvent(new Event("aiml-session-expired"));
    }
    throw error;
  }
}

export const get = <T,>(path: string) => request<T>("GET", path);
export const post = <T = unknown,>(path: string, body?: unknown) => request<T>("POST", path, body);
export const patch = <T = unknown,>(path: string, body: unknown) => request<T>("PATCH", path, body);
export const del = <T = unknown,>(path: string) => request<T>("DELETE", path);

export const fmtTime = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—";

export const fmtAgo = (iso: string | null | undefined) => {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
};

export const fmtDuration = (sec: number | null | undefined) => {
  if (!sec) return "0s";
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  return m < 60 ? `${m}m ${sec % 60}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
};
