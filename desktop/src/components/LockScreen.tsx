import { useEffect, useRef, useState } from "react";
import { get, post } from "../api";
import Icon from "./Icon";

type Mode = "signin" | "signup";

/** Full-window blur overlay + modal. Shown on launch and after 5 min idle. */
export default function LockScreen({
  hasAccounts,
  lastUser,
  onUnlock,
}: {
  hasAccounts: boolean;
  lastUser: string;
  onUnlock: (username: string) => void;
}) {
  const [mode] = useState<Mode>(hasAccounts ? "signin" : "signup");
  const username = lastUser;
  const [pin, setPin] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const pinRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    pinRef.current?.focus();
  }, [mode]);

  const submit = async (e?: React.FormEvent, pinValue: string = pin) => {
    e?.preventDefault();
    if (busy) return;
    setError("");
    setBusy(true);
    try {
      if (mode === "signup") await post("/api/auth/signup", { username, pin: pinValue });
      const session = await post<{ username: string }>("/api/auth/login", { username, pin: pinValue });
      onUnlock(session.username);
    } catch (err) {
      const raw = String((err as Error).message);
      let msg = raw;
      try {
        msg = JSON.parse(raw.slice(raw.indexOf("{"))).detail ?? raw;
      } catch { /* keep raw */ }
      setError(typeof msg === "string" ? msg : "Something went wrong");
      setPin(""); // let the user retry
      pinRef.current?.focus();
    } finally {
      setBusy(false);
    }
  };

  const title = mode === "signin" ? "Unlock Keepsake" : "Set your PIN";

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center lock-shade">
      <form onSubmit={submit} className="surface pop-in w-[380px] rounded-2xl p-7">
        <div className="mb-5 flex items-center gap-3">
          <div className="btn-accent flex h-11 w-11 items-center justify-center rounded-xl">
            <Icon name="lock" className="h-5 w-5" />
          </div>
          <div>
            <div className="text-xl font-bold tracking-tight">{title}</div>
            <div className="text-xs text-slate-500">
              Enter a six-digit PIN
            </div>
          </div>
        </div>

        <label className="mb-1 block text-xs font-semibold text-slate-500">Username</label>
        <input
          className="input mb-3"
          value={username}
          readOnly
          autoComplete="username"
          placeholder="your_username"
        />

            <label className="mb-1 block text-xs font-semibold text-slate-500">PIN</label>
            <input
              ref={pinRef}
              className="input mb-1 text-center text-xl tracking-[0.6em]"
              type="password"
              inputMode="numeric"
              maxLength={6}
              value={pin}
              onChange={(e) => {
                const v = e.target.value.replace(/\D/g, "");
                setPin(v);
                if (mode === "signin" && v.length === 6 && username.trim()) submit(undefined, v); // auto-unlock
              }}
              placeholder="••••••"
            />

        <div className="mt-2 min-h-[18px] text-xs">
          {error && <span className="text-red-400">{error}</span>}
        </div>

        <button disabled={busy || pin.length !== 6} className="btn-accent mt-3 w-full rounded-xl py-2.5 text-sm">
          {mode === "signup" ? "Set PIN" : "Unlock"}
        </button>
      </form>
    </div>
  );
}

export const fetchAccountState = () => get<{ accounts: number; username: string }>("/api/auth/state");
