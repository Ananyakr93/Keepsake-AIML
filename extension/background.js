// Track foreground dwell and queue completed intervals across MV3 worker restarts.
importScripts("client.js");
void chrome.storage.local.setAccessLevel({ accessLevel: "TRUSTED_CONTEXTS" });
const FLUSH_ALARM = "aiml-el-flush";
const MIN_DWELL_MS = 3000;
const MAX_QUEUE = 1000;
const MAX_QUEUE_AGE_MS = 24 * 60 * 60 * 1000;
const STATE_KEY = "activityTracker";

let operations = Promise.resolve();
let flushing = false;

function run(action) {
  const result = operations.then(action);
  operations = result.catch((err) => console.warn("AIML-EL activity tracker:", err));
  return result;
}

async function load() {
  const { [STATE_KEY]: saved } = await chrome.storage.session.get(STATE_KEY);
  return saved || { browserId: crypto.randomUUID(), active: null, queue: [] };
}

function save(state) {
  return chrome.storage.session.set({ [STATE_KEY]: state });
}

async function foreground(policy) {
  const [{ trackingEnabled }, win, idle] = await Promise.all([
    chrome.storage.local.get({ trackingEnabled: false }),
    chrome.windows.getLastFocused({ populate: true }).catch(() => null),
    chrome.idle.queryState(15),
  ]);
  const tab = win?.focused && !win.incognito && idle === "active"
    ? win.tabs?.find((candidate) => candidate.active)
    : null;
  return { trackingEnabled, tab: tab && permittedTab(tab, policy) };
}

function closeInterval(state, now) {
  const { active } = state;
  const elapsed = now - active.since;
  if (elapsed < MIN_DWELL_MS) return;
  state.queue.push({
    url: active.url,
    title: active.title,
    visited_at: new Date(active.since).toISOString(),
    time_spent_sec: Math.floor(elapsed / 1000),
    session_id: state.browserId,
    client_event_id: crypto.randomUUID(),
    client_visit_id: active.visitId,
    consent_id: state.consent,
  });
  active.since = now;
}

async function update(now, checkpoint = false) {
  const state = await load();
  let policy;
  try {
    policy = await callApi("/api/capture-policy");
  } catch {
    state.active = null;
    await save(state);
    return [];
  }
  if (state.consent !== policy.consent_id) {
    state.active = null;
    state.queue = [];
    state.consent = policy.consent_id;
  }
  const { trackingEnabled, tab } = await foreground(policy);
  if (!trackingEnabled || !policy.browsing) {
    state.active = null;
    state.queue = [];
  } else {
    const active = state.active;
    const samePage = active && tab && active.id === tab.id && active.url === tab.url;
    if (samePage) {
      if (tab.title) active.title = tab.title;
      if (checkpoint) closeInterval(state, now);
    } else {
      if (active) closeInterval(state, now);
      state.active = tab && { ...tab, since: now, visitId: crypto.randomUUID() };
    }
    const earliest = new Date(now - MAX_QUEUE_AGE_MS).toISOString();
    state.queue = state.queue.filter((event) => event.visited_at >= earliest).slice(-MAX_QUEUE);
  }
  await save(state);
  return state.queue;
}

function track() {
  const now = Date.now();
  return run(() => update(now));
}

async function flush() {
  if (flushing) return;
  flushing = true;
  const now = Date.now();
  try {
    const events = await run(() => update(now, true));
    if (!events.length) return;
    await callApi("/api/events", { events });
    const sent = new Set(events.map((event) => event.client_event_id));
    await run(async () => {
      const state = await load();
      state.queue = state.queue.filter((event) => !sent.has(event.client_event_id));
      await save(state);
    });
  } catch (error) {
    console.warn("AIML-EL could not send browsing activity:", error.message);
  } finally {
    flushing = false;
  }
}

chrome.idle.setDetectionInterval(15);
chrome.tabs.onActivated.addListener(track);
chrome.tabs.onUpdated.addListener((_tabId, change, tab) => {
  if (tab.active && (change.url || change.title || change.status === "complete")) track();
});
chrome.tabs.onRemoved.addListener(track);
chrome.windows.onFocusChanged.addListener(track);
chrome.idle.onStateChanged.addListener(track);
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.trackingEnabled) track();
});
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === FLUSH_ALARM) flush();
});
async function startTracking() {
  if (!await chrome.alarms.get(FLUSH_ALARM)) {
    await chrome.alarms.create(FLUSH_ALARM, { periodInMinutes: 0.5 });
  }
  await track();
}
startTracking().catch((error) => console.warn("AIML-EL tracker startup:", error.message));

async function injectionEnabled() {
  const settings = await chrome.storage.local.get({ injectionEnabled: false, pairingKey: "" });
  return settings.injectionEnabled && Boolean(settings.pairingKey);
}

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local" || !(changes.injectionEnabled || changes.pairingKey)) return;
  injectionEnabled().then(async (enabled) => {
    for (const tab of await chrome.tabs.query({})) {
      if (tab.id) chrome.tabs.sendMessage(tab.id, { type: "injectionPolicy", enabled }).catch(() => {});
    }
  }).catch(() => {});
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  let host;
  try { host = new URL(sender.url).hostname; } catch { return; }
  if (!["chatgpt.com", "claude.ai", "gemini.google.com"].includes(host)) return;
  if (msg?.type === "getInjectionPolicy") {
    injectionEnabled().then((enabled) => sendResponse({ enabled })).catch(() => sendResponse({ enabled: false }));
    return true;
  }
  if (msg?.type !== "getContext" || typeof msg.query !== "string") return;
  injectionEnabled().then((enabled) => enabled
    ? callApi("/api/context", { query: msg.query, site: host })
    : { context: "" }).then(sendResponse).catch(() => sendResponse(null));
  return true;
});
