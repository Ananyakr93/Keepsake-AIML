const status = document.getElementById("captureStatus");
const error = document.getElementById("err");
const capture = document.getElementById("capture");

async function refresh() {
  try {
    const policy = await callApi("/api/capture-policy");
    const { trackingEnabled } = await chrome.storage.local.get({ trackingEnabled: false });
    status.textContent = policy.browsing && trackingEnabled ? "Browsing capture is on." : "Browsing capture is paused.";
    capture.disabled = !policy.screenshots;
    document.getElementById("pair").hidden = true;
  } catch (issue) {
    status.textContent = issue.message;
    capture.disabled = true;
    document.getElementById("pair").hidden = false;
  }
}

function bindToggle(id, key) {
  const item = document.getElementById(id);
  const render = (enabled) => {
    item.textContent = enabled ? "ON" : "OFF";
    item.className = "toggle " + (enabled ? "on" : "off");
  };
  chrome.storage.local.get({ [key]: false }).then((state) => render(state[key]));
  item.onclick = async () => {
    const state = await chrome.storage.local.get({ [key]: false });
    await chrome.storage.local.set({ [key]: !state[key] });
    render(!state[key]);
    await refresh();
  };
}

bindToggle("trackingToggle", "trackingEnabled");
bindToggle("injectionToggle", "injectionEnabled");
document.getElementById("pair").onsubmit = async (event) => {
  event.preventDefault();
  const key = document.getElementById("key");
  await chrome.storage.local.set({ pairingKey: key.value.trim() });
  key.value = "";
  await refresh();
};
capture.onclick = async () => {
  capture.disabled = true;
  error.textContent = "";
  try {
    const policy = await callApi("/api/capture-policy");
    const [active] = await chrome.tabs.query({ active: true, currentWindow: true });
    const tab = policy.screenshots && active && permittedTab(active, policy);
    if (!tab) throw new Error("Capture is paused or the active tab is excluded.");
    const capturedAt = new Date().toISOString();
    const image = await chrome.tabs.captureVisibleTab(active.windowId, { format: "png" });
    const [current] = await chrome.tabs.query({ active: true, currentWindow: true });
    const confirmed = current && permittedTab(current, policy);
    if (confirmed?.id !== tab.id || current.url !== active.url || confirmed.title !== tab.title) {
      throw new Error("The active tab changed. Try again.");
    }
    const saved = await callApi("/api/screenshots", {
      image_b64: image.split(",")[1], source_url: tab.url, title: tab.title,
      captured_at: capturedAt, consent_id: policy.consent_id,
    });
    await refresh();
    status.textContent = saved.skipped_duplicate ? "A matching capture already exists." : "Tab screenshot saved.";
  } catch (issue) {
    await refresh();
    error.textContent = issue.message;
  }
};
void refresh();
