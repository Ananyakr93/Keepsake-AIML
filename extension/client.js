const LOCAL_API = "http://localhost:8000";

async function callApi(endpoint, body) {
  const { pairingKey } = await chrome.storage.local.get("pairingKey");
  if (!pairingKey) throw new Error("Pair this extension in the desktop settings first.");
  const response = await fetch(LOCAL_API + endpoint, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json", "X-AIML-Token": pairingKey },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(4500),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || "Local request failed.");
  return result;
}

function permittedTab(tab, policy) {
  try {
    const url = new URL(tab.url);
    if (tab.incognito || !["http:", "https:"].includes(url.protocol) || url.origin === LOCAL_API) return null;
    const host = url.hostname.toLowerCase().replace(/\.$/, "");
    if (policy.blocked_domains.some((domain) => host === domain || host.endsWith("." + domain))) return null;
    const title = tab.title || "";
    const full = (url.href + " " + title).toLowerCase();
    if (policy.exclusions.some((phrase) => full.includes(phrase))) return null;
    url.username = url.password = url.search = url.hash = "";
    return { id: tab.id, url: url.href, title };
  } catch {
    return null;
  }
}
