// AIML-EL — chat-site content script.
//
// Preview optional memory context and protect edits made while retrieval is pending.

const HOST = location.hostname;

const SITES = {
  "chatgpt.com": {
    composer: () =>
      document.querySelector("#prompt-textarea") ||
      document.querySelector('div[contenteditable="true"]'),
    sendButton: () =>
      document.querySelector('button[data-testid="send-button"]') ||
      document.querySelector('button[aria-label="Send prompt"]') ||
      document.querySelector('button[aria-label="Send message"]') ||
      document.querySelector("form button[type='submit']"),
  },
  "claude.ai": {
    composer: () => document.querySelector('div[contenteditable="true"].ProseMirror'),
    sendButton: () =>
      document.querySelector('button[aria-label="Send message"]') ||
      document.querySelector('button[aria-label="Send Message"]'),
  },
  "gemini.google.com": {
    composer: () => document.querySelector("rich-textarea .ql-editor"),
    sendButton: () =>
      document.querySelector("button.send-button") ||
      document.querySelector("button[aria-label*='Send message' i]"),
  },
};

const site = SITES[HOST];
let injecting = false; // guard against intercepting our own re-send

function composerText(el) {
  return (el?.innerText || el?.value || "").trim();
}

let enabled = false;
chrome.runtime.sendMessage({ type: "getInjectionPolicy" }).then((result) => {
  enabled = result?.enabled === true;
}).catch(() => {});
chrome.runtime.onMessage.addListener((message) => {
  if (message?.type === "injectionPolicy") enabled = message.enabled === true;
});

function fetchContext(query) {
  return Promise.race([
    chrome.runtime.sendMessage({ type: "getContext", query }),
    new Promise((resolve) => setTimeout(() => resolve(null), 5000)),
  ]);
}

function appendContext(editor, context) {
  editor.focus();
  const suffix = "\n\n" + context;
  if (typeof editor.setSelectionRange === "function") {
    editor.setSelectionRange(editor.value.length, editor.value.length);
  } else {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(editor);
    range.collapse(false);
    selection.removeAllRanges();
    selection.addRange(range);
  }
  return document.execCommand("insertText", false, suffix);
}

function toast(message, ok = true) {
  const host = document.createElement("div");
  host.style.cssText = `
    position: fixed; bottom: 20px; right: 20px; z-index: 2147483647;
    font: 13px/1.4 system-ui, sans-serif; color: #e5e7eb;
    background: ${ok ? "rgba(17,24,39,.95)" : "rgba(120,20,30,.95)"};
    border: 1px solid rgba(255,255,255,.12); border-radius: 10px;
    padding: 10px 14px; box-shadow: 0 8px 24px rgba(0,0,0,.35);
    max-width: 320px; pointer-events: none; transition: opacity .3s;`;
  host.textContent = message;
  document.documentElement.appendChild(host);
  setTimeout(() => { host.style.opacity = "0"; }, 2600);
  setTimeout(() => host.remove(), 3100);
}

async function handleSend(ev) {
  if (injecting || !site || !enabled) return;
  const composer = site.composer();
  const sendBtn = site.sendButton();
  const clickedSend = ev.type === "click" && sendBtn?.contains(ev.target);
  const enter = ev.type === "keydown" && !ev.isComposing && !ev.repeat &&
    ev.key === "Enter" && !ev.shiftKey && !ev.ctrlKey && !ev.altKey && !ev.metaKey &&
    composer?.contains(ev.target);
  if ((!clickedSend && !enter) || !composer || !sendBtn || sendBtn.disabled) return;

  const text = composerText(composer);
  if (!text || text.includes("[Personal memory context")) return;

  injecting = true;
  ev.preventDefault();
  ev.stopImmediatePropagation();
  try {
    const response = await fetchContext(text).catch(() => null);
    if (site.composer() !== composer || composerText(composer) !== text) {
      toast("Your draft changed. Review it and press Send again.", false);
      return;
    }
    const context = enabled && response?.context;
    if (context && window.confirm("Include these memories in the message sent to " + HOST + "?\n\n" + context)) {
      if (!appendContext(composer, context)) {
        toast("Could not attach the memories. Review the draft before sending.", false);
        return;
      }
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    const button = site.sendButton();
    if (site.composer() === composer && button && !button.disabled) button.click();
    else toast("Review the draft and press Send again.", false);
  } finally {
    injecting = false;
  }
}

document.addEventListener("click", handleSend, true);
document.addEventListener("keydown", handleSend, true);
