const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const { runInNewContext } = require("node:vm");

const content = readFileSync("extension/content.js", "utf8");

async function simulate({ enabled, approve, changed = false }) {
  const events = {};
  const sent = [];
  let confirmCount = 0;
  const editor = { innerText: "my draft", contains: () => true, focus() {} };
  const button = { disabled: false, contains: () => true, click: () => sent.push(editor.innerText) };
  runInNewContext(content, {
    location: { hostname: "chatgpt.com" },
    chrome: { runtime: {
      sendMessage(message) {
        if (message.type === "getInjectionPolicy") return Promise.resolve({ enabled });
        if (changed) editor.innerText = "my revised draft";
        return Promise.resolve({ context: "[Personal memory context] Example" });
      },
      onMessage: { addListener() {} },
    } },
    document: {
      querySelector: (selector) => selector === "#prompt-textarea" ? editor : button,
      addEventListener: (name, fn) => { events[name] = fn; },
      execCommand: (_name, _ui, suffix) => { editor.innerText += suffix; return true; },
      createRange: () => ({ selectNodeContents() {}, collapse() {} }),
      createElement: () => ({ style: {}, remove() {} }),
      documentElement: { appendChild() {} },
    },
    window: {
      confirm() { confirmCount++; return approve; },
      getSelection: () => ({ removeAllRanges() {}, addRange() {} }),
    },
    requestAnimationFrame: (resolve) => resolve(),
    setTimeout: () => 0,
  });
  await Promise.resolve();
  let intercepted = false;
  await events.click({ type: "click", target: button, preventDefault() { intercepted = true; },
    stopImmediatePropagation() {} });
  return { sent, intercepted, confirmCount };
}

test("injection off leaves sends untouched", async () => {
  assert.deepEqual(await simulate({ enabled: false }), { sent: [], intercepted: false, confirmCount: 0 });
});

test("approved context appends once, declined context sends only the draft", async () => {
  assert.deepEqual((await simulate({ enabled: true, approve: true })).sent,
    ["my draft\n\n[Personal memory context] Example"]);
  assert.deepEqual((await simulate({ enabled: true, approve: false })).sent, ["my draft"]);
});

test("a changed draft is never sent by the older request", async () => {
  const output = await simulate({ enabled: true, approve: true, changed: true });
  assert.deepEqual(output.sent, []);
  assert.equal(output.confirmCount, 0);
});
