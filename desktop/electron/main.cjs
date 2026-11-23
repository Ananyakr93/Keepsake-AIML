// AIML-EL desktop app — Electron main process.
// Owns the window and authenticated local API calls.

const { app, BrowserWindow, dialog, ipcMain, protocol, shell } = require("electron");
const { mkdir, readFile, writeFile } = require("node:fs/promises");
const path = require("path");

const API = "http://localhost:8000";

let mainWindow = null;
let sessionToken = "";

// ---------------------------------------------------------------- window ---
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 840,
    minWidth: 960,
    minHeight: 640,
    backgroundColor: "#0b0d12",
    title: "AIML-EL — Personalized AI Memory",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  const startUrl = process.env.ELECTRON_START_URL;
  const devUrl = startUrl ? new URL(startUrl) : null;
  if (devUrl && devUrl.protocol === "http:" && ["localhost", "127.0.0.1"].includes(devUrl.hostname)) {
    mainWindow.loadURL(devUrl.href).catch((e) => console.error("loadURL failed:", e));
  } else {
    mainWindow
      .loadFile(path.join(__dirname, "../dist/index.html"))
      .catch((e) => console.error("loadFile failed:", e));
  }
  mainWindow.webContents.on("did-fail-load", (_e, code, desc, url) =>
    console.error(`did-fail-load ${code} ${desc} ${url}`)
  );
  mainWindow.webContents.on("console-message", (_e, level, message) => {
    if (level >= 2) console.error("renderer console:", message);
  });
  // Ctrl +/- / 0 zoom (the app has no menu, so the built-in accelerators are absent).
  mainWindow.webContents.on("before-input-event", (event, input) => {
    if (input.type !== "keyDown" || !input.control) return;
    const wc = mainWindow.webContents;
    if (input.key === "+" || input.key === "=") wc.setZoomLevel(Math.min(wc.getZoomLevel() + 0.5, 4));
    else if (input.key === "-" || input.key === "_") wc.setZoomLevel(Math.max(wc.getZoomLevel() - 0.5, -3));
    else if (input.key === "0") wc.setZoomLevel(0);
    else return;
    event.preventDefault();
  });
  mainWindow.webContents.on("will-navigate", (event) => event.preventDefault());
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.on("closed", () => {
    sessionToken = "";
    mainWindow = null;
  });
}

// aiml-el://open — used by the Chrome extension popup to focus this app.
protocol.registerSchemesAsPrivileged([
  { scheme: "aiml-el", privileges: { standard: true, secure: true } },
]);

app.whenReady().then(() => {
  protocol.handle("aiml-el", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    } else {
      createWindow();
    }
    return new Response("", { status: 200 });
  });
  createWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

const publicAuth = new Set(["/api/auth/state", "/api/auth/signup", "/api/auth/login"]);

function trustedFrame(event) {
  return event.sender === mainWindow?.webContents && event.senderFrame === mainWindow.webContents.mainFrame;
}

function setupKeyPath() {
  return process.env.AIML_SETUP_KEY_FILE || path.join(__dirname, "../../data/setup.key");
}

async function requestApi(method, endpoint, body) {
  const url = new URL(endpoint, API);
  if (!endpoint.startsWith("/api/") || url.origin !== API || !url.pathname.startsWith("/api/") ||
      !["GET", "POST", "PATCH", "PUT", "DELETE"].includes(method)) {
    throw new Error("Invalid AIML API request");
  }
  const headers = { "Content-Type": "application/json", "X-AIML-Token": sessionToken };
  if (publicAuth.has(url.pathname)) {
    headers["X-AIML-Setup"] = (await readFile(setupKeyPath(), "utf8")).trim();
  }
  if (url.pathname === "/api/auth/logout") sessionToken = "";
  const response = await fetch(url, {
    method, headers, body: body === undefined ? undefined : JSON.stringify(body),
    redirect: "error", signal: AbortSignal.timeout(30000),
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  if (url.pathname.endsWith("/image")) {
    const bytes = Buffer.from(await response.arrayBuffer());
    return `data:${response.headers.get("content-type")};base64,${bytes.toString("base64")}`;
  }
  const data = await response.json();
  if (url.pathname === "/api/auth/login") {
    sessionToken = data.token;
    delete data.token;
  }
  return data;
}

ipcMain.handle("api", (event, method, endpoint, body) => {
  if (!trustedFrame(event)) throw new Error("API unavailable");
  if (method === "GET" && endpoint === "/api/training/judgments") {
    return requestApi(method, endpoint, body).then(async ({ json }) => {
      const target = await saveDataset("relevance-judgments.json", "json");
      if (target.canceled || !target.filePath) return { saved: false };
      await writeFile(target.filePath, json, { encoding: "utf8", mode: 0o600 });
      return { saved: true };
    });
  }
  return requestApi(method, endpoint, body);
});

async function saveDataset(filename, extension) {
  const directory = path.join(path.dirname(setupKeyPath()), "datasets");
  await mkdir(directory, { recursive: true, mode: 0o700 });
  return dialog.showSaveDialog(mainWindow, { defaultPath: path.join(directory, filename),
    filters: [{ name: "AIML dataset", extensions: [extension] }] });
}

const datasetRoutes = {
  list: ["GET", "/sessions"],
  collection: ["POST", "/collection"],
  label: ["POST", "/sessions/"],
  analysis: ["GET", "/analysis/"],
  remove: ["DELETE", "/sessions/"],
  export: ["POST", "/export"],
};

ipcMain.handle("dataset", async (event, action, input = {}) => {
  if (!trustedFrame(event) || !Object.hasOwn(datasetRoutes, action)) {
    throw new Error("Dataset action unavailable");
  }

  const [method, route] = datasetRoutes[action];
  let endpoint = `/api/dataset${route}`;
  if (action === "list") endpoint += `?offset=${encodeURIComponent(input.offset ?? 0)}`;
  if (["label", "remove", "analysis"].includes(action)) {
    if (!Number.isSafeInteger(input.anchorId) || input.anchorId < 1) throw new Error("Invalid session");
    endpoint += input.anchorId;
  }
  const data = await requestApi(method, endpoint, input.payload);
  if (action !== "export") return data;

  const target = await saveDataset("aiml-dataset.csv", "csv");
  if (target.canceled || !target.filePath) return { saved: false };
  await writeFile(target.filePath, data.csv, { encoding: "utf8", mode: 0o600 });
  return { saved: true, rows: data.rows };
});
