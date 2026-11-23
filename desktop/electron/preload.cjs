const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktop", {
  request: (method, endpoint, body) => ipcRenderer.invoke("api", method, endpoint, body),
  dataset: (action, input) => ipcRenderer.invoke("dataset", action, input),
});
