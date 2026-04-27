const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("missionControl", {
  request: (request) => ipcRenderer.invoke("orchestrator:request", request),
  selectAndUploadDataset: (options) => ipcRenderer.invoke("dataset:selectAndUpload", options),
  selectDatasetFolder: () => ipcRenderer.invoke("dataset:selectFolder"),
  uploadDatasetFolder: (options) => ipcRenderer.invoke("dataset:uploadFolder", options),
});
