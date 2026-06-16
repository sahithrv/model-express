const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("missionControl", {
  request: (request) => ipcRenderer.invoke("orchestrator:request", request),
  selectAndUploadDataset: (options) => ipcRenderer.invoke("dataset:selectAndUpload", options),
  selectDatasetFolder: () => ipcRenderer.invoke("dataset:selectFolder"),
  preflightDatasetFolder: (options) => ipcRenderer.invoke("dataset:preflightFolder", options),
  uploadDatasetFolder: (options) => ipcRenderer.invoke("dataset:uploadFolder", options),
  selectDemoImage: () => ipcRenderer.invoke("demo:selectImage"),
  loadModelArtifact: (options) => ipcRenderer.invoke("artifact:loadModel", options),
  saveArtifact: (options) => ipcRenderer.invoke("artifact:save", options),
  ensureProjectWorker: (options) => ipcRenderer.invoke("worker:ensureProjectWorker", options),
  stopProjectWorker: (options) => ipcRenderer.invoke("worker:stopProjectWorker", options),
});
