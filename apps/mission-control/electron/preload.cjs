const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("missionControl", {
  request: (request) => ipcRenderer.invoke("orchestrator:request", request),
  selectAndUploadDataset: (options) => ipcRenderer.invoke("dataset:selectAndUpload", options),
  selectDatasetFolder: () => ipcRenderer.invoke("dataset:selectFolder"),
  preflightDatasetFolder: (options) => ipcRenderer.invoke("dataset:preflightFolder", options),
  preflightCloud: (options) => ipcRenderer.invoke("cloud:preflight", options),
  uploadDatasetFolder: (options) => ipcRenderer.invoke("dataset:uploadFolder", options),
  selectDemoImage: () => ipcRenderer.invoke("demo:selectImage"),
  predictChampionDemoLocal: (options) => ipcRenderer.invoke("demo:predictChampionLocal", options),
  disposeChampionDemoLocalRuntime: (options) => ipcRenderer.invoke("demo:disposeChampionLocalRuntime", options),
  loadModelArtifact: (options) => ipcRenderer.invoke("artifact:loadModel", options),
  saveArtifact: (options) => ipcRenderer.invoke("artifact:save", options),
  saveExportArtifact: (options) => ipcRenderer.invoke("artifact:saveExport", options),
  ensureProjectWorker: (options) => ipcRenderer.invoke("worker:ensureProjectWorker", options),
  stopProjectWorker: (options) => ipcRenderer.invoke("worker:stopProjectWorker", options),
});
