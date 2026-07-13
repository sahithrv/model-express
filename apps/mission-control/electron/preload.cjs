const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("missionControl", {
  request: (request) => ipcRenderer.invoke("orchestrator:request", request),
  abortRequest: (requestId) => ipcRenderer.invoke("orchestrator:abortRequest", requestId),
  getFeatureFlags: () => ipcRenderer.invoke("runtime:featureFlags"),
  openEventStream: (options) => ipcRenderer.invoke("orchestrator:eventStream:open", options),
  closeEventStream: (streamId) => ipcRenderer.invoke("orchestrator:eventStream:close", streamId),
  onEventStreamMessage: (callback) => {
    const listener = (_event, message) => callback(message);
    ipcRenderer.on("orchestrator:eventStream", listener);
    return () => ipcRenderer.removeListener("orchestrator:eventStream", listener);
  },
  recordActivityVisibility: (summary) => ipcRenderer.invoke("diagnostics:activityVisibility", summary),
  recordActivityStreamAttempt: (summary) => ipcRenderer.invoke("diagnostics:activityStreamAttempt", summary),
  recordIncrementalLiveDiagnostic: (summary) => ipcRenderer.invoke("diagnostics:incrementalLive", summary),
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
