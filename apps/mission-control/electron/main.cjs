const { app, BrowserWindow, dialog, ipcMain, Menu } = require("electron");
const { spawn } = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const { Transform } = require("stream");
const { fileURLToPath, pathToFileURL } = require("url");
const { CreateBucketCommand, GetObjectCommand, HeadBucketCommand, PutObjectCommand, S3Client } = require("@aws-sdk/client-s3");
const { collectDatasetFiles, createZipArchiveStream, planZipArchive } = require("./zip-stream.cjs");

let mainWindow;
const projectWorkers = new Map();

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1320,
    height: 860,
    minWidth: 1120,
    minHeight: 720,
    backgroundColor: "#050807",
    title: "Model Express",
    autoHideMenuBar: true,
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "hidden",
    titleBarOverlay: process.platform === "darwin"
      ? undefined
      : {
          color: "#050807",
          symbolColor: "#dce7e2",
          height: 42,
        },
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (app.isPackaged) {
    mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
  } else {
    mainWindow.loadURL("http://127.0.0.1:5173");
  }
}

app.whenReady().then(() => {
  Menu.setApplicationMenu(null);
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  for (const worker of projectWorkers.values()) {
    if (worker.exitCode === null && !worker.killed) {
      worker.kill();
    }
  }
  projectWorkers.clear();
});

ipcMain.handle("orchestrator:request", async (_event, request) => {
  const { baseUrl, method = "GET", path: requestPath, body } = request;
  const url = new URL(requestPath, baseUrl);

  const response = await fetch(url, {
    method,
    headers: {
      "Content-Type": "application/json",
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }

  if (!response.ok) {
    const message =
      payload && typeof payload === "object" && payload.error
        ? payload.error
        : text || response.statusText;
    return {
      __mission_control_http_error: true,
      status: response.status,
      statusText: response.statusText,
      message,
      path: requestPath,
      url: url.toString(),
      payload,
    };
  }

  return payload;
});

ipcMain.handle("dataset:selectAndUpload", async (_event, options) => {
  const { projectId } = options;
  if (!projectId) {
    throw new Error("Select a project before uploading a dataset.");
  }

  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select image dataset folder",
    properties: ["openDirectory"],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }

  return uploadDatasetFolder({
    ...options,
    datasetPath: result.filePaths[0],
  });
});

ipcMain.handle("dataset:selectFolder", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select image dataset folder",
    properties: ["openDirectory"],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }

  const datasetPath = result.filePaths[0];
  return {
    path: datasetPath,
    name: path.basename(datasetPath),
  };
});

ipcMain.handle("dataset:uploadFolder", async (_event, options) => {
  return uploadDatasetFolder(options);
});

ipcMain.handle("demo:selectImage", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select image for champion test",
    properties: ["openFile"],
    filters: [
      { name: "Images", extensions: ["jpg", "jpeg", "png", "webp", "bmp"] },
      { name: "All files", extensions: ["*"] },
    ],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }

  const imagePath = result.filePaths[0];
  const stats = fs.statSync(imagePath);
  const imageURI = pathToFileURL(imagePath).toString();

  return {
    path: imagePath,
    name: path.basename(imagePath),
    uri: imageURI,
    image_uri: imageURI,
    image_id: path.basename(imagePath),
    thumbnail_uri: demoImagePreviewURI(imagePath, imageURI, stats.size),
    split: "custom",
    size_bytes: stats.size,
    metadata: {
      source: "local_file",
      file_name: path.basename(imagePath),
      size_bytes: stats.size,
    },
  };
});

ipcMain.handle("artifact:loadModel", async (_event, request) => {
  const artifactUri = String(request?.artifactUri ?? "").trim();
  if (!artifactUri) {
    throw new Error("artifactUri is required.");
  }
  const buffer = artifactUri.startsWith("s3://")
    ? await readS3ObjectBuffer(artifactUri, request)
    : readLocalArtifactBuffer(artifactUri);
  const externalData = artifactUri.startsWith("s3://")
    ? await readS3ExternalDataFiles(artifactUri, request)
    : readLocalExternalDataFiles(artifactUri, request);
  return {
    artifact_uri: artifactUri,
    size_bytes: buffer.byteLength,
    bytes: bufferToArrayBuffer(buffer),
    external_data: externalData,
  };
});

ipcMain.handle("worker:ensureProjectWorker", async (_event, options) => {
  return ensureProjectWorker(options);
});

ipcMain.handle("worker:stopProjectWorker", async (_event, options) => {
  return stopProjectWorker(options);
});

function demoImagePreviewURI(imagePath, fallbackURI, sizeBytes) {
  const maxInlinePreviewBytes = 8 * 1024 * 1024;
  if (sizeBytes > maxInlinePreviewBytes) {
    return fallbackURI;
  }
  try {
    const encoded = fs.readFileSync(imagePath).toString("base64");
    return `data:${imageMimeType(imagePath)};base64,${encoded}`;
  } catch {
    return fallbackURI;
  }
}

function imageMimeType(imagePath) {
  switch (path.extname(imagePath).toLowerCase()) {
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".png":
      return "image/png";
    case ".webp":
      return "image/webp";
    case ".bmp":
      return "image/bmp";
    default:
      return "application/octet-stream";
  }
}

function readLocalArtifactBuffer(artifactUri) {
  const artifactPath = localPathFromURI(artifactUri);
  if (!artifactPath) {
    throw new Error(`Unsupported model artifact URI: ${artifactUri}`);
  }
  return fs.readFileSync(artifactPath);
}

function readLocalExternalDataFiles(artifactUri, request = {}) {
  const artifactPath = localPathFromURI(artifactUri);
  if (!artifactPath) {
    return [];
  }
  const artifactDir = path.dirname(artifactPath);
  const out = [];
  const seen = new Set();
  for (const candidate of externalDataCandidates(artifactUri, request.externalData)) {
    const mountPath = externalDataMountPath(candidate.path);
    if (!mountPath || seen.has(mountPath)) {
      continue;
    }
    const sourcePath = externalDataLocalPath(candidate, artifactDir, mountPath);
    if (!sourcePath || !fs.existsSync(sourcePath) || !fs.statSync(sourcePath).isFile()) {
      if (candidate.explicit) {
        throw new Error(`Unable to load ONNX external data file: ${mountPath}`);
      }
      continue;
    }
    const buffer = fs.readFileSync(sourcePath);
    seen.add(mountPath);
    out.push({
      path: mountPath,
      uri: pathToFileURL(sourcePath).toString(),
      size_bytes: buffer.byteLength,
      bytes: bufferToArrayBuffer(buffer),
    });
  }
  return out;
}

async function readS3ExternalDataFiles(artifactUri, request = {}) {
  const out = [];
  const seen = new Set();
  for (const candidate of externalDataCandidates(artifactUri, request.externalData)) {
    const mountPath = externalDataMountPath(candidate.path);
    if (!mountPath || seen.has(mountPath)) {
      continue;
    }
    const sidecarUri = externalDataS3URI(artifactUri, candidate, mountPath);
    if (!sidecarUri) {
      continue;
    }
    try {
      const buffer = await readS3ObjectBuffer(sidecarUri, request);
      seen.add(mountPath);
      out.push({
        path: mountPath,
        uri: sidecarUri,
        size_bytes: buffer.byteLength,
        bytes: bufferToArrayBuffer(buffer),
      });
    } catch (error) {
      if (candidate.explicit) {
        throw new Error(`Unable to load ONNX external data file ${mountPath}: ${error.message}`);
      }
    }
  }
  return out;
}

function localPathFromURI(value) {
  if (process.platform === "win32" && value.length > 2 && value[1] === ":") {
    return value;
  }
  if (value.startsWith("file://")) {
    return fileURLToPath(value);
  }
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(value)) {
    return null;
  }
  return value;
}

function externalDataLocalPath(candidate, artifactDir, mountPath) {
  for (const key of ["uri", "artifact_uri", "artifactPath", "artifact_path", "localPath", "local_path"]) {
    const value = String(candidate[key] ?? "").trim();
    if (!value) {
      continue;
    }
    const localPath = localPathFromURI(value);
    if (!localPath) {
      continue;
    }
    return path.isAbsolute(localPath) ? localPath : path.resolve(artifactDir, localPath);
  }
  return path.resolve(artifactDir, storageRelativePath(mountPath));
}

function externalDataCandidates(artifactUri, externalData) {
  const candidates = Array.isArray(externalData)
    ? externalData
        .filter((item) => item && typeof item === "object")
        .map((item) => ({
          ...item,
          path: item.path ?? item.relative_path ?? item.file_name,
          explicit: true,
        }))
    : [];
  const inferred = `${artifactFileName(artifactUri)}.data`;
  if (inferred) {
    candidates.push({ path: inferred, explicit: false });
  }
  return candidates;
}

function artifactFileName(artifactUri) {
  if (artifactUri.startsWith("s3://") || artifactUri.startsWith("file://")) {
    return path.basename(decodeURIComponent(new URL(artifactUri).pathname));
  }
  return path.basename(artifactUri);
}

function externalDataMountPath(value) {
  const text = String(value ?? "").trim();
  if (!text || /^[a-z][a-z0-9+.-]*:\/\//i.test(text) || path.isAbsolute(text) || path.win32.isAbsolute(text)) {
    return "";
  }
  const parts = text.replace(/\\/g, "/").split("/");
  if (parts.some((part) => part === "..")) {
    return "";
  }
  return text;
}

function storageRelativePath(value) {
  return String(value)
    .replace(/\\/g, "/")
    .split("/")
    .filter((part) => part && part !== ".")
    .join("/");
}

function externalDataS3URI(artifactUri, candidate, mountPath) {
  for (const key of ["uri", "artifact_uri", "artifactPath", "artifact_path"]) {
    const value = String(candidate[key] ?? "").trim();
    if (value.startsWith("s3://")) {
      return value;
    }
  }
  const parsed = new URL(artifactUri);
  const baseDir = path.posix.dirname(parsed.pathname.replace(/^\/+/, ""));
  const key = path.posix.join(baseDir === "." ? "" : baseDir, storageRelativePath(mountPath));
  return `s3://${parsed.hostname}/${key}`;
}

function bufferToArrayBuffer(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

async function readS3ObjectBuffer(artifactUri, options = {}) {
  const parsed = new URL(artifactUri);
  const bucket = parsed.hostname;
  const key = parsed.pathname.replace(/^\/+/, "");
  if (!bucket || !key) {
    throw new Error(`Invalid S3 model artifact URI: ${artifactUri}`);
  }
  const client = createS3Client(options);
  const response = await client.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
  return streamToBuffer(response.Body);
}

function createS3Client(options = {}) {
  return new S3Client({
    endpoint: options.endpoint ?? process.env.S3_ENDPOINT_URL ?? "http://localhost:9000",
    region: options.region ?? process.env.AWS_DEFAULT_REGION ?? "us-east-1",
    forcePathStyle: true,
    credentials: {
      accessKeyId: options.accessKeyId ?? process.env.AWS_ACCESS_KEY_ID ?? "model_express",
      secretAccessKey: options.secretAccessKey ?? process.env.AWS_SECRET_ACCESS_KEY ?? "model_express_password",
    },
  });
}

async function streamToBuffer(stream) {
  if (!stream) return Buffer.alloc(0);
  const chunks = [];
  for await (const chunk of stream) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

function ensureProjectWorker(options) {
  const { projectId, baseUrl } = options;
  if (!projectId) {
    throw new Error("Project id is required before starting a worker.");
  }
  if (!baseUrl) {
    throw new Error("Orchestrator URL is required before starting a worker.");
  }

  const repoRoot = process.env.MODEL_EXPRESS_ROOT ?? path.resolve(__dirname, "..", "..", "..");
  const workerDir = path.join(repoRoot, "services", "worker");
  if (!fs.existsSync(workerDir)) {
    throw new Error(`Worker directory does not exist: ${workerDir}`);
  }

  const logDir = resolveLogDir(repoRoot);
  fs.mkdirSync(logDir, { recursive: true });
  const targetCount = normalizeWorkerCount(options.count);
  const useModalDispatcher = shouldUseModalDispatcher(options);
  const processCount = useModalDispatcher ? 1 : targetCount;
  const repoEnv = loadRepoEnv(repoRoot);
  const pids = [];
  let startedCount = 0;

  for (let slot = 1; slot <= processCount; slot += 1) {
    const key = projectWorkerKey(
      projectId,
      slot,
      useModalDispatcher ? "modal-dispatcher" : options.gpuType ?? "local",
    );
    const existing = projectWorkers.get(key);
    if (isWorkerRunning(existing)) {
      pids.push(existing.pid);
      continue;
    }

    const workerName = options.name
      ? `${options.name}-${slot}`
      : `project-${projectId}-worker-${slot}`;
    const childEnv = {
      ...process.env,
      ...repoEnv,
      ORCHESTRATOR_URL: baseUrl,
      PROJECT_ID: projectId,
      WORKER_NAME: workerName,
      GPU_TYPE: options.gpuType ?? "local",
      MODEL_EXPRESS_ROOT: repoRoot,
      MODEL_EXPRESS_LOG_DIR: process.env.MODEL_EXPRESS_LOG_DIR ?? repoEnv.MODEL_EXPRESS_LOG_DIR ?? logDir,
      PYTHONUNBUFFERED: "1",
    };
    if (useModalDispatcher) {
      childEnv.MODEL_EXPRESS_MODAL_DISPATCHER = "true";
      childEnv.MODEL_EXPRESS_MODAL_DISPATCHER_SLOTS = String(targetCount);
    }

    const child = startProjectWorker({
      projectId,
      slot,
      key,
      workerDir,
      childEnv,
    });

    pids.push(child.pid);
    startedCount += 1;
  }

  return {
    project_id: projectId,
    pid: pids[0] ?? 0,
    pids,
    started: startedCount > 0,
    started_count: startedCount,
    running_count: useModalDispatcher ? targetCount : pids.length,
    process_count: pids.length,
    dispatcher: useModalDispatcher,
    status: startedCount > 0 ? "started" : "already_running",
  };
}

function stopProjectWorker(options = {}) {
  const projectId = String(options.projectId ?? "").trim();
  if (!projectId) {
    throw new Error("Project id is required before stopping workers.");
  }
  const stopped = [];
  const alreadyStopped = [];
  for (const [key, worker] of projectWorkers.entries()) {
    if (!key.startsWith(`${projectId}:`)) {
      continue;
    }
    if (isWorkerRunning(worker)) {
      worker.kill();
      stopped.push(worker.pid);
    } else {
      alreadyStopped.push(worker.pid ?? 0);
    }
    projectWorkers.delete(key);
  }
  return {
    project_id: projectId,
    stopped_count: stopped.length,
    stopped_pids: stopped,
    already_stopped_count: alreadyStopped.length,
    status: stopped.length > 0 ? "stopped" : "not_running",
  };
}

function startProjectWorker({ projectId, slot, key, workerDir, childEnv }) {
  const python = resolveWorkerPython(workerDir, childEnv);
  console.log(`[worker:${projectId}:${slot}] starting with python=${python}`);
  appendDiagnosticLog(childEnv.MODEL_EXPRESS_LOG_DIR, "info", "worker_process_starting", {
    project_id: projectId,
    slot,
    worker_name: childEnv.WORKER_NAME,
    gpu_type: childEnv.GPU_TYPE,
    python,
  });
  console.log(
    `[worker:${projectId}:${slot}] modal env orchestrator=${childEnv.MODAL_ORCHESTRATOR_URL ?? "(unset)"} ` +
    `s3=${childEnv.MODAL_S3_ENDPOINT_URL ?? "(unset)"}`,
  );

  const child = spawn(python, ["-m", "worker.main"], {
    cwd: workerDir,
    env: childEnv,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });

  projectWorkers.set(key, child);

  child.stdout.on("data", (data) => {
    console.log(`[worker:${projectId}:${slot}] ${data.toString().trimEnd()}`);
    appendWorkerStreamLog(childEnv.MODEL_EXPRESS_LOG_DIR, "stdout", projectId, slot, data);
  });

  child.stderr.on("data", (data) => {
    console.error(`[worker:${projectId}:${slot}] ${data.toString().trimEnd()}`);
    appendWorkerStreamLog(childEnv.MODEL_EXPRESS_LOG_DIR, "stderr", projectId, slot, data);
  });

  child.on("exit", (code, signal) => {
    const current = projectWorkers.get(key);
    if (current === child) {
      projectWorkers.delete(key);
    }
    console.log(`[worker:${projectId}:${slot}] exited code=${code} signal=${signal}`);
    appendDiagnosticLog(childEnv.MODEL_EXPRESS_LOG_DIR, "warn", "worker_process_exited", {
      project_id: projectId,
      slot,
      code,
      signal,
    });
  });

  child.on("error", (error) => {
    const current = projectWorkers.get(key);
    if (current === child) {
      projectWorkers.delete(key);
    }
    console.error(`[worker:${projectId}:${slot}] failed to start: ${error.message}`);
    appendDiagnosticLog(childEnv.MODEL_EXPRESS_LOG_DIR, "error", "worker_process_start_failed", {
      project_id: projectId,
      slot,
      error: error.message,
    });
  });

  if (!child.pid) {
    throw new Error("Worker process did not start.");
  }

  return child;
}

function projectWorkerKey(projectId, slot, kind = "local") {
  return `${projectId}:${String(kind).trim().toLowerCase() || "local"}:${slot}`;
}

function isWorkerRunning(worker) {
  return worker && worker.exitCode === null && !worker.killed;
}

function shouldUseModalDispatcher(options) {
  const gpuType = String(options?.gpuType ?? "").trim().toLowerCase();
  if (gpuType !== "modal") {
    return false;
  }
  const disabled = String(process.env.MODEL_EXPRESS_DISABLE_MODAL_DISPATCHER ?? "").trim().toLowerCase();
  return !["1", "true", "yes", "on"].includes(disabled);
}

function normalizeWorkerCount(count) {
  const parsed = Number(count);
  if (!Number.isFinite(parsed)) {
    return 1;
  }

  return Math.max(1, Math.min(8, Math.trunc(parsed)));
}

function resolveWorkerPython(workerDir, env) {
  if (env.MODEL_EXPRESS_PYTHON) {
    return env.MODEL_EXPRESS_PYTHON;
  }

  const venvPython = process.platform === "win32"
    ? path.join(workerDir, ".venv", "Scripts", "python.exe")
    : path.join(workerDir, ".venv", "bin", "python");

  if (fs.existsSync(venvPython)) {
    return venvPython;
  }

  return process.platform === "win32" ? "python" : "python3";
}

function resolveLogDir(repoRoot) {
  return process.env.MODEL_EXPRESS_LOG_DIR ?? path.join(repoRoot, "artifacts", "logs");
}

function appendDiagnosticLog(logDir, level, event, fields = {}) {
  try {
    fs.mkdirSync(logDir, { recursive: true });
    const record = {
      ts: new Date().toISOString(),
      level,
      component: "mission-control",
      event,
      ...safeLogObject(fields),
    };
    fs.appendFileSync(path.join(logDir, "mission-control.jsonl"), `${JSON.stringify(record)}\n`, "utf8");
  } catch {
    // Diagnostics must never break the app.
  }
}

function appendWorkerStreamLog(logDir, stream, projectId, slot, data) {
  try {
    fs.mkdirSync(logDir, { recursive: true });
    const record = {
      ts: new Date().toISOString(),
      level: stream === "stderr" ? "error" : "info",
      component: "mission-control",
      event: `worker_${stream}`,
      project_id: projectId,
      slot,
      message: safeLogText(data.toString().trimEnd()),
    };
    fs.appendFileSync(path.join(logDir, "workers.jsonl"), `${JSON.stringify(record)}\n`, "utf8");
  } catch {
    // Diagnostics must never break the app.
  }
}

function safeLogObject(value) {
  const out = {};
  for (const [key, child] of Object.entries(value ?? {})) {
    out[key] = sensitiveLogKey(key) ? "[redacted]" : safeLogValue(child);
  }
  return out;
}

function safeLogValue(value) {
  if (value === null || value === undefined || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.slice(0, 24).map((item) => safeLogValue(item));
  }
  if (typeof value === "object") {
    return safeLogObject(value);
  }
  return safeLogText(String(value));
}

function safeLogText(value) {
  const redacted = String(value ?? "")
    .replace(/data:image\/[a-z0-9.+-]+;base64,[^\s"]+/gi, "[redacted]")
    .replace(/bearer\s+[a-z0-9._-]+/gi, "Bearer [redacted]")
    .replace(/(aws_access_key|api[_-]?key|secret|token|password)\s*[:=]\s*[^\s"]+/gi, "$1=[redacted]");
  return redacted.length > 1800 ? `${redacted.slice(0, 1799).trimEnd()}...` : redacted;
}

function sensitiveLogKey(key) {
  return /api[_-]?key|authorization|base64|credential|image|password|prompt|raw_output|secret|storage_uri|token|uri|url/i.test(key);
}

function loadRepoEnv(repoRoot) {
  const env = {};
  const envFile = process.env.MODEL_EXPRESS_ENV_FILE;
  const files = envFile
    ? [path.resolve(repoRoot, envFile)]
    : [path.join(repoRoot, ".env"), path.join(repoRoot, ".env.local")];

  for (const file of files) {
    if (!fs.existsSync(file)) {
      continue;
    }

    const contents = fs.readFileSync(file, "utf8");
    for (const rawLine of contents.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#")) {
        continue;
      }

      const separator = line.indexOf("=");
      if (separator === -1) {
        continue;
      }

      const key = line.slice(0, separator).trim();
      let value = line.slice(separator + 1).trim();
      if (!key) {
        continue;
      }

      if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1);
      }

      env[key] = value;
    }
  }

  return env;
}

async function uploadDatasetFolder(options) {
  const { projectId, datasetPath } = options;
  if (!projectId) {
    throw new Error("Project id is required before uploading a dataset.");
  }
  if (!datasetPath) {
    throw new Error("Dataset folder path is required.");
  }
  if (!fs.existsSync(datasetPath) || !fs.statSync(datasetPath).isDirectory()) {
    throw new Error(`Dataset folder does not exist: ${datasetPath}`);
  }

  const datasetName = path.basename(datasetPath);
  const safeName = datasetName.replace(/[^a-zA-Z0-9._-]/g, "-");
  const bucket = options.bucket ?? "model-express";
  const endpoint = options.endpoint ?? "http://localhost:9000";
  const accessKeyId = options.accessKeyId ?? "model_express";
  const secretAccessKey = options.secretAccessKey ?? "model_express_password";
  const region = options.region ?? "us-east-1";
  const key = `datasets/${projectId}/${safeName}.zip`;
  const entries = await collectDatasetFiles(datasetPath);
  if (entries.length === 0) {
    throw new Error(`Dataset folder does not contain any files: ${datasetPath}`);
  }
  const archivePlan = planZipArchive(entries);

  const client = new S3Client({
    endpoint,
    region,
    forcePathStyle: true,
    credentials: {
      accessKeyId,
      secretAccessKey,
    },
  });

  try {
    await client.send(new HeadBucketCommand({ Bucket: bucket }));
  } catch (_error) {
    await client.send(new CreateBucketCommand({ Bucket: bucket }));
  }

  const hash = crypto.createHash("sha256");
  const hashingStream = new Transform({
    transform(chunk, _encoding, callback) {
      hash.update(chunk);
      callback(null, chunk);
    },
  });
  const archiveStream = createZipArchiveStream(archivePlan).pipe(hashingStream);

  await client.send(
    new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: archiveStream,
      ContentLength: archivePlan.archiveSize,
      ContentType: "application/zip",
    }),
  );

  const checksum = hash.digest("hex");
  return {
    name: datasetName,
    storage_uri: `s3://${bucket}/${key}`,
    checksum_sha256: checksum,
    size_bytes: archivePlan.archiveSize,
  };
}
