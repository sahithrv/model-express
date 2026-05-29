const { app, BrowserWindow, dialog, ipcMain, Menu } = require("electron");
const { spawn } = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const { Transform } = require("stream");
const { CreateBucketCommand, HeadBucketCommand, PutObjectCommand, S3Client } = require("@aws-sdk/client-s3");
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
    throw new Error(`${response.status} ${message}`);
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

ipcMain.handle("worker:ensureProjectWorker", async (_event, options) => {
  return ensureProjectWorker(options);
});

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
  const repoEnv = loadRepoEnv(repoRoot);
  const pids = [];
  let startedCount = 0;

  for (let slot = 1; slot <= targetCount; slot += 1) {
    const key = projectWorkerKey(projectId, slot);
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
    running_count: pids.length,
    status: startedCount > 0 ? "started" : "already_running",
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

function projectWorkerKey(projectId, slot) {
  return `${projectId}:${slot}`;
}

function isWorkerRunning(worker) {
  return worker && worker.exitCode === null && !worker.killed;
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
