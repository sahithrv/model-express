const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const { spawn } = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");
const AdmZip = require("adm-zip");
const { CreateBucketCommand, HeadBucketCommand, PutObjectCommand, S3Client } = require("@aws-sdk/client-s3");

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
  const payload = text ? JSON.parse(text) : null;

  if (!response.ok) {
    const message = payload && payload.error ? payload.error : response.statusText;
    throw new Error(message);
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

  const existing = projectWorkers.get(projectId);
  if (existing && existing.exitCode === null && !existing.killed) {
    return {
      project_id: projectId,
      pid: existing.pid,
      started: false,
      status: "already_running",
    };
  }

  const repoRoot = process.env.MODEL_EXPRESS_ROOT ?? path.resolve(__dirname, "..", "..", "..");
  const workerDir = path.join(repoRoot, "services", "worker");
  if (!fs.existsSync(workerDir)) {
    throw new Error(`Worker directory does not exist: ${workerDir}`);
  }

  const python = process.env.MODEL_EXPRESS_PYTHON ?? (process.platform === "win32" ? "python" : "python3");
  const workerName = options.name ?? `profile-worker-${projectId}`;
  const child = spawn(python, ["-m", "worker.main"], {
    cwd: workerDir,
    env: {
      ...process.env,
      ORCHESTRATOR_URL: baseUrl,
      PROJECT_ID: projectId,
      WORKER_NAME: workerName,
      GPU_TYPE: options.gpuType ?? "local",
      PYTHONUNBUFFERED: "1",
    },
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });

  projectWorkers.set(projectId, child);

  child.stdout.on("data", (data) => {
    console.log(`[worker:${projectId}] ${data.toString().trimEnd()}`);
  });

  child.stderr.on("data", (data) => {
    console.error(`[worker:${projectId}] ${data.toString().trimEnd()}`);
  });

  child.on("exit", (code, signal) => {
    const current = projectWorkers.get(projectId);
    if (current === child) {
      projectWorkers.delete(projectId);
    }
    console.log(`[worker:${projectId}] exited code=${code} signal=${signal}`);
  });

  child.on("error", (error) => {
    const current = projectWorkers.get(projectId);
    if (current === child) {
      projectWorkers.delete(projectId);
    }
    console.error(`[worker:${projectId}] failed to start: ${error.message}`);
  });

  if (!child.pid) {
    throw new Error("Worker process did not start.");
  }

  return {
    project_id: projectId,
    pid: child.pid,
    started: true,
    status: "started",
  };
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
  const tempDir = path.join(os.tmpdir(), "model-express");
  fs.mkdirSync(tempDir, { recursive: true });

  const archivePath = path.join(tempDir, `${safeName}-${Date.now()}.zip`);
  const zip = new AdmZip();
  zip.addLocalFolder(datasetPath);
  zip.writeZip(archivePath);

  const checksum = sha256File(archivePath);
  const sizeBytes = fs.statSync(archivePath).size;

  const bucket = options.bucket ?? "model-express";
  const endpoint = options.endpoint ?? "http://localhost:9000";
  const accessKeyId = options.accessKeyId ?? "model_express";
  const secretAccessKey = options.secretAccessKey ?? "model_express_password";
  const region = options.region ?? "us-east-1";
  const key = `datasets/${projectId}/${safeName}.zip`;

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

  await client.send(
    new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: fs.createReadStream(archivePath),
      ContentType: "application/zip",
      Metadata: {
        "checksum-sha256": checksum,
      },
    }),
  );

  return {
    name: datasetName,
    storage_uri: `s3://${bucket}/${key}`,
    checksum_sha256: checksum,
    size_bytes: sizeBytes,
  };
}

function sha256File(filePath) {
  const hash = crypto.createHash("sha256");
  const buffer = fs.readFileSync(filePath);
  hash.update(buffer);
  return hash.digest("hex");
}
