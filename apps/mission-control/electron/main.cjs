let electron = {};
try {
  electron = require("electron");
} catch {
  electron = {};
}
const { app, BrowserWindow, dialog, ipcMain, Menu } = electron;
const { spawn } = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const net = require("net");
const os = require("os");
const path = require("path");
const { Transform } = require("stream");
const { fileURLToPath, pathToFileURL } = require("url");
const { CreateBucketCommand, GetObjectCommand, HeadBucketCommand, PutObjectCommand, S3Client } = require("@aws-sdk/client-s3");
const { buildDatasetUploadPreflight, collectDatasetFiles, createZipArchiveStream, planZipArchive } = require("./zip-stream.cjs");

let mainWindow;
const projectWorkers = new Map();
const projectTunnels = new Map();
const selectedDatasetFolders = new Map();

const DEFAULT_ORCHESTRATOR_URL = "http://127.0.0.1:8080";
const DEFAULT_S3_ENDPOINT_URL = "http://127.0.0.1:9000";
const ALLOWED_ORCHESTRATOR_METHODS = new Set(["GET", "POST", "PATCH", "DELETE"]);
const DEFAULT_JSON_BODY_MAX_BYTES = 2 * 1024 * 1024;
const DATASET_SELECTION_TTL_MS = 4 * 60 * 60 * 1000;
const CLOUDFLARED_URL_TIMEOUT_MS = 25_000;
const DEFAULT_REMOTE_TRAINING_SESSION_TTL_MS = 6 * 60 * 60 * 1000;
const MODEL_ARTIFACT_EXTENSIONS = [".onnx", ".ort", ".pt", ".pth", ".torchscript", ".safetensors"];
const electronRuntimeAvailable = Boolean(app && BrowserWindow && dialog && ipcMain && Menu);

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

function validateOrchestratorRequest(request = {}) {
  if (!request || typeof request !== "object" || Array.isArray(request)) {
    throw new Error("Orchestrator request must be an object.");
  }
  const method = validateOrchestratorMethod(request.method);
  const baseUrl = validateOrchestratorBaseUrl(request.baseUrl ?? DEFAULT_ORCHESTRATOR_URL);
  const requestPath = validateAppRequestPath(request.path);
  const bodyText = serializeJsonRequestBody(method, request.body);
  const url = new URL(requestPath, baseUrl).toString();
  return {
    method,
    baseUrl,
    path: requestPath,
    url,
    bodyText,
  };
}

function validateOrchestratorMethod(method = "GET") {
  const value = String(method ?? "GET").trim().toUpperCase();
  if (!ALLOWED_ORCHESTRATOR_METHODS.has(value)) {
    throw new Error(`Unsupported orchestrator request method: ${value || "(empty)"}`);
  }
  return value;
}

function validateAppRequestPath(value) {
  const text = String(value ?? "").trim();
  if (!text) {
    throw new Error("Orchestrator request path is required.");
  }
  if (!text.startsWith("/") || text.startsWith("//")) {
    throw new Error("Orchestrator request path must be an absolute app path.");
  }
  if (/^[a-z][a-z0-9+.-]*:/i.test(text) || /[\u0000-\u001f\\]/.test(text)) {
    throw new Error("Orchestrator request path must not include a protocol or control characters.");
  }
  const parsed = new URL(text, DEFAULT_ORCHESTRATOR_URL);
  if (parsed.hash) {
    throw new Error("Orchestrator request path must not include a URL fragment.");
  }
  const decodedParts = parsed.pathname.split("/").map((part) => safeDecodeURIComponent(part));
  if (decodedParts.some((part) => part === "..")) {
    throw new Error("Orchestrator request path must not contain parent directory segments.");
  }
  return `${parsed.pathname}${parsed.search}`;
}

function safeDecodeURIComponent(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function serializeJsonRequestBody(method, body) {
  if (body === undefined) {
    return undefined;
  }
  if (method === "GET") {
    throw new Error("GET orchestrator requests must not include a request body.");
  }
  const json = JSON.stringify(body);
  if (json === undefined) {
    throw new Error("Orchestrator request body must be JSON serializable.");
  }
  const maxBytes = positiveIntegerEnv("MODEL_EXPRESS_IPC_JSON_MAX_BYTES", DEFAULT_JSON_BODY_MAX_BYTES);
  if (Buffer.byteLength(json, "utf8") > maxBytes) {
    throw new Error(`Orchestrator request body exceeds ${maxBytes} bytes.`);
  }
  return json;
}

function validateOrchestratorBaseUrl(value, env = process.env) {
  const parsed = parseHttpOrigin(value || DEFAULT_ORCHESTRATOR_URL, "Orchestrator URL");
  if (isLoopbackHostname(parsed.hostname)) {
    return parsed.origin;
  }
  if (allowedOriginsFromEnv(["MODEL_EXPRESS_ALLOWED_ORCHESTRATOR_ORIGINS"], env).has(parsed.origin)) {
    return parsed.origin;
  }
  const tunnelOrigins = envFlagFrom(env, "MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE", false)
    ? allowedOriginsFromValues([env.MODAL_ORCHESTRATOR_URL, env.MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL])
    : new Set();
  if (tunnelOrigins.has(parsed.origin)) {
    return parsed.origin;
  }
  throw new Error("Non-loopback orchestrator URLs require MODEL_EXPRESS_ALLOWED_ORCHESTRATOR_ORIGINS.");
}

function parseHttpOrigin(value, label) {
  let parsed;
  try {
    parsed = new URL(String(value ?? "").trim());
  } catch {
    throw new Error(`${label} must be a valid HTTP(S) URL.`);
  }
  if (!["http:", "https:"].includes(parsed.protocol)) {
    throw new Error(`${label} must use http or https.`);
  }
  if (parsed.username || parsed.password) {
    throw new Error(`${label} must not contain embedded credentials.`);
  }
  if ((parsed.pathname && parsed.pathname !== "/") || parsed.search || parsed.hash) {
    throw new Error(`${label} must be an origin without a path, query, or fragment.`);
  }
  return parsed;
}

function isLoopbackHostname(hostname) {
  const host = String(hostname ?? "").trim().toLowerCase().replace(/^\[|\]$/g, "");
  return (
    host === "localhost" ||
    host === "ip6-localhost" ||
    host === "::1" ||
    host === "0:0:0:0:0:0:0:1" ||
    /^127(?:\.\d{1,3}){3}$/.test(host)
  );
}

function isPrivateOrLocalHostname(hostname) {
  const host = String(hostname ?? "").trim().toLowerCase().replace(/^\[|\]$/g, "");
  if (isLoopbackHostname(host) || host === "0.0.0.0" || host === "::") {
    return true;
  }
  if (net.isIP(host) === 4) {
    const parts = host.split(".").map((part) => Number.parseInt(part, 10));
    const [a, b] = parts;
    return (
      a === 10 ||
      (a === 172 && b >= 16 && b <= 31) ||
      (a === 192 && b === 168) ||
      (a === 169 && b === 254) ||
      (a === 100 && b >= 64 && b <= 127)
    );
  }
  if (net.isIP(host) === 6) {
    return host === "::" || host.startsWith("fc") || host.startsWith("fd") || host.startsWith("fe80:");
  }
  return host === "localhost" || host.endsWith(".localhost");
}

function allowedOriginsFromEnv(names, env = process.env) {
  return allowedOriginsFromValues(names.flatMap((name) => splitEnvList(env[name])));
}

function allowedOriginsFromValues(values) {
  const origins = new Set();
  for (const value of values) {
    const normalized = normalizeOrigin(value);
    if (normalized) {
      origins.add(normalized);
    }
  }
  return origins;
}

function normalizeOrigin(value) {
  const text = String(value ?? "").trim();
  if (!text) {
    return "";
  }
  try {
    return parseHttpOrigin(text, "Allowed origin").origin;
  } catch {
    return "";
  }
}

function splitEnvList(value) {
  return String(value ?? "")
    .split(/[,\s;]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function envFlagFrom(env, name, fallback = false) {
  const value = String(env?.[name] ?? "").trim().toLowerCase();
  if (!value) {
    return fallback;
  }
  return ["1", "true", "yes", "on"].includes(value);
}

function apiTokenHeaders(env = process.env) {
  const token = String(env.MODEL_EXPRESS_API_TOKEN ?? "").trim();
  if (!token) {
    return {};
  }
  return {
    Authorization: `Bearer ${token}`,
    "X-Model-Express-API-Token": token,
  };
}

function requireAuthenticatedOrchestratorExposure(env = process.env) {
  const token = String(env.MODEL_EXPRESS_API_TOKEN ?? "").trim();
  if (!token || !envFlagFrom(env, "MODEL_EXPRESS_ALLOW_LAN", false)) {
    throw new Error(
      "Modal orchestrator tunnels require MODEL_EXPRESS_ALLOW_LAN=true and MODEL_EXPRESS_API_TOKEN so the public callback URL is authenticated.",
    );
  }
}

function rememberDatasetFolder(datasetPath) {
  const folder = validateDatasetDirectory(datasetPath);
  pruneDatasetSelections();
  const token = crypto.randomUUID();
  selectedDatasetFolders.set(token, {
    token,
    path: folder.path,
    name: folder.name,
    createdAt: Date.now(),
  });
  return {
    token,
    path: folder.path,
    name: folder.name,
  };
}

function resolveDatasetFolderOption(options = {}) {
  const token = String(options?.datasetToken ?? options?.token ?? "").trim();
  if (token) {
    pruneDatasetSelections();
    const selected = selectedDatasetFolders.get(token);
    if (!selected) {
      throw new Error("Dataset folder selection has expired. Choose the folder again.");
    }
    return validateDatasetDirectory(selected.path);
  }

  const datasetPath = String(options?.datasetPath ?? "").trim();
  if (datasetPath) {
    const folder = validateDatasetDirectory(datasetPath);
    for (const selected of selectedDatasetFolders.values()) {
      if (pathKey(selected.path) === pathKey(folder.path)) {
        return folder;
      }
    }
  }

  throw new Error("Dataset folder must be selected in Mission Control before preflight or upload.");
}

function validateDatasetDirectory(datasetPath) {
  const raw = String(datasetPath ?? "").trim();
  if (!raw) {
    throw new Error("Dataset folder path is required.");
  }
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(raw)) {
    throw new Error("Dataset folder must be a local folder selected with the picker.");
  }
  const realPath = fs.realpathSync.native(raw);
  const stat = fs.statSync(realPath);
  if (!stat.isDirectory()) {
    throw new Error(`Dataset folder is not a directory: ${realPath}`);
  }
  if (isDangerousDatasetDirectory(realPath)) {
    throw new Error("Refusing to upload a filesystem root, home folder, or system directory as a dataset.");
  }
  return {
    path: realPath,
    name: path.basename(realPath),
  };
}

function isDangerousDatasetDirectory(datasetPath) {
  const root = path.parse(datasetPath).root;
  if (pathKey(datasetPath) === pathKey(root)) {
    return true;
  }
  const home = safeRealpath(os.homedir());
  if (home && pathKey(datasetPath) === pathKey(home)) {
    return true;
  }
  const systemRoots = [
    process.env.SystemRoot,
    process.env.WINDIR,
    process.env.ProgramFiles,
    process.env["ProgramFiles(x86)"],
  ].map((item) => safeRealpath(item)).filter(Boolean);
  return systemRoots.some((systemRoot) => isPathInsideOrEqual(datasetPath, systemRoot));
}

function pruneDatasetSelections(now = Date.now()) {
  for (const [token, selected] of selectedDatasetFolders.entries()) {
    if (!selected?.createdAt || now - selected.createdAt > DATASET_SELECTION_TTL_MS) {
      selectedDatasetFolders.delete(token);
    }
  }
}

function validateUploadEndpoint(value, env = process.env) {
  const configuredEndpoint = env.S3_ENDPOINT_URL || DEFAULT_S3_ENDPOINT_URL;
  const endpoint = parseHttpOrigin(value || configuredEndpoint, "Dataset upload endpoint");
  if (endpoint.port === "9001") {
    throw new Error("Dataset upload endpoint must use the MinIO API port, not the console port.");
  }
  const configuredOrigin = normalizeOrigin(configuredEndpoint);
  if (isLoopbackHostname(endpoint.hostname) || endpoint.origin === configuredOrigin) {
    return endpoint.origin;
  }
  if (allowedOriginsFromEnv(["MODEL_EXPRESS_ALLOWED_UPLOAD_ORIGINS", "MODEL_EXPRESS_ALLOWED_STORAGE_ORIGINS"], env).has(endpoint.origin)) {
    return endpoint.origin;
  }
  throw new Error("Remote dataset upload endpoints require MODEL_EXPRESS_ALLOWED_UPLOAD_ORIGINS.");
}

function resolveS3Endpoint(options = {}, purpose = "storage", env = process.env) {
  const configuredEndpoint = env.S3_ENDPOINT_URL || DEFAULT_S3_ENDPOINT_URL;
  const configuredOrigin = normalizeOrigin(configuredEndpoint);
  const provided = String(options?.endpoint ?? "").trim();
  const endpoint = parseHttpOrigin(provided || configuredEndpoint, "S3 endpoint");
  if (endpoint.port === "9001") {
    throw new Error("S3 endpoint must use the MinIO API port, not the console port.");
  }
  if (!provided || endpoint.origin === configuredOrigin) {
    return endpoint.origin;
  }
  if (purpose === "artifact") {
    if (allowedOriginsFromEnv(["MODEL_EXPRESS_ALLOWED_STORAGE_ORIGINS"], env).has(endpoint.origin)) {
      return endpoint.origin;
    }
    throw new Error("S3 artifact loading must use the configured S3 endpoint or MODEL_EXPRESS_ALLOWED_STORAGE_ORIGINS.");
  }
  if (isLoopbackHostname(endpoint.hostname) || allowedOriginsFromEnv(["MODEL_EXPRESS_ALLOWED_STORAGE_ORIGINS"], env).has(endpoint.origin)) {
    return endpoint.origin;
  }
  throw new Error("Remote S3 endpoints require MODEL_EXPRESS_ALLOWED_STORAGE_ORIGINS.");
}

function repoRoot(env = process.env) {
  return path.resolve(env.MODEL_EXPRESS_ROOT || path.resolve(__dirname, "..", "..", ".."));
}

function workerRoot(root = repoRoot()) {
  return path.join(root, "services", "worker");
}

function configuredLocalArtifactRoots(env = process.env) {
  const root = repoRoot(env);
  const worker = workerRoot(root);
  const roots = [
    path.join(root, "artifacts"),
    path.join(root, "exports"),
    path.join(worker, ".cache", "exports"),
    path.join(worker, ".cache", "champion_exports"),
    path.join(worker, ".cache", "artifacts"),
  ];

  for (const name of ["WORKER_EXPORT_ROOT", "WORKER_CHAMPION_EXPORT_ROOT", "WORKER_ARTIFACT_DOWNLOAD_ROOT"]) {
    const value = String(env[name] ?? "").trim();
    if (value) {
      roots.push(path.isAbsolute(value) ? value : path.resolve(worker, value));
    }
  }

  for (const value of [
    ...splitPathList(env.MODEL_EXPRESS_ALLOWED_ARTIFACT_ROOTS),
    ...splitPathList(env.MODEL_EXPRESS_ALLOWED_EXPORT_ROOTS),
  ]) {
    roots.push(path.isAbsolute(value) ? value : path.resolve(root, value));
  }

  const unique = [];
  const seen = new Set();
  for (const item of roots) {
    const normalized = safeRealpath(item) || path.resolve(item);
    const key = pathKey(normalized);
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(normalized);
    }
  }
  return unique;
}

function validateLocalArtifactPath(artifactUri, env = process.env) {
  const localPath = localPathFromURI(String(artifactUri ?? "").trim());
  if (!localPath) {
    throw new Error(`Unsupported model artifact URI: ${artifactUri}`);
  }
  const candidate = path.isAbsolute(localPath) ? localPath : path.resolve(repoRoot(env), localPath);
  if (!fs.existsSync(candidate) || !fs.statSync(candidate).isFile()) {
    throw new Error(`Model artifact does not exist: ${redactPathForError(candidate)}`);
  }
  const realCandidate = fs.realpathSync.native(candidate);
  const allowed = configuredLocalArtifactRoots(env).some((rootPath) => isPathInsideOrEqual(realCandidate, rootPath));
  if (!allowed) {
    throw new Error("Model artifact must be under a configured Model Express artifact or export root.");
  }
  if (isPathInsideOrEqual(realCandidate, path.join(repoRoot(env), "artifacts", "logs"))) {
    throw new Error("Model artifact loading refuses log directories.");
  }
  if (!isAllowedModelArtifactFile(realCandidate)) {
    throw new Error("Model artifact must use a supported model artifact extension.");
  }
  return realCandidate;
}

function isAllowedModelArtifactFile(filePath) {
  const name = path.basename(String(filePath ?? "")).toLowerCase();
  return MODEL_ARTIFACT_EXTENSIONS.some((extension) => name.endsWith(extension));
}

function validateExternalDataLocalPath(sourcePath, artifactDir, mountPath, explicit) {
  const resolvedSource = path.resolve(sourcePath);
  const resolvedArtifactDir = safeRealpath(artifactDir) || path.resolve(artifactDir);
  if (!isPathInsideOrEqual(resolvedSource, resolvedArtifactDir)) {
    if (explicit) {
      throw new Error(`ONNX external data file must stay next to the artifact: ${mountPath}`);
    }
    return "";
  }
  if (!fs.existsSync(resolvedSource) || !fs.statSync(resolvedSource).isFile()) {
    if (explicit) {
      throw new Error(`Unable to load ONNX external data file: ${mountPath}`);
    }
    return "";
  }
  const realSource = fs.realpathSync.native(resolvedSource);
  if (!isPathInsideOrEqual(realSource, resolvedArtifactDir)) {
    throw new Error(`ONNX external data file must stay next to the artifact: ${mountPath}`);
  }
  return realSource;
}

function splitPathList(value) {
  return String(value ?? "")
    .split(/[;\n\r]+/)
    .flatMap((item) => item.split(process.platform === "win32" ? "\n" : ":"))
    .map((item) => item.trim())
    .filter(Boolean);
}

function isPathInsideOrEqual(candidate, rootPath) {
  const resolvedCandidate = path.resolve(candidate);
  const resolvedRoot = path.resolve(rootPath);
  const relative = path.relative(resolvedRoot, resolvedCandidate);
  return relative === "" || (!!relative && !relative.startsWith("..") && !path.isAbsolute(relative));
}

function pathKey(value) {
  const resolved = path.resolve(value);
  return process.platform === "win32" ? resolved.toLowerCase() : resolved;
}

function safeRealpath(value) {
  if (!value) {
    return "";
  }
  try {
    return fs.realpathSync.native(value);
  } catch {
    return "";
  }
}

function redactPathForError(value) {
  const text = String(value ?? "");
  const root = repoRoot();
  if (isPathInsideOrEqual(text, root)) {
    return path.relative(root, text) || ".";
  }
  return path.basename(text) || "[path]";
}

function redactUrlForLog(value) {
  const text = String(value ?? "");
  try {
    const parsed = new URL(text);
    if (!isLoopbackHostname(parsed.hostname) || /trycloudflare\.com$/i.test(parsed.hostname)) {
      return `${parsed.protocol}//[redacted-host]`;
    }
    parsed.search = parsed.search ? "?[redacted]" : "";
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return safeLogText(text);
  }
}

if (electronRuntimeAvailable) {
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
    stopAllProjectTunnels();
  });

  ipcMain.handle("orchestrator:request", async (_event, request) => {
    const validated = validateOrchestratorRequest(request);
    const response = await fetch(validated.url, {
      method: validated.method,
      headers: {
        "Content-Type": "application/json",
        ...apiTokenHeaders(),
      },
      body: validated.bodyText,
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
        path: validated.path,
        url: redactUrlForLog(validated.url),
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

    const folder = rememberDatasetFolder(result.filePaths[0]);
    return uploadDatasetFolder({
      ...options,
      datasetToken: folder.token,
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

    return rememberDatasetFolder(result.filePaths[0]);
  });

  ipcMain.handle("dataset:uploadFolder", async (_event, options) => {
    return uploadDatasetFolder(options);
  });

  ipcMain.handle("dataset:preflightFolder", async (_event, options) => {
    const { datasetPath } = resolveDatasetFolderOption(options);
    const entries = await collectDatasetFiles(datasetPath);
    const archivePlan = planZipArchive(entries);
    return datasetUploadPreflight(entries, archivePlan, options);
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
      ? await readS3ObjectBuffer(artifactUri, request, "artifact")
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
}

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
  const artifactPath = validateLocalArtifactPath(artifactUri);
  return fs.readFileSync(artifactPath);
}

function readLocalExternalDataFiles(artifactUri, request = {}) {
  const artifactPath = validateLocalArtifactPath(artifactUri);
  const artifactDir = path.dirname(artifactPath);
  const out = [];
  const seen = new Set();
  for (const candidate of externalDataCandidates(artifactUri, request.externalData)) {
    const mountPath = externalDataMountPath(candidate.path);
    if (!mountPath || seen.has(mountPath)) {
      continue;
    }
    const sourcePath = externalDataLocalPath(candidate, artifactDir, mountPath);
    if (!sourcePath) {
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
      if (candidate.explicit) {
        throw new Error(`ONNX external data file must stay next to the artifact: ${mountPath}`);
      }
      continue;
    }
    try {
      const buffer = await readS3ObjectBuffer(sidecarUri, request, "artifact");
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
  let selectedPath = "";
  for (const key of ["uri", "artifact_uri", "artifactPath", "artifact_path", "localPath", "local_path"]) {
    const value = String(candidate[key] ?? "").trim();
    if (!value) {
      continue;
    }
    const localPath = localPathFromURI(value);
    if (!localPath) {
      continue;
    }
    selectedPath = path.isAbsolute(localPath) ? localPath : path.resolve(artifactDir, localPath);
    break;
  }
  const sourcePath = selectedPath || path.resolve(artifactDir, storageRelativePath(mountPath));
  return validateExternalDataLocalPath(sourcePath, artifactDir, mountPath, candidate.explicit);
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
  const parsed = new URL(artifactUri);
  const baseDir = path.posix.dirname(parsed.pathname.replace(/^\/+/, ""));
  const key = path.posix.join(baseDir === "." ? "" : baseDir, storageRelativePath(mountPath));
  const expectedUri = `s3://${parsed.hostname}/${key}`;
  for (const key of ["uri", "artifact_uri", "artifactPath", "artifact_path"]) {
    const value = String(candidate[key] ?? "").trim();
    if (value.startsWith("s3://")) {
      const explicit = normalizeS3ObjectURI(value);
      return explicit === expectedUri ? value : "";
    }
  }
  return expectedUri;
}

function normalizeS3ObjectURI(value) {
  try {
    const parsed = new URL(value);
    const key = storageRelativePath(parsed.pathname.replace(/^\/+/, ""));
    return `s3://${parsed.hostname}/${key}`;
  } catch {
    return "";
  }
}

function bufferToArrayBuffer(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

async function readS3ObjectBuffer(artifactUri, options = {}, purpose = "storage") {
  const parsed = new URL(artifactUri);
  const bucket = parsed.hostname;
  const key = parsed.pathname.replace(/^\/+/, "");
  if (!bucket || !key) {
    throw new Error(`Invalid S3 model artifact URI: ${artifactUri}`);
  }
  const client = createS3Client(options, { purpose });
  const response = await client.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
  return streamToBuffer(response.Body);
}

function createS3Client(options = {}, { purpose = "storage" } = {}) {
  return new S3Client({
    endpoint: resolveS3Endpoint(options, purpose),
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

async function ensureProjectWorker(options = {}) {
  const projectId = String(options.projectId ?? "").trim();
  if (!projectId) {
    throw new Error("Project id is required before starting a worker.");
  }
  const baseUrl = validateOrchestratorBaseUrl(options.baseUrl || DEFAULT_ORCHESTRATOR_URL);

  const repoRoot = process.env.MODEL_EXPRESS_ROOT ?? path.resolve(__dirname, "..", "..", "..");
  const workerDir = path.join(repoRoot, "services", "worker");
  if (!fs.existsSync(workerDir)) {
    throw new Error(`Worker directory does not exist: ${workerDir}`);
  }

  const repoEnv = loadRepoEnv(repoRoot);
  const logDir = resolveLogDir(repoRoot);
  fs.mkdirSync(logDir, { recursive: true });
  const targetCount = normalizeWorkerCount(options.count);
  const useModalDispatcher = shouldUseModalDispatcher(options);
  const processCount = useModalDispatcher ? 1 : targetCount;
  const pids = [];
  let startedCount = 0;
  let modalSession = null;

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

    if (useModalDispatcher && !modalSession) {
      modalSession = await ensureRemoteTrainingSession({
        projectId,
        baseUrl,
        repoEnv,
        logDir,
      });
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
      Object.assign(childEnv, modalSession?.env ?? {});
    }

    let child;
    try {
      child = startProjectWorker({
        projectId,
        slot,
        key,
        workerDir,
        childEnv,
        tunnelSession: modalSession,
      });
    } catch (error) {
      if (useModalDispatcher) {
        stopProjectTunnels(projectId);
      }
      throw error;
    }

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
  stopProjectTunnels(projectId);
  return {
    project_id: projectId,
    stopped_count: stopped.length,
    stopped_pids: stopped,
    already_stopped_count: alreadyStopped.length,
    status: stopped.length > 0 ? "stopped" : "not_running",
  };
}

async function ensureRemoteTrainingSession({ projectId, baseUrl, repoEnv = {}, logDir }) {
  const existing = projectTunnels.get(projectId);
  if (existing && remoteTrainingSessionActive(existing)) {
    return existing;
  }
  stopProjectTunnels(projectId);

  const env = { ...process.env, ...repoEnv };
  const childEnv = {};
  const processes = [];

  const configuredOrchestrator = firstNonEmpty(env.MODAL_ORCHESTRATOR_URL, env.MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL);
  if (configuredOrchestrator) {
    requireAuthenticatedOrchestratorExposure(env);
    childEnv.MODAL_ORCHESTRATOR_URL = validateRemoteModalUrl(configuredOrchestrator, "MODAL_ORCHESTRATOR_URL");
  } else if (!isLoopbackHostname(new URL(baseUrl).hostname)) {
    requireAuthenticatedOrchestratorExposure(env);
    childEnv.MODAL_ORCHESTRATOR_URL = validateRemoteModalUrl(baseUrl, "MODAL_ORCHESTRATOR_URL");
  } else {
    requireAuthenticatedOrchestratorExposure(env);
    const orchestratorTunnel = await startCloudflaredTunnel({
      projectId,
      label: "orchestrator",
      targetUrl: baseUrl,
      logDir,
      env,
    });
    childEnv.MODAL_ORCHESTRATOR_URL = orchestratorTunnel.url;
    processes.push(orchestratorTunnel.child);
  }

  const configuredModalS3 = firstNonEmpty(env.MODAL_S3_ENDPOINT_URL, env.MODEL_EXPRESS_MODAL_S3_ENDPOINT_URL);
  if (configuredModalS3) {
    childEnv.MODAL_S3_ENDPOINT_URL = validateRemoteModalUrl(configuredModalS3, "MODAL_S3_ENDPOINT_URL");
  } else {
    const configuredS3Endpoint = firstNonEmpty(env.S3_ENDPOINT_URL, "");
    const s3Endpoint = configuredS3Endpoint ? parseHttpOrigin(configuredS3Endpoint, "S3 endpoint") : null;
    if (s3Endpoint && !isLoopbackHostname(s3Endpoint.hostname)) {
      if (s3Endpoint.port === "9001") {
        throw new Error("MODAL_S3_ENDPOINT_URL must not point at the MinIO console port.");
      }
      childEnv.MODAL_S3_ENDPOINT_URL = s3Endpoint.origin;
    } else if (envFlagFrom(env, "MODEL_EXPRESS_MODAL_TUNNEL_S3", false) || envFlagFrom(env, "MODEL_EXPRESS_ALLOW_MODAL_MINIO_TUNNEL", false)) {
      const targetUrl = s3Endpoint ? s3Endpoint.origin : DEFAULT_S3_ENDPOINT_URL;
      validateCloudflaredTarget(targetUrl, "s3");
      const s3Tunnel = await startCloudflaredTunnel({
        projectId,
        label: "s3",
        targetUrl,
        logDir,
        env,
      });
      childEnv.MODAL_S3_ENDPOINT_URL = s3Tunnel.url;
      processes.push(s3Tunnel.child);
    } else {
      throw new Error(
        "Modal workers require MODAL_S3_ENDPOINT_URL, a remote S3_ENDPOINT_URL, or MODEL_EXPRESS_MODAL_TUNNEL_S3=1 for local MinIO API tunneling.",
      );
    }
  }

  const session = {
    projectId,
    env: childEnv,
    processes,
    createdAt: Date.now(),
    expiresAt: Date.now() + remoteTrainingSessionTtlMs(env),
    ttlTimer: null,
  };
  session.ttlTimer = setTimeout(() => {
    appendDiagnosticLog(logDir, "info", "remote_training_session_expired", {
      project_id: projectId,
      managed_tunnel_count: processes.length,
    });
    stopProjectTunnels(projectId);
  }, Math.max(1, session.expiresAt - Date.now()));
  if (typeof session.ttlTimer.unref === "function") {
    session.ttlTimer.unref();
  }
  projectTunnels.set(projectId, session);
  appendDiagnosticLog(logDir, "info", "remote_training_session_ready", {
    project_id: projectId,
    orchestrator_url: childEnv.MODAL_ORCHESTRATOR_URL,
    s3_endpoint_url: childEnv.MODAL_S3_ENDPOINT_URL,
    managed_tunnel_count: processes.length,
    expires_at: new Date(session.expiresAt).toISOString(),
  });
  return session;
}

function remoteTrainingSessionActive(session) {
  return Boolean(
    session &&
    (!Number.isFinite(session.expiresAt) || Date.now() < session.expiresAt) &&
    Array.isArray(session.processes) &&
    session.processes.every((child) => isWorkerRunning(child))
  );
}

function remoteTrainingSessionTtlMs(env = process.env) {
  const seconds = Number.parseInt(String(env.MODEL_EXPRESS_REMOTE_TRAINING_SESSION_TTL_SECONDS ?? ""), 10);
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return DEFAULT_REMOTE_TRAINING_SESSION_TTL_MS;
  }
  const ttlMs = seconds * 1000;
  return Math.max(5 * 60 * 1000, Math.min(ttlMs, 24 * 60 * 60 * 1000));
}

function firstNonEmpty(...values) {
  return values.map((value) => String(value ?? "").trim()).find(Boolean) || "";
}

function validateRemoteModalUrl(value, label, env = process.env) {
  const parsed = parseHttpOrigin(value, label);
  if (parsed.protocol !== "https:" && !envFlagFrom(env, "MODEL_EXPRESS_ALLOW_INSECURE_MODAL_URLS", false)) {
    throw new Error(`${label} must use https for Modal workers.`);
  }
  if (isPrivateOrLocalHostname(parsed.hostname)) {
    throw new Error(`${label} must be remotely reachable for Modal workers.`);
  }
  if (parsed.port === "5432" || parsed.port === "5000" || parsed.port === "9001") {
    throw new Error(`${label} must not expose Postgres, MLflow, or the MinIO console.`);
  }
  return parsed.origin;
}

function validateCloudflaredTarget(value, label) {
  const parsed = parseHttpOrigin(value, `${label} tunnel target`);
  if (!isLoopbackHostname(parsed.hostname)) {
    throw new Error(`${label} tunnel target must be loopback.`);
  }
  if (parsed.port === "5432" || parsed.port === "5000" || parsed.port === "9001") {
    throw new Error(`${label} tunnel target must not be Postgres, MLflow, or the MinIO console.`);
  }
  return parsed.origin;
}

function startCloudflaredTunnel({ projectId, label, targetUrl, logDir, env = process.env }) {
  if (envFlagFrom(env, "MODEL_EXPRESS_DISABLE_AUTO_TUNNELS", false)) {
    throw new Error("Automatic tunnel management is disabled by MODEL_EXPRESS_DISABLE_AUTO_TUNNELS.");
  }
  const target = validateCloudflaredTarget(targetUrl, label);
  const command = env.MODEL_EXPRESS_CLOUDFLARED_PATH || env.CLOUDFLARED_PATH || "cloudflared";
  const child = spawn(command, ["tunnel", "--url", target, "--no-autoupdate"], {
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });

  return new Promise((resolve, reject) => {
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      stopTunnelProcess(child);
      reject(new Error(`Timed out waiting for cloudflared ${label} tunnel URL.`));
    }, CLOUDFLARED_URL_TIMEOUT_MS);

    const finish = (url) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      appendDiagnosticLog(logDir, "info", "cloudflared_tunnel_ready", {
        project_id: projectId,
        label,
        url,
      });
      resolve({ url, child });
    };

    const onData = (streamName, data) => {
      const text = data.toString();
      const safeText = safeLogText(text.trimEnd());
      if (safeText) {
        console.log(`[tunnel:${projectId}:${label}:${streamName}] ${safeText}`);
      }
      const url = parseCloudflaredTunnelUrl(text);
      if (url) {
        finish(url);
      }
    };

    child.stdout.on("data", (data) => onData("stdout", data));
    child.stderr.on("data", (data) => onData("stderr", data));
    child.on("error", (error) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(new Error(`Unable to start cloudflared for ${label}: ${error.message}`));
      }
    });
    child.on("exit", (code, signal) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(new Error(`cloudflared ${label} tunnel exited before publishing a URL (code=${code}, signal=${signal}).`));
      }
    });
  });
}

function parseCloudflaredTunnelUrl(value) {
  const match = String(value ?? "").match(/https:\/\/[a-z0-9-]+\.trycloudflare\.com\b/i);
  return match ? match[0] : "";
}

function stopAllProjectTunnels() {
  for (const projectId of Array.from(projectTunnels.keys())) {
    stopProjectTunnels(projectId);
  }
}

function stopProjectTunnels(projectId) {
  const session = projectTunnels.get(projectId);
  if (!session) {
    return;
  }
  for (const child of session.processes ?? []) {
    stopTunnelProcess(child);
  }
  if (session.ttlTimer) {
    clearTimeout(session.ttlTimer);
  }
  projectTunnels.delete(projectId);
}

function stopTunnelProcess(child) {
  if (child && child.exitCode === null && !child.killed) {
    child.kill();
  }
}

function startProjectWorker({ projectId, slot, key, workerDir, childEnv, tunnelSession }) {
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
    `[worker:${projectId}:${slot}] modal env orchestrator=${redactUrlForLog(childEnv.MODAL_ORCHESTRATOR_URL ?? "(unset)")} ` +
    `s3=${redactUrlForLog(childEnv.MODAL_S3_ENDPOINT_URL ?? "(unset)")}`,
  );

  const child = spawn(python, ["-m", "worker.main"], {
    cwd: workerDir,
    env: childEnv,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });

  projectWorkers.set(key, child);

  child.stdout.on("data", (data) => {
    console.log(`[worker:${projectId}:${slot}] ${safeLogText(data.toString().trimEnd())}`);
    appendWorkerStreamLog(childEnv.MODEL_EXPRESS_LOG_DIR, "stdout", projectId, slot, data);
  });

  child.stderr.on("data", (data) => {
    console.error(`[worker:${projectId}:${slot}] ${safeLogText(data.toString().trimEnd())}`);
    appendWorkerStreamLog(childEnv.MODEL_EXPRESS_LOG_DIR, "stderr", projectId, slot, data);
  });

  child.on("exit", (code, signal) => {
    const current = projectWorkers.get(key);
    if (current === child) {
      projectWorkers.delete(key);
    }
    if (tunnelSession) {
      stopProjectTunnels(projectId);
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
    if (tunnelSession) {
      stopProjectTunnels(projectId);
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
    const logPath = path.join(logDir, "mission-control.jsonl");
    rotateJsonlLog(logPath);
    fs.appendFileSync(logPath, `${JSON.stringify(record)}\n`, "utf8");
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
    const logPath = path.join(logDir, "workers.jsonl");
    rotateJsonlLog(logPath);
    fs.appendFileSync(logPath, `${JSON.stringify(record)}\n`, "utf8");
  } catch {
    // Diagnostics must never break the app.
  }
}

function rotateJsonlLog(logPath) {
  const maxBytes = positiveIntegerEnv("MODEL_EXPRESS_LOG_MAX_BYTES", 10 * 1024 * 1024);
  const maxFiles = positiveIntegerEnv("MODEL_EXPRESS_LOG_MAX_FILES", 5);
  try {
    if (!fs.existsSync(logPath) || fs.statSync(logPath).size < maxBytes) {
      return;
    }
    for (let index = maxFiles - 1; index >= 1; index -= 1) {
      const source = `${logPath}.${index}`;
      const destination = `${logPath}.${index + 1}`;
      if (!fs.existsSync(source)) {
        continue;
      }
      if (fs.existsSync(destination)) {
        fs.rmSync(destination, { force: true });
      }
      fs.renameSync(source, destination);
    }
    const first = `${logPath}.1`;
    if (fs.existsSync(first)) {
      fs.rmSync(first, { force: true });
    }
    fs.renameSync(logPath, first);
  } catch {
    // Rotation is best-effort.
  }
}

function positiveIntegerEnv(name, fallback) {
  const parsed = Number.parseInt(String(process.env[name] ?? ""), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
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
    .replace(/https?:\/\/[^\s"]*trycloudflare\.com[^\s"]*/gi, "[redacted-url]")
    .replace(/([?&](?:x-amz-credential|x-amz-signature|x-amz-security-token|awsaccesskeyid|signature|token|secret|authorization)=)[^&\s"]+/gi, "$1[redacted]")
    .replace(/https?:\/\/[^\s"]*(?:token|secret|signature|x-amz-signature|x-amz-credential|authorization)=[^\s"]*/gi, "[redacted-url]")
    .replace(/\bsk-[a-z0-9_-]{10,}\b/gi, "[redacted-token]")
    .replace(/bearer\s+[a-z0-9._-]+/gi, "Bearer [redacted]")
    .replace(/\b((?:model_express_)?modal_[a-z0-9_]*url|cloudflared_url|tunnel_url)\s*[:=]\s*https?:\/\/[^\s"]+/gi, "$1=[redacted-url]")
    .replace(/(aws[_-]?access[_-]?key(?:[_-]?id)?|aws[_-]?secret[_-]?access[_-]?key|accessKeyId|secretAccessKey|api[_-]?key|secret|token|password)\s*[:=]\s*[^\s"]+/gi, "$1=[redacted]");
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
  const { projectId } = options;
  if (!projectId) {
    throw new Error("Project id is required before uploading a dataset.");
  }
  const { datasetPath } = resolveDatasetFolderOption(options);

  const datasetName = path.basename(datasetPath);
  const safeName = datasetName.replace(/[^a-zA-Z0-9._-]/g, "-");
  const bucket = options.bucket ?? "model-express";
  const endpoint = validateUploadEndpoint(options.endpoint);
  const accessKeyId = options.accessKeyId ?? "model_express";
  const secretAccessKey = options.secretAccessKey ?? "model_express_password";
  const region = options.region ?? "us-east-1";
  const key = `datasets/${projectId}/${safeName}.zip`;
  const entries = await collectDatasetFiles(datasetPath);
  if (entries.length === 0) {
    throw new Error(`Dataset folder does not contain any files: ${datasetPath}`);
  }
  const archivePlan = planZipArchive(entries);
  const preflight = datasetUploadPreflight(entries, archivePlan, options);
  if (preflight.errors.length > 0) {
    throw new Error(preflight.errors.map((item) => item.message).join(" "));
  }
  if (preflight.warnings.length > 0) {
    appendDiagnosticLog(resolveLogDir(process.env.MODEL_EXPRESS_ROOT ?? path.resolve(__dirname, "..", "..", "..")), "warn", "dataset_upload_preflight_warning", {
      project_id: projectId,
      file_count: preflight.file_count,
      uncompressed_size_bytes: preflight.uncompressed_size_bytes,
      archive_size_bytes: preflight.archive_size_bytes,
      warnings: preflight.warnings,
    });
  }

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
    file_count: preflight.file_count,
    uncompressed_size_bytes: preflight.uncompressed_size_bytes,
    upload_warnings: preflight.warnings,
  };
}

function datasetUploadPreflight(entries, archivePlan, options = {}) {
  return buildDatasetUploadPreflight(entries, archivePlan, {
    warnFileCount: options.warnFileCount ?? process.env.MODEL_EXPRESS_UPLOAD_WARN_FILE_COUNT,
    warnBytes: options.warnBytes ?? process.env.MODEL_EXPRESS_UPLOAD_WARN_BYTES,
    maxFileCount: options.maxFileCount ?? process.env.MODEL_EXPRESS_UPLOAD_MAX_FILE_COUNT,
    maxBytes: options.maxBytes ?? process.env.MODEL_EXPRESS_UPLOAD_MAX_BYTES,
  });
}

module.exports = {
  __test: {
    apiTokenHeaders,
    configuredLocalArtifactRoots,
    externalDataMountPath,
    isLoopbackHostname,
    parseCloudflaredTunnelUrl,
    rememberDatasetFolder,
    requireAuthenticatedOrchestratorExposure,
    remoteTrainingSessionActive,
    remoteTrainingSessionTtlMs,
    resolveDatasetFolderOption,
    resolveS3Endpoint,
    safeLogText,
    selectedDatasetFolders,
    validateAppRequestPath,
    validateLocalArtifactPath,
    validateOrchestratorBaseUrl,
    validateOrchestratorRequest,
    validateRemoteModalUrl,
    validateUploadEndpoint,
  },
};
