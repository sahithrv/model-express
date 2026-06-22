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
const { pipeline } = require("stream/promises");
const { fileURLToPath, pathToFileURL } = require("url");
const { CreateBucketCommand, DeleteObjectCommand, GetObjectCommand, HeadBucketCommand, PutObjectCommand, S3Client } = require("@aws-sdk/client-s3");
const { buildDatasetUploadPreflight, collectDatasetFiles, createZipArchiveStream, planZipArchive } = require("./zip-stream.cjs");

let mainWindow;
const projectWorkers = new Map();
const projectTunnels = new Map();
const selectedDatasetFolders = new Map();

const DEFAULT_ORCHESTRATOR_URL = "http://127.0.0.1:8080";
const DEFAULT_S3_ENDPOINT_URL = "http://127.0.0.1:9000";
const DEFAULT_S3_BUCKET = "model-express";
const DEFAULT_ARTIFACT_PREFIX = "model-express/artifacts";
const DEFAULT_DATABASE_URL = "postgres://model_express:model_express@127.0.0.1:5432/model_express?sslmode=disable";
const ALLOWED_ORCHESTRATOR_METHODS = new Set(["GET", "POST", "PATCH", "DELETE"]);
const DEFAULT_JSON_BODY_MAX_BYTES = 2 * 1024 * 1024;
const DATASET_SELECTION_TTL_MS = 4 * 60 * 60 * 1000;
const CLOUDFLARED_URL_TIMEOUT_MS = 25_000;
const DEFAULT_REMOTE_TRAINING_SESSION_TTL_MS = 6 * 60 * 60 * 1000;
const LOCAL_RUNTIME_BOOTSTRAP_TIMEOUT_MS = 90_000;
const MINIO_BOOTSTRAP_ATTEMPTS = 40;
const MINIO_BOOTSTRAP_RETRY_MS = 750;
const LOCAL_CONFIG_FILE = "model-express.local.json";
const MODEL_ARTIFACT_EXTENSIONS = [".onnx", ".ort", ".pt", ".pth", ".torchscript", ".safetensors"];
const PORTABLE_BUNDLE_EXTENSIONS = [".zip"];
const EXPORT_ARTIFACT_EXTENSIONS = [".zip", ".onnx", ".pt", ".pth", ".torchscript", ".safetensors", ".json"];
const electronRuntimeAvailable = Boolean(app && BrowserWindow && dialog && ipcMain && Menu);
let missionControlEnvCache = null;
let missionControlEnvCacheKey = "";
let localRuntimeBootstrap = null;
let championDemoRuntime = null;
const CHAMPION_DEMO_RUNTIME_IDLE_TTL_MS = 3 * 60 * 1000;
const CHAMPION_DEMO_RUNTIME_TIMEOUT_MS = 45_000;

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

function validateOrchestratorRequest(request = {}, env = process.env) {
  if (!request || typeof request !== "object" || Array.isArray(request)) {
    throw new Error("Orchestrator request must be an object.");
  }
  const method = validateOrchestratorMethod(request.method);
  const baseUrl = validateOrchestratorBaseUrl(request.baseUrl ?? DEFAULT_ORCHESTRATOR_URL, env);
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

function missionControlUserDataDir(env = process.env, runtime = {}) {
  const configured = String(env.MODEL_EXPRESS_USER_DATA_DIR ?? runtime.userDataDir ?? "").trim();
  if (configured) {
    return path.resolve(configured);
  }
  return defaultModelExpressUserDataDir(env);
}

function defaultModelExpressUserDataDir(env = process.env) {
  const home = os.homedir();
  if (process.platform === "win32") {
    return path.join(env.APPDATA || path.join(home, "AppData", "Roaming"), "Model Express");
  }
  if (process.platform === "darwin") {
    return path.join(home, "Library", "Application Support", "Model Express");
  }
  return path.join(env.XDG_CONFIG_HOME || path.join(home, ".config"), "model-express");
}

function localConfigPath(env = process.env, runtime = {}) {
  return path.join(missionControlUserDataDir(env, runtime), LOCAL_CONFIG_FILE);
}

function readLocalConfig(env = process.env, runtime = {}) {
  try {
    const file = localConfigPath(env, runtime);
    if (!fs.existsSync(file)) {
      return {};
    }
    const parsed = JSON.parse(fs.readFileSync(file, "utf8"));
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function writeLocalConfig(config, env = process.env, runtime = {}) {
  const file = localConfigPath(env, runtime);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tempFile = `${file}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tempFile, `${JSON.stringify(config, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
  fs.renameSync(tempFile, file);
}

function writeNewLocalConfig(config, env = process.env, runtime = {}) {
  const file = localConfigPath(env, runtime);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  let handle;
  try {
    handle = fs.openSync(file, "wx", 0o600);
    fs.writeFileSync(handle, `${JSON.stringify(config, null, 2)}\n`, "utf8");
    return true;
  } catch (error) {
    if (error && error.code === "EEXIST") {
      return false;
    }
    throw error;
  } finally {
    if (handle !== undefined) {
      fs.closeSync(handle);
    }
  }
}

function generateLocalApiToken() {
  return crypto.randomBytes(32).toString("base64url");
}

function ensureLocalApiTokenEnv(env = process.env, runtime = {}) {
  const explicit = String(env.MODEL_EXPRESS_API_TOKEN ?? "").trim();
  if (explicit) {
    return env;
  }
  const { apiToken: token } = ensureLocalRuntimeConfig(env, runtime);
  return {
    ...env,
    MODEL_EXPRESS_API_TOKEN: token,
  };
}

function ensureLocalRuntimeConfig(env = process.env, runtime = {}) {
  const configFile = localConfigPath(env, runtime);
  const configExists = fs.existsSync(configFile);
  const config = readLocalConfig(env, runtime);
  const nextConfig = { ...config };
  let changed = false;

  let apiToken = String(env.MODEL_EXPRESS_API_TOKEN ?? "").trim() || String(nextConfig.model_express_api_token ?? "").trim();
  if (!apiToken) {
    apiToken = generateLocalApiToken();
    nextConfig.model_express_api_token = apiToken;
    changed = true;
  } else if (!String(nextConfig.model_express_api_token ?? "").trim() && !String(env.MODEL_EXPRESS_API_TOKEN ?? "").trim()) {
    nextConfig.model_express_api_token = apiToken;
    changed = true;
  }

  let minioAccessKey = String(nextConfig.minio_access_key ?? "").trim();
  if (!minioAccessKey) {
    minioAccessKey = generateMinioAccessKey();
    nextConfig.minio_access_key = minioAccessKey;
    changed = true;
  }

  let minioSecretKey = String(nextConfig.minio_secret_key ?? "").trim();
  if (!minioSecretKey) {
    minioSecretKey = generateRuntimeSecret();
    nextConfig.minio_secret_key = minioSecretKey;
    changed = true;
  }

  let bucket = String(nextConfig.s3_bucket ?? "").trim();
  if (!bucket) {
    bucket = DEFAULT_S3_BUCKET;
    nextConfig.s3_bucket = bucket;
    changed = true;
  }

  let artifactPrefix = normalizeArtifactPrefix(nextConfig.artifact_prefix);
  if (!artifactPrefix) {
    artifactPrefix = DEFAULT_ARTIFACT_PREFIX;
    nextConfig.artifact_prefix = artifactPrefix;
    changed = true;
  } else if (artifactPrefix !== nextConfig.artifact_prefix) {
    nextConfig.artifact_prefix = artifactPrefix;
    changed = true;
  }

  if (changed) {
    nextConfig.updated_at = new Date().toISOString();
    if (!configExists && !writeNewLocalConfig(nextConfig, env, runtime)) {
      return ensureLocalRuntimeConfig(env, runtime);
    }
    if (configExists) {
      writeLocalConfig(nextConfig, env, runtime);
    }
  }

  return {
    apiToken,
    minioAccessKey,
    minioSecretKey,
    bucket,
    artifactPrefix,
    configPath: configFile,
  };
}

function generateMinioAccessKey() {
  return `mx${crypto.randomBytes(16).toString("hex")}`;
}

function generateRuntimeSecret() {
  return crypto.randomBytes(32).toString("base64url");
}

function normalizeArtifactPrefix(value) {
  return String(value ?? "")
    .trim()
    .replace(/^\/+|\/+$/g, "")
    .replace(/\/{2,}/g, "/");
}

function appManagedLocalRuntimeEnabled(env = process.env) {
  if (envFlagFrom(env, "MODEL_EXPRESS_DISABLE_APP_LOCAL_RUNTIME", false)) {
    return false;
  }
  const configured = String(env.MODEL_EXPRESS_APP_MANAGED_LOCAL_RUNTIME ?? "").trim().toLowerCase();
  if (configured) {
    return ["1", "true", "yes", "on"].includes(configured);
  }
  const endpoint = String(env.S3_ENDPOINT_URL ?? "").trim();
  if (!endpoint) {
    return true;
  }
  try {
    return isLoopbackHostname(parseHttpOrigin(endpoint, "S3 endpoint").hostname);
  } catch {
    return false;
  }
}

function appLocalRuntimeEnv(env = process.env, runtime = {}) {
  const tokenEnv = ensureLocalApiTokenEnv(env, runtime);
  if (!appManagedLocalRuntimeEnabled(tokenEnv)) {
    return tokenEnv;
  }
  const config = ensureLocalRuntimeConfig(tokenEnv, runtime);
  return {
    ...tokenEnv,
    MODEL_EXPRESS_APP_MANAGED_LOCAL_RUNTIME: "true",
    MODEL_EXPRESS_API_TOKEN: tokenEnv.MODEL_EXPRESS_API_TOKEN || config.apiToken,
    MODEL_EXPRESS_V1_PROFILE: tokenEnv.MODEL_EXPRESS_V1_PROFILE || "cloud",
    MODEL_EXPRESS_EXECUTION_PROFILE: tokenEnv.MODEL_EXPRESS_EXECUTION_PROFILE || "fast-remote",
    MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER: tokenEnv.MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER || "modal",
    MODEL_EXPRESS_DEFAULT_GPU_TYPE: tokenEnv.MODEL_EXPRESS_DEFAULT_GPU_TYPE || "T4",
    MODEL_EXPRESS_MODAL_DEFAULT_GPU_TYPE: tokenEnv.MODEL_EXPRESS_MODAL_DEFAULT_GPU_TYPE || tokenEnv.MODEL_EXPRESS_DEFAULT_GPU_TYPE || "T4",
    MODEL_EXPRESS_ORCHESTRATOR_ADDR: tokenEnv.MODEL_EXPRESS_ORCHESTRATOR_ADDR || "127.0.0.1:8080",
    MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE: tokenEnv.MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE || "true",
    DATABASE_URL: tokenEnv.DATABASE_URL || DEFAULT_DATABASE_URL,
    S3_ENDPOINT_URL: DEFAULT_S3_ENDPOINT_URL,
    S3_BUCKET: config.bucket,
    MODEL_EXPRESS_ARTIFACT_BUCKET: config.bucket,
    MODEL_EXPRESS_ARTIFACT_PREFIX: config.artifactPrefix,
    AWS_ACCESS_KEY_ID: config.minioAccessKey,
    AWS_SECRET_ACCESS_KEY: config.minioSecretKey,
    MODEL_EXPRESS_MODAL_AWS_ACCESS_KEY_ID: config.minioAccessKey,
    MODEL_EXPRESS_MODAL_AWS_SECRET_ACCESS_KEY: config.minioSecretKey,
    MINIO_ROOT_USER: config.minioAccessKey,
    MINIO_ROOT_PASSWORD: config.minioSecretKey,
    AWS_DEFAULT_REGION: tokenEnv.AWS_DEFAULT_REGION || "us-east-1",
    MODEL_EXPRESS_MODAL_TUNNEL_S3: "true",
    // RC local-only: Modal receives generated local MinIO root credentials through a short-lived tunnel.
    MODEL_EXPRESS_ALLOW_MODAL_ROOT_STORAGE: "true",
  };
}

async function ensureAppLocalRuntime(runtime = {}) {
  const env = runtime.env ?? missionControlEnv(process.env, runtime);
  if (!appManagedLocalRuntimeEnabled(env)) {
    return { managed: false, env };
  }
  if (!localRuntimeBootstrap || runtime.force) {
    localRuntimeBootstrap = bootstrapAppLocalRuntime({ ...runtime, env }).catch((error) => {
      localRuntimeBootstrap = null;
      throw error;
    });
  }
  return localRuntimeBootstrap;
}

async function bootstrapAppLocalRuntime(runtime = {}) {
  const env = appLocalRuntimeEnv(runtime.env ?? process.env, runtime);
  const root = repoRoot(env);
  const logDir = resolveLogDir(root, env);
  await startDockerComposeLocalServices({ ...runtime, env, repoRoot: root, logDir });
  await waitForMinioBucket({ ...runtime, env });
  appendDiagnosticLog(logDir, "info", "local_runtime_ready", {
    services: ["postgres", "minio"],
    s3_endpoint_url: env.S3_ENDPOINT_URL,
    s3_bucket: env.S3_BUCKET,
    artifact_prefix: env.MODEL_EXPRESS_ARTIFACT_PREFIX,
  });
  return {
    managed: true,
    env,
    services: ["postgres", "minio"],
    s3_endpoint_url: env.S3_ENDPOINT_URL,
    s3_bucket: env.S3_BUCKET,
    artifact_prefix: env.MODEL_EXPRESS_ARTIFACT_PREFIX,
  };
}

async function startDockerComposeLocalServices({ env = process.env, repoRoot: root = repoRoot(env), logDir, spawnFn = spawn, timeoutMs = LOCAL_RUNTIME_BOOTSTRAP_TIMEOUT_MS } = {}) {
  const spec = dockerComposeLocalServicesSpec(root, env);
  try {
    await runChildProcess(spec.command, spec.args, {
      cwd: root,
      env: spec.env,
      spawnFn,
      timeoutMs,
      label: "docker compose",
      logDir,
    });
  } catch (error) {
    const friendly = dockerRuntimeError(error);
    friendly.localRuntimeStage = "docker_compose";
    throw friendly;
  }
  return spec;
}

function dockerComposeLocalServicesSpec(root = repoRoot(), env = process.env) {
  return {
    command: env.MODEL_EXPRESS_DOCKER_PATH || "docker",
    args: ["compose", "-f", path.join(root, "infra", "compose.yaml"), "up", "-d", "postgres", "minio"],
    env: {
      ...env,
      MINIO_ROOT_USER: env.MINIO_ROOT_USER || env.AWS_ACCESS_KEY_ID || "model_express_dev",
      MINIO_ROOT_PASSWORD: env.MINIO_ROOT_PASSWORD || env.AWS_SECRET_ACCESS_KEY || "model_express_dev_password",
    },
  };
}

function runChildProcess(command, args, { cwd, env, spawnFn = spawn, timeoutMs = 30_000, label = command, logDir } = {}) {
  return new Promise((resolve, reject) => {
    let child;
    try {
      child = spawnFn(command, args, {
        cwd,
        env,
        windowsHide: true,
        stdio: ["ignore", "pipe", "pipe"],
      });
    } catch (error) {
      reject(error);
      return;
    }

    let stdout = "";
    let stderr = "";
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      if (child && child.exitCode === null && !child.killed) {
        child.kill();
      }
      const error = new Error(`${label} timed out.`);
      error.stdout = stdout;
      error.stderr = stderr;
      reject(error);
    }, timeoutMs);

    const onOutput = (streamName, data) => {
      const text = data.toString();
      if (streamName === "stdout") {
        stdout += text;
      } else {
        stderr += text;
      }
      appendDiagnosticLog(logDir, "info", `${label.replace(/\s+/g, "_")}_${streamName}`, {
        message: safeLogText(text.trimEnd()),
      });
    };

    child.stdout?.on?.("data", (data) => onOutput("stdout", data));
    child.stderr?.on?.("data", (data) => onOutput("stderr", data));
    child.on("error", (error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      error.stdout = stdout;
      error.stderr = stderr;
      reject(error);
    });
    child.on("exit", (code, signal) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      if (code === 0) {
        resolve({ stdout, stderr, code, signal });
        return;
      }
      const error = new Error(`${label} exited with code ${code ?? "unknown"}${signal ? ` signal ${signal}` : ""}.`);
      error.stdout = stdout;
      error.stderr = stderr;
      error.code = code;
      error.signal = signal;
      reject(error);
    });
  });
}

function dockerRuntimeError(error) {
  const output = `${error?.message ?? ""}\n${error?.stderr ?? ""}\n${error?.stdout ?? ""}`;
  const remediation = "Start Docker Desktop.";
  if (error?.code === "ENOENT" || /not recognized|not found|no such file/i.test(output)) {
    return new Error(`Docker Desktop is required. ${remediation}`);
  }
  if (/daemon|docker engine|docker desktop|pipe|npipe|connection refused|cannot connect|is the docker daemon running/i.test(output)) {
    return new Error(`Docker is not running. ${remediation}`);
  }
  return new Error(`Local runtime bootstrap failed while starting Docker Compose. ${remediation}`);
}

async function waitForMinioBucket({ env = process.env, client, attempts = MINIO_BOOTSTRAP_ATTEMPTS, retryMs = MINIO_BOOTSTRAP_RETRY_MS, sleepFn = sleep } = {}) {
  const bucket = String(env.S3_BUCKET || env.MODEL_EXPRESS_ARTIFACT_BUCKET || DEFAULT_S3_BUCKET).trim();
  const s3Client = client || createS3ClientFromEnv(env);
  let lastError = null;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      await ensureS3Bucket(s3Client, bucket);
      return { bucket, attempts: attempt };
    } catch (error) {
      lastError = error;
      if (attempt < attempts) {
        await sleepFn(retryMs);
      }
    }
  }
  const friendly = new Error(`Local MinIO is not reachable at ${env.S3_ENDPOINT_URL || DEFAULT_S3_ENDPOINT_URL}. Start Docker Desktop.`);
  friendly.cause = lastError;
  friendly.localRuntimeStage = "minio_api";
  throw friendly;
}

async function ensureS3Bucket(client, bucket) {
  try {
    await client.send(new HeadBucketCommand({ Bucket: bucket }));
    return;
  } catch (error) {
    if (!isBucketMissingError(error) && !isPossiblyMissingBucketError(error)) {
      throw error;
    }
  }
  await client.send(new CreateBucketCommand({ Bucket: bucket }));
  await client.send(new HeadBucketCommand({ Bucket: bucket }));
}

function isBucketMissingError(error) {
  const statusCode = error?.$metadata?.httpStatusCode;
  const name = String(error?.name ?? "");
  const code = String(error?.Code ?? error?.code ?? "");
  return statusCode === 404 || /NoSuchBucket|NotFound/i.test(`${name} ${code}`);
}

function isPossiblyMissingBucketError(error) {
  const statusCode = error?.$metadata?.httpStatusCode;
  return statusCode === 301 || statusCode === 403;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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
  const exposureAuthEnabled =
    envFlagFrom(env, "MODEL_EXPRESS_ALLOW_LAN", false) ||
    envFlagFrom(env, "MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE", false);
  if (!token || !exposureAuthEnabled) {
    throw new Error(
      "Modal orchestrator tunnels require MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE=true or MODEL_EXPRESS_ALLOW_LAN=true and MODEL_EXPRESS_API_TOKEN so the public callback URL is authenticated.",
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

function missionControlEnv(env = process.env, runtime = {}) {
  const root = repoRoot(env);
  const envFile = String(env.MODEL_EXPRESS_ENV_FILE ?? "");
  const cacheKey = `${root}\0${envFile}`;
  if (missionControlEnvCacheKey !== cacheKey) {
    missionControlEnvCache = loadRepoEnv(root, env);
    missionControlEnvCacheKey = cacheKey;
  }
  return appLocalRuntimeEnv({
    ...env,
    ...(missionControlEnvCache ?? {}),
  }, runtime);
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
  return validateLocalArtifactPathForExtensions(
    artifactUri,
    env,
    MODEL_ARTIFACT_EXTENSIONS,
    "Model artifact",
    "Model artifact must use a supported model artifact extension.",
  );
}

function validateLocalPortableBundlePath(artifactUri, env = process.env) {
  return validateLocalArtifactPathForExtensions(
    artifactUri,
    env,
    PORTABLE_BUNDLE_EXTENSIONS,
    "Export bundle",
    "Export bundle must use a supported archive extension.",
  );
}

function validateLocalExportArtifactPath(
  artifactUri,
  env = process.env,
  allowedExtensions = EXPORT_ARTIFACT_EXTENSIONS,
  extensionMessage = "Export artifact must use a supported download extension.",
) {
  rejectParentTraversalSegments(artifactUri, "Export artifact");
  return validateLocalArtifactPathForExtensions(
    artifactUri,
    env,
    allowedExtensions,
    "Export artifact",
    extensionMessage,
  );
}

function validateLocalArtifactPathForExtensions(artifactUri, env, allowedExtensions, label, extensionMessage) {
  const localPath = localPathFromURI(String(artifactUri ?? "").trim());
  if (!localPath) {
    throw new Error(`Unsupported ${label.toLowerCase()} URI: ${artifactUri}`);
  }
  const candidate = path.isAbsolute(localPath) ? localPath : path.resolve(repoRoot(env), localPath);
  if (!fs.existsSync(candidate) || !fs.statSync(candidate).isFile()) {
    throw new Error(`${label} does not exist: ${redactPathForError(candidate)}`);
  }
  const realCandidate = fs.realpathSync.native(candidate);
  const allowed = configuredLocalArtifactRoots(env).some((rootPath) => isPathInsideOrEqual(realCandidate, rootPath));
  if (!allowed) {
    throw new Error(`${label} must be under a configured Model Express artifact or export root.`);
  }
  if (isPathInsideOrEqual(realCandidate, path.join(repoRoot(env), "artifacts", "logs"))) {
    throw new Error(`${label} loading refuses log directories.`);
  }
  if (!isAllowedArtifactFile(realCandidate, allowedExtensions)) {
    throw new Error(extensionMessage);
  }
  return realCandidate;
}

function isAllowedModelArtifactFile(filePath) {
  return isAllowedArtifactFile(filePath, MODEL_ARTIFACT_EXTENSIONS);
}

function isAllowedPortableBundleFile(filePath) {
  return isAllowedArtifactFile(filePath, PORTABLE_BUNDLE_EXTENSIONS);
}

function isAllowedArtifactFile(filePath, allowedExtensions) {
  const name = path.basename(String(filePath ?? "")).toLowerCase();
  return allowedExtensions.some((extension) => name.endsWith(extension));
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
    stopChampionDemoRuntime({ reason: "app_before_quit" });
  });

  ipcMain.handle("orchestrator:request", async (_event, request) => {
    const env = missionControlEnv();
    const validated = validateOrchestratorRequest(request, env);
    const response = await fetch(validated.url, {
      method: validated.method,
      headers: {
        "Content-Type": "application/json",
        ...apiTokenHeaders(env),
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
    return preflightDatasetFolder(options);
  });

  ipcMain.handle("cloud:preflight", async (_event, options) => {
    return preflightCloud(options);
  });

  ipcMain.handle("demo:predictChampionLocal", async (_event, request) => {
    return predictChampionDemoLocal(request);
  });

  ipcMain.handle("demo:disposeChampionLocalRuntime", async (_event, request) => {
    return stopChampionDemoRuntime({ ...(request ?? {}), reason: request?.reason || "ipc_dispose" });
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

  ipcMain.handle("artifact:save", async (_event, request) => {
    return saveArtifact(request);
  });

  ipcMain.handle("artifact:saveExport", async (_event, request) => {
    return saveExportArtifact(request);
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

async function saveArtifact(request = {}) {
  const artifactUri = String(request?.artifactUri ?? "").trim();
  if (!artifactUri) {
    throw new Error("artifactUri is required.");
  }
  validatePortableBundleArtifactUri(artifactUri);
  const result = await saveExportArtifact(
    {
      ...request,
      suggestedName: request.suggestedName ?? request.defaultPath,
      kind: request.kind ?? "portable_bundle",
    },
    {},
  );
  if (result.canceled) {
    return null;
  }
  return {
    artifact_uri: result.artifact_uri,
    path: result.file_path,
    size_bytes: result.bytes,
  };
}

async function saveExportArtifact(request = {}, runtime = {}) {
  const source = resolveExportArtifactSource(request, runtime.env ?? process.env);
  const showSaveDialog = runtime.showSaveDialog ?? dialog?.showSaveDialog?.bind(dialog);
  if (!showSaveDialog) {
    throw new Error("Save dialog is unavailable.");
  }
  const suggestedName = safeDownloadFileName(
    request.suggestedName ||
      request.defaultPath ||
      artifactFileName(source.artifact_uri) ||
      defaultExportArtifactName(request.kind),
  );
  const dialogResult = await showSaveDialog(mainWindow, {
    title: source.kind === "portable_bundle" ? "Save portable inference bundle" : "Save export artifact",
    defaultPath: suggestedName,
    filters: exportArtifactDialogFilters(source.extension),
  });
  if (dialogResult.canceled || !dialogResult.filePath) {
    return {
      canceled: true,
      file_path: "",
      bytes: 0,
      artifact_uri: source.artifact_uri,
    };
  }
  const bytes = source.s3
    ? await copyS3ObjectToFile(source.artifact_uri, dialogResult.filePath, request)
    : await copyLocalFileToFile(source.file_path, dialogResult.filePath);
  return {
    canceled: false,
    file_path: dialogResult.filePath,
    bytes,
    artifact_uri: source.artifact_uri,
  };
}

function validatePortableBundleArtifactUri(artifactUri) {
  if (artifactUri.startsWith("s3://")) {
    const parsed = new URL(artifactUri);
    const key = parsed.pathname.replace(/^\/+/, "");
    if (!parsed.hostname || !key) {
      throw new Error(`Invalid S3 export bundle URI: ${artifactUri}`);
    }
    if (!isAllowedPortableBundleFile(key)) {
      throw new Error("Export bundle must use a supported archive extension.");
    }
    return;
  }
  validateLocalPortableBundlePath(artifactUri);
}

function readLocalPortableBundleBuffer(artifactUri) {
  const artifactPath = validateLocalPortableBundlePath(artifactUri);
  return fs.readFileSync(artifactPath);
}

function resolveExportArtifactSource(request = {}, env = process.env) {
  const artifactUri = String(request?.artifactUri ?? "").trim();
  const artifactPath = String(request?.artifactPath ?? request?.artifact_path ?? "").trim();
  const kind = String(request.kind ?? "");
  const allowedExtensions = exportArtifactExtensionsForKind(kind);
  const extensionMessage = exportArtifactExtensionMessage(kind);
  const sourceUri = artifactUri || artifactPath;
  if (!sourceUri) {
    throw new Error("artifactUri or artifactPath is required.");
  }
  if (/^https?:\/\//i.test(sourceUri)) {
    throw new Error("HTTP export artifact downloads are not supported by Mission Control.");
  }
  if (sourceUri.startsWith("s3://")) {
    validateS3ExportArtifactUri(sourceUri, allowedExtensions, extensionMessage);
    return {
      artifact_uri: sourceUri,
      file_path: "",
      extension: exportArtifactExtension(sourceUri, allowedExtensions),
      kind,
      s3: true,
    };
  }
  const localSource = artifactPath && !artifactUri.startsWith("file://") && !artifactUri.startsWith("s3://")
    ? artifactPath
    : sourceUri;
  const filePath = validateLocalExportArtifactPath(localSource, env, allowedExtensions, extensionMessage);
  return {
    artifact_uri: artifactUri || pathToFileURL(filePath).toString(),
    file_path: filePath,
    extension: exportArtifactExtension(filePath, allowedExtensions),
    kind,
    s3: false,
  };
}

function validateS3ExportArtifactUri(
  artifactUri,
  allowedExtensions = EXPORT_ARTIFACT_EXTENSIONS,
  extensionMessage = "Export artifact must use a supported download extension.",
) {
  let parsed;
  try {
    parsed = new URL(artifactUri);
  } catch {
    throw new Error(`Invalid S3 export artifact URI: ${artifactUri}`);
  }
  const key = parsed.pathname.replace(/^\/+/, "");
  if (!parsed.hostname || !key) {
    throw new Error(`Invalid S3 export artifact URI: ${artifactUri}`);
  }
  rejectParentTraversalSegments(key, "Export artifact");
  if (!isAllowedExportArtifactFile(key, allowedExtensions)) {
    throw new Error(extensionMessage);
  }
}

function isAllowedExportArtifactFile(filePath, allowedExtensions = EXPORT_ARTIFACT_EXTENSIONS) {
  return isAllowedArtifactFile(filePath, allowedExtensions);
}

function exportArtifactExtension(filePath, allowedExtensions = EXPORT_ARTIFACT_EXTENSIONS) {
  const comparable = artifactComparablePathForExtension(filePath);
  const extension = allowedExtensions.find((item) => comparable.endsWith(item));
  return extension || path.extname(comparable);
}

function exportArtifactExtensionsForKind(kind) {
  return String(kind ?? "").toLowerCase().includes("portable")
    ? PORTABLE_BUNDLE_EXTENSIONS
    : EXPORT_ARTIFACT_EXTENSIONS;
}

function exportArtifactExtensionMessage(kind) {
  return String(kind ?? "").toLowerCase().includes("portable")
    ? "Portable export bundle must use a supported ZIP extension."
    : "Export artifact must use a supported download extension.";
}

function artifactComparablePathForExtension(value) {
  const text = String(value ?? "").trim();
  if (text.startsWith("s3://") || text.startsWith("file://")) {
    try {
      return decodeURIComponent(new URL(text).pathname).toLowerCase();
    } catch {
      return text.toLowerCase();
    }
  }
  return text.toLowerCase();
}

function defaultExportArtifactName(kind) {
  return String(kind ?? "").toLowerCase().includes("portable") ? "portable_inference_bundle.zip" : "model-export-artifact.zip";
}

function exportArtifactDialogFilters(extension) {
  if (extension === ".zip") {
    return [
      { name: "ZIP archives", extensions: ["zip"] },
      { name: "All supported export artifacts", extensions: EXPORT_ARTIFACT_EXTENSIONS.map((item) => item.slice(1)) },
    ];
  }
  return [
    { name: "All supported export artifacts", extensions: EXPORT_ARTIFACT_EXTENSIONS.map((item) => item.slice(1)) },
    { name: "All files", extensions: ["*"] },
  ];
}

async function copyLocalFileToFile(sourcePath, destinationPath) {
  const source = fs.realpathSync.native(sourcePath);
  const destination = path.resolve(destinationPath);
  if (pathKey(source) === pathKey(destination)) {
    return fs.statSync(source).size;
  }
  return copyReadableToFile(fs.createReadStream(source), destination);
}

async function copyS3ObjectToFile(artifactUri, destinationPath, options = {}) {
  const parsed = new URL(artifactUri);
  const client = createS3Client(options, { purpose: "artifact" });
  const response = await client.send(
    new GetObjectCommand({
      Bucket: parsed.hostname,
      Key: parsed.pathname.replace(/^\/+/, ""),
    }),
  );
  return copyReadableToFile(response.Body, destinationPath);
}

async function copyReadableToFile(readable, destinationPath) {
  if (!readable) {
    await fs.promises.writeFile(destinationPath, "");
    return 0;
  }
  let bytes = 0;
  const counter = new Transform({
    transform(chunk, _encoding, callback) {
      bytes += Buffer.isBuffer(chunk) ? chunk.byteLength : Buffer.byteLength(chunk);
      callback(null, chunk);
    },
  });
  await pipeline(readable, counter, fs.createWriteStream(destinationPath));
  return bytes;
}

function safeDownloadFileName(value) {
  const base = path.basename(String(value ?? "").trim() || "portable_inference_bundle.zip");
  const safe = base.replace(/[<>:"/\\|?*\u0000-\u001f]/g, "-").trim();
  return safe || "portable_inference_bundle.zip";
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

function rejectParentTraversalSegments(value, label) {
  const text = String(value ?? "").trim();
  if (!text) {
    return;
  }
  const candidate = text.startsWith("file://") ? safeURLPathname(text) : text;
  const normalized = safeDecodeURIComponent(candidate).replace(/\\/g, "/");
  if (normalized.split("/").some((part) => part === "..")) {
    throw new Error(`${label} path must not contain parent directory segments.`);
  }
}

function safeURLPathname(value) {
  try {
    return new URL(value).pathname;
  } catch {
    return value;
  }
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
  const env = missionControlEnv();
  const accessKeyId = String(options.accessKeyId ?? env.AWS_ACCESS_KEY_ID ?? "").trim();
  const secretAccessKey = String(options.secretAccessKey ?? env.AWS_SECRET_ACCESS_KEY ?? "").trim();
  if (!accessKeyId || !secretAccessKey) {
    throw new Error("S3 credentials are required.");
  }
  return new S3Client({
    endpoint: resolveS3Endpoint(options, purpose, env),
    region: options.region ?? env.AWS_DEFAULT_REGION ?? "us-east-1",
    forcePathStyle: true,
    credentials: {
      accessKeyId,
      secretAccessKey,
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

async function predictChampionDemoLocal(request = {}, runtime = {}) {
  const session = ensureChampionDemoRuntime(request, runtime);
  cancelChampionDemoRuntimeIdleTimer(session);
  let response;
  try {
    response = await sendChampionDemoRuntimeMessage(session, {
      ...request,
      op: "predict",
    });
  } finally {
    scheduleChampionDemoRuntimeIdleCleanup(session);
  }
  if (!response || response.ok !== true) {
    throw new Error(localChampionDemoRuntimeError(response, session));
  }
  if (!response.prediction || typeof response.prediction !== "object") {
    throw new Error("Local Python demo runtime returned no prediction payload.");
  }
  return response;
}

function ensureChampionDemoRuntime(request = {}, runtime = {}) {
  const key = championDemoRuntimeKey(request);
  if (!key) {
    throw new Error("A champion/export runtime key is required for local demo inference.");
  }
  if (championDemoRuntime && championDemoRuntime.key === key && isWorkerRunning(championDemoRuntime.child)) {
    return championDemoRuntime;
  }
  stopChampionDemoRuntime({ reason: "runtime_key_changed" });

  const env = missionControlEnv(runtime.env ?? process.env);
  const root = repoRoot(env);
  const workerDir = workerRoot(root);
  if (!fs.existsSync(workerDir)) {
    throw new Error(`Worker directory does not exist: ${workerDir}`);
  }
  const logDir = resolveLogDir(root, env);
  fs.mkdirSync(logDir, { recursive: true });
  const childEnv = {
    ...env,
    MODEL_EXPRESS_ROOT: root,
    MODEL_EXPRESS_LOG_DIR: env.MODEL_EXPRESS_LOG_DIR ?? logDir,
    PYTHONUNBUFFERED: "1",
  };
  const python = runtime.python ?? resolveWorkerPython(workerDir, childEnv);
  const spawnFn = runtime.spawnFn ?? spawn;
  const child = spawnFn(python, ["-m", "worker.exporting.demo_runtime"], {
    cwd: workerDir,
    env: childEnv,
    windowsHide: true,
    stdio: ["pipe", "pipe", "pipe"],
  });
  const session = {
    key,
    child,
    pending: new Map(),
    nextId: 1,
    stdoutBuffer: "",
    stderrTail: "",
    logDir,
    timeoutMs: positiveDurationMs(runtime.timeoutMs ?? request.timeoutMs, CHAMPION_DEMO_RUNTIME_TIMEOUT_MS),
    idleTtlMs: positiveDurationMs(runtime.idleTtlMs ?? request.idleTtlMs, CHAMPION_DEMO_RUNTIME_IDLE_TTL_MS),
    idleTimer: null,
  };
  championDemoRuntime = session;

  child.stdout?.on("data", (data) => handleChampionDemoRuntimeStdout(session, data));
  child.stderr?.on("data", (data) => {
    const text = safeLogText(data.toString());
    session.stderrTail = `${session.stderrTail}${text}`.slice(-4000);
    console.error(`[demo-runtime:${key}] ${text.trimEnd()}`);
    appendDiagnosticLog(logDir, "warn", "champion_demo_runtime_stderr", { runtime_key: key, message: text.trim().slice(0, 1000) });
  });
  child.on("exit", (code, signal) => {
    rejectChampionDemoRuntimePending(session, new Error(`Local Python demo runtime exited code=${code} signal=${signal || ""}`.trim()));
    if (championDemoRuntime === session) {
      championDemoRuntime = null;
    }
    appendDiagnosticLog(logDir, "warn", "champion_demo_runtime_exited", { runtime_key: key, code, signal });
  });
  child.on("error", (error) => {
    rejectChampionDemoRuntimePending(session, error);
    if (championDemoRuntime === session) {
      championDemoRuntime = null;
    }
    appendDiagnosticLog(logDir, "error", "champion_demo_runtime_start_failed", { runtime_key: key, error: error.message });
  });

  if (!child.pid) {
    throw new Error("Local Python demo runtime did not start. Check that Python and worker dependencies are installed.");
  }
  appendDiagnosticLog(logDir, "info", "champion_demo_runtime_started", { runtime_key: key, pid: child.pid, python });
  return session;
}

function championDemoRuntimeKey(request = {}) {
  const explicit = String(request.runtimeKey ?? request.runtime_key ?? "").trim();
  if (explicit) return explicit;
  return [request.projectId, request.championId, request.exportId, request.exportArtifactUri]
    .map((value) => String(value ?? "").trim())
    .filter(Boolean)
    .join(":");
}

function sendChampionDemoRuntimeMessage(session, message) {
  if (!isWorkerRunning(session.child) || !session.child.stdin) {
    throw new Error("Local Python demo runtime is not running.");
  }
  const id = String(message.id ?? message.request_id ?? `demo-${session.nextId++}`);
  const payload = { ...message, id };
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      session.pending.delete(id);
      reject(new Error(`Local Python demo inference timed out after ${session.timeoutMs}ms.${session.stderrTail ? ` ${session.stderrTail}` : ""}`));
    }, session.timeoutMs);
    session.pending.set(id, { resolve, reject, timeout });
    session.child.stdin.write(`${JSON.stringify(payload)}\n`, (error) => {
      if (!error) return;
      clearTimeout(timeout);
      session.pending.delete(id);
      reject(error);
    });
  });
}

function handleChampionDemoRuntimeStdout(session, data) {
  session.stdoutBuffer += data.toString();
  const lines = session.stdoutBuffer.split(/\r?\n/);
  session.stdoutBuffer = lines.pop() ?? "";
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let response;
    try {
      response = JSON.parse(trimmed);
    } catch {
      session.stderrTail = `${session.stderrTail}\nNon-JSON runtime output: ${safeLogText(trimmed)}`.slice(-4000);
      continue;
    }
    const id = String(response.id ?? "");
    const pending = session.pending.get(id);
    if (!pending) continue;
    clearTimeout(pending.timeout);
    session.pending.delete(id);
    pending.resolve(response);
  }
}

function scheduleChampionDemoRuntimeIdleCleanup(session) {
  cancelChampionDemoRuntimeIdleTimer(session);
  if (session.idleTtlMs <= 0) return;
  session.idleTimer = setTimeout(() => {
    if (championDemoRuntime === session && session.pending.size === 0) {
      stopChampionDemoRuntime({ reason: "idle_ttl" });
    }
  }, session.idleTtlMs);
}

function cancelChampionDemoRuntimeIdleTimer(session) {
  if (session?.idleTimer) {
    clearTimeout(session.idleTimer);
    session.idleTimer = null;
  }
}

function stopChampionDemoRuntime(options = {}) {
  const session = championDemoRuntime;
  if (!session) {
    return { stopped: false, status: "not_running" };
  }
  championDemoRuntime = null;
  cancelChampionDemoRuntimeIdleTimer(session);
  rejectChampionDemoRuntimePending(session, new Error("Local Python demo runtime was stopped."));
  try {
    if (isWorkerRunning(session.child) && session.child.stdin?.writable) {
      session.child.stdin.write(`${JSON.stringify({ id: `shutdown-${Date.now()}`, op: "shutdown" })}\n`);
    }
  } catch {
    // Best-effort graceful shutdown; kill below if the process remains alive.
  }
  if (isWorkerRunning(session.child)) {
    session.child.kill();
  }
  appendDiagnosticLog(session.logDir, "info", "champion_demo_runtime_stopped", { runtime_key: session.key, reason: options.reason || "explicit" });
  return { stopped: true, status: "stopped", runtime_key: session.key };
}

function rejectChampionDemoRuntimePending(session, error) {
  for (const [id, pending] of session.pending.entries()) {
    clearTimeout(pending.timeout);
    pending.reject(error);
    session.pending.delete(id);
  }
}

function localChampionDemoRuntimeError(response, session) {
  const message = response && typeof response === "object" ? response.error || response.message : "";
  const code = response && typeof response === "object" ? response.code || response.error_code : "";
  const details = [code, message, session?.stderrTail].filter(Boolean).join(": ");
  return details || "Local Python demo inference failed.";
}

function positiveDurationMs(value, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    return fallback;
  }
  return Math.trunc(parsed);
}

function championDemoRuntimeSnapshot() {
  return championDemoRuntime
    ? { key: championDemoRuntime.key, pid: championDemoRuntime.child?.pid ?? 0, pending: championDemoRuntime.pending.size }
    : null;
}

async function ensureProjectWorker(options = {}) {
  const projectId = String(options.projectId ?? "").trim();
  if (!projectId) {
    throw new Error("Project id is required before starting a worker.");
  }

  const repoRoot = process.env.MODEL_EXPRESS_ROOT ?? path.resolve(__dirname, "..", "..", "..");
  const env = missionControlEnv({ ...process.env, MODEL_EXPRESS_ROOT: repoRoot });
  const baseUrl = validateOrchestratorBaseUrl(options.baseUrl || DEFAULT_ORCHESTRATOR_URL, env);
  const workerDir = path.join(repoRoot, "services", "worker");
  if (!fs.existsSync(workerDir)) {
    throw new Error(`Worker directory does not exist: ${workerDir}`);
  }

  const logDir = resolveLogDir(repoRoot, env);
  fs.mkdirSync(logDir, { recursive: true });
  const targetCount = normalizeWorkerCount(options.count);
  const useModalDispatcher = shouldUseModalDispatcher(options);
  if (useModalDispatcher) {
    const preflight = await preflightCloud(
      { stage: "worker_start", baseUrl, live: true },
      { env },
    );
    if (preflight.status !== "ok") {
      throw new Error(formatCloudPreflightError(preflight));
    }
  }
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
        env,
        logDir,
      });
    }

    const workerName = options.name
      ? `${options.name}-${slot}`
      : `project-${projectId}-worker-${slot}`;
    const childEnv = {
      ...env,
      ORCHESTRATOR_URL: baseUrl,
      PROJECT_ID: projectId,
      WORKER_NAME: workerName,
      GPU_TYPE: options.gpuType ?? "local",
      MODEL_EXPRESS_ROOT: repoRoot,
      MODEL_EXPRESS_LOG_DIR: env.MODEL_EXPRESS_LOG_DIR ?? logDir,
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

async function ensureRemoteTrainingSession({ projectId, baseUrl, repoEnv = {}, env: providedEnv, logDir, startTunnel = startCloudflaredTunnel, ensureLocalRuntime = ensureAppLocalRuntime }) {
  const existing = projectTunnels.get(projectId);
  if (existing && remoteTrainingSessionActive(existing)) {
    return existing;
  }
  stopProjectTunnels(projectId);

  const env = providedEnv ?? { ...process.env, ...repoEnv };
  const childEnv = {};
  const processes = [];
  const apiToken = String(env.MODEL_EXPRESS_API_TOKEN ?? "").trim();
  if (apiToken) {
    childEnv.MODEL_EXPRESS_API_TOKEN = apiToken;
  }

  const configuredOrchestrator = firstNonEmpty(env.MODAL_ORCHESTRATOR_URL, env.MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL);
  if (configuredOrchestrator) {
    requireAuthenticatedOrchestratorExposure(env);
    childEnv.MODAL_ORCHESTRATOR_URL = validateRemoteModalUrl(configuredOrchestrator, "MODAL_ORCHESTRATOR_URL");
  } else if (!isLoopbackHostname(new URL(baseUrl).hostname)) {
    requireAuthenticatedOrchestratorExposure(env);
    childEnv.MODAL_ORCHESTRATOR_URL = validateRemoteModalUrl(baseUrl, "MODAL_ORCHESTRATOR_URL");
  } else {
    requireAuthenticatedOrchestratorExposure(env);
    const orchestratorTunnel = await startTunnel({
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
      try {
        if (appManagedLocalRuntimeEnabled(env)) {
          await ensureLocalRuntime({ env });
        }
        const s3Tunnel = await startTunnel({
          projectId,
          label: "s3",
          targetUrl,
          logDir,
          env,
        });
        childEnv.MODAL_S3_ENDPOINT_URL = s3Tunnel.url;
        processes.push(s3Tunnel.child);
      } catch (error) {
        for (const child of processes) {
          stopTunnelProcess(child);
        }
        throw error;
      }
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

function resolveLogDir(repoRoot, env = process.env) {
  return env.MODEL_EXPRESS_LOG_DIR ?? path.join(repoRoot, "artifacts", "logs");
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

function loadRepoEnv(repoRoot, runtimeEnv = process.env) {
  const env = {};
  const envFile = runtimeEnv.MODEL_EXPRESS_ENV_FILE;
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
  let env = missionControlEnv();
  const runtime = await ensureAppLocalRuntime({ env });
  env = runtime.env ?? env;
  const folder = resolveDatasetFolderOption(options);
  const datasetPath = folder.path;

  const datasetName = path.basename(datasetPath);
  const safeName = datasetName.replace(/[^a-zA-Z0-9._-]/g, "-");
  const bucket = options.bucket ?? env.S3_BUCKET ?? "model-express";
  const endpoint = validateUploadEndpoint(options.endpoint, env);
  const accessKeyId = String(options.accessKeyId ?? env.AWS_ACCESS_KEY_ID ?? "").trim();
  const secretAccessKey = String(options.secretAccessKey ?? env.AWS_SECRET_ACCESS_KEY ?? "").trim();
  if (!accessKeyId || !secretAccessKey) {
    throw new Error("S3 credentials are required before dataset upload.");
  }
  const region = options.region ?? env.AWS_DEFAULT_REGION ?? "us-east-1";
  const key = `datasets/${projectId}/${safeName}.zip`;
  const entries = await collectDatasetFiles(datasetPath);
  if (entries.length === 0) {
    throw new Error(`Dataset folder does not contain any files: ${datasetPath}`);
  }
  const archivePlan = planZipArchive(entries);
  const preflight = datasetUploadPreflight(entries, archivePlan, options, env);
  if (preflight.errors.length > 0) {
    throw new Error(preflight.errors.map((item) => item.message).join(" "));
  }
  if (preflight.warnings.length > 0) {
    appendDiagnosticLog(resolveLogDir(repoRoot(env), env), "warn", "dataset_upload_preflight_warning", {
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

async function preflightDatasetFolder(options = {}) {
  const env = missionControlEnv();
  const folder = resolveDatasetFolderOption(options);
  const entries = await collectDatasetFiles(folder.path);
  const archivePlan = planZipArchive(entries);
  return datasetUploadPreflight(entries, archivePlan, options, env);
}

async function preflightCloud(options = {}, runtime = {}) {
  const env = runtime.env ?? missionControlEnv();
  const checks = [];
  const stage = String(options.stage ?? "manual").trim() || "manual";
  const live = options.live !== false;
  const baseUrl = validateOrchestratorBaseUrl(options.baseUrl || DEFAULT_ORCHESTRATOR_URL, env);

  addCloudCheck(
    checks,
    cloudProfile(env) === "cloud" ? "ok" : "failed",
    "cloud_profile",
    cloudProfile(env) === "cloud"
      ? "Cloud v1 profile is enabled."
      : "Cloud Agentic Demo requires the cloud v1 profile.",
    cloudProfile(env) === "cloud"
      ? ""
      : "Set MODEL_EXPRESS_V1_PROFILE=cloud and start with MODEL_EXPRESS_ENV_FILE=.env.v1.cloud.",
    { stage, profile: cloudProfile(env) },
  );

  const provider = normalizedProvider(env.MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER || "local");
  addCloudCheck(
    checks,
    provider === "modal" ? "ok" : "failed",
    "training_provider_env",
    provider === "modal"
      ? "Environment default training provider is Modal."
      : "Cloud Agentic Demo requires Modal as the default training provider.",
    provider === "modal" ? "" : "Set MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER=modal.",
    { provider },
  );

  addOpenAIEnvChecks(checks, env);
  addModalAuthCheck(checks, env);
  addModalOrchestratorCheck(checks, env, baseUrl);
  addS3EndpointCheck(checks, env);
  addModalS3EndpointCheck(checks, env);
  addCloudBucketCheck(checks, env);
  addStorageCredentialChecks(checks, env);
  await addLocalRuntimeChecks(checks, { env, live, runtime });

  try {
    requireAuthenticatedOrchestratorExposure(env);
    addCloudCheck(checks, "ok", "api_token_env", "Public orchestrator token controls are configured.", "", {
      token_present: Boolean(String(env.MODEL_EXPRESS_API_TOKEN ?? "").trim()),
    });
  } catch (error) {
    addCloudCheck(
      checks,
      "failed",
      "api_token_env",
      "Modal cannot reach the orchestrator safely without public-token controls.",
      errorMessage(error),
    );
  }
  addCloudCheck(
    checks,
    automaticTunnelsEnabled(env) ? "ok" : "failed",
    "automatic_tunnels",
    automaticTunnelsEnabled(env) ? "Automatic Cloudflare tunnels are enabled." : "Automatic Cloudflare tunnels are disabled.",
    automaticTunnelsEnabled(env) ? "" : "Unset MODEL_EXPRESS_DISABLE_AUTO_TUNNELS so Mission Control can create Modal tunnels automatically.",
  );

  const fetchFn = runtime.fetch ?? globalThis.fetch;
  await mergeBackendCloudPreflight(checks, { baseUrl, stage, live, env, fetchFn, automaticTunnels: automaticTunnelsEnabled(env) });
  if (live) {
    await preflightPublicOrchestrator(checks, { env, fetchFn });
    await preflightS3RoundTrip(checks, { env, s3LiveCheck: runtime.s3LiveCheck });
  }

  const status = checks.some((check) => check.status === "failed") ? "failed" : "ok";
  return { status, checks };
}

async function addLocalRuntimeChecks(checks, { env, live, runtime }) {
  if (!appManagedLocalRuntimeEnabled(env)) {
    addCloudCheck(checks, "warning", "local_runtime_config", "App-managed local runtime is disabled.");
    return;
  }
  addCloudCheck(checks, "ok", "local_runtime_config", "App-local runtime config is generated.", "", {
    token_present: Boolean(String(env.MODEL_EXPRESS_API_TOKEN ?? "").trim()),
    minio_access_key_present: Boolean(String(env.AWS_ACCESS_KEY_ID ?? "").trim()),
    minio_secret_key_present: Boolean(String(env.AWS_SECRET_ACCESS_KEY ?? "").trim()),
    bucket: env.S3_BUCKET || DEFAULT_S3_BUCKET,
    artifact_prefix: env.MODEL_EXPRESS_ARTIFACT_PREFIX || DEFAULT_ARTIFACT_PREFIX,
  });
  if (!live) {
    addCloudCheck(checks, "warning", "docker_compose", "Docker Compose local services will be started automatically when needed.", "", {
      services: ["postgres", "minio"],
    });
    addCloudCheck(checks, "warning", "minio_api", "Local MinIO API will be verified before upload or worker start.", "", {
      endpoint: env.S3_ENDPOINT_URL || DEFAULT_S3_ENDPOINT_URL,
    });
    return;
  }
  try {
    const ensureRuntime = runtime.ensureLocalRuntime ?? ensureAppLocalRuntime;
    const result = await ensureRuntime({ ...runtime, env });
    addCloudCheck(checks, "ok", "docker_compose", "Docker Compose local services are running.", "", {
      services: result.services ?? ["postgres", "minio"],
    });
    addCloudCheck(checks, "ok", "minio_api", "Local MinIO API is reachable.", "", {
      endpoint: env.S3_ENDPOINT_URL || DEFAULT_S3_ENDPOINT_URL,
    });
    addCloudCheck(checks, "ok", "s3_bucket_exists", "Local MinIO bucket exists.", "", {
      bucket: result.s3_bucket ?? env.S3_BUCKET ?? DEFAULT_S3_BUCKET,
    });
  } catch (error) {
    const stage = error?.localRuntimeStage === "minio_api" ? "minio_api" : "docker_compose";
    addCloudCheck(
      checks,
      "failed",
      stage,
      stage === "minio_api" ? "Local MinIO API is not reachable." : "Docker Compose local services are not running.",
      safeProviderError(errorMessage(error)),
    );
  }
}

function addOpenAIEnvChecks(checks, env) {
  const provider = normalizedProvider(env.MODEL_EXPRESS_LLM_PROVIDER || env.MODEL_EXPRESS_VISUAL_LLM_PROVIDER || "openai");
  const apiStyle = String(env.MODEL_EXPRESS_LLM_API_STYLE || env.MODEL_EXPRESS_VISUAL_LLM_API_STYLE || "").trim().toLowerCase();
  const storedResponses = String(env.MODEL_EXPRESS_LLM_STORED_RESPONSES ?? env.MODEL_EXPRESS_VISUAL_LLM_STORED_RESPONSES ?? "true").trim().toLowerCase();
  const keySource = openAIKeySource(env, provider);

  addCloudCheck(
    checks,
    provider === "openai" ? "ok" : "failed",
    "openai_provider_env",
    provider === "openai" ? "OpenAI provider is configured." : "Cloud Agentic Demo requires the OpenAI provider.",
    provider === "openai" ? "" : "Set MODEL_EXPRESS_LLM_PROVIDER=openai.",
    { provider },
  );
  addCloudCheck(
    checks,
    apiStyle === "responses" ? "ok" : "failed",
    "openai_responses_env",
    apiStyle === "responses" ? "OpenAI Responses API style is configured." : "Cloud Agentic Demo requires the OpenAI Responses API.",
    apiStyle === "responses" ? "" : "Set MODEL_EXPRESS_LLM_API_STYLE=responses.",
    { api_style: apiStyle || "chat_completions" },
  );
  addCloudCheck(
    checks,
    !["0", "false", "no", "off"].includes(storedResponses) ? "ok" : "failed",
    "openai_stored_responses_env",
    !["0", "false", "no", "off"].includes(storedResponses)
      ? "Stored Responses are enabled."
      : "Cloud Agentic Demo requires stored Responses.",
    !["0", "false", "no", "off"].includes(storedResponses) ? "" : "Set MODEL_EXPRESS_LLM_STORED_RESPONSES=true.",
  );
  addCloudCheck(
    checks,
    keySource ? "ok" : "failed",
    "openai_key_env",
    keySource ? "OpenAI API key is configured." : "Cloud Agentic Demo requires an OpenAI API key.",
    keySource ? "" : "Set MODEL_EXPRESS_LLM_API_KEY or OPENAI_API_KEY and run preflight again.",
    keySource ? { source: keySource } : undefined,
  );
}

function addModalAuthCheck(checks, env) {
  const source = modalAuthSource(env);
  addCloudCheck(
    checks,
    source ? "ok" : "failed",
    "modal_auth",
    source ? "Modal authentication is configured." : "Cloud Agentic Demo requires Modal authentication.",
    source ? "" : "Run Modal auth setup or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.",
    source ? { source } : undefined,
  );
}

function modalAuthSource(env = process.env) {
  if (String(env.MODAL_TOKEN_ID ?? "").trim() && String(env.MODAL_TOKEN_SECRET ?? "").trim()) {
    return "MODAL_TOKEN_ID/MODAL_TOKEN_SECRET";
  }
  if (String(env.MODAL_PROFILE ?? "").trim()) {
    return "MODAL_PROFILE";
  }
  const configuredPath = String(env.MODAL_CONFIG_PATH ?? "").trim();
  if (configuredPath && fs.existsSync(configuredPath)) {
    return "MODAL_CONFIG_PATH";
  }
  const homeConfig = path.join(os.homedir(), ".modal.toml");
  if (fs.existsSync(homeConfig)) {
    return "modal_config";
  }
  return "";
}

function openAIKeySource(env, provider) {
  if (String(env.MODEL_EXPRESS_LLM_API_KEY ?? "").trim()) return "MODEL_EXPRESS_LLM_API_KEY";
  if (String(env.MODEL_EXPRESS_VISUAL_LLM_API_KEY ?? "").trim()) return "MODEL_EXPRESS_VISUAL_LLM_API_KEY";
  if (provider === "openai" && String(env.OPENAI_API_KEY ?? "").trim()) return "OPENAI_API_KEY";
  return "";
}

function addCloudURLCheck(checks, id, label, value, env) {
  try {
    const origin = validateRemoteModalUrl(value, label, env);
    const parsed = new URL(origin);
    if (parsed.protocol !== "https:") {
      throw new Error(`${label} must use https for the cloud v1 preflight.`);
    }
    addCloudCheck(checks, "ok", id, `${label} is a public HTTPS origin.`, "", {
      scheme: parsed.protocol.replace(":", ""),
      host: parsed.hostname,
    });
  } catch (error) {
    addCloudCheck(
      checks,
      "failed",
      id,
      `${label} is not a public HTTPS URL reachable by Modal.`,
      `${label} must be a public HTTPS origin and must not point at private/local hosts, Postgres, MLflow, or the MinIO console.`,
      { error: errorMessage(error) },
    );
  }
}

function addModalOrchestratorCheck(checks, env, baseUrl) {
  const configured = firstNonEmpty(env.MODAL_ORCHESTRATOR_URL, env.MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL);
  if (configured) {
    addCloudURLCheck(checks, "modal_orchestrator_url", "MODAL_ORCHESTRATOR_URL", configured, env);
    return;
  }
  if (!automaticTunnelsEnabled(env)) {
    addCloudCheck(
      checks,
      "failed",
      "modal_orchestrator_url",
      "Modal orchestrator tunnel is not configured.",
      "Enable automatic tunnels or set MODAL_ORCHESTRATOR_URL to a public HTTPS origin.",
    );
    return;
  }
  try {
    const parsed = parseHttpOrigin(baseUrl, "Orchestrator URL");
    if (isLoopbackHostname(parsed.hostname)) {
      validateCloudflaredTarget(parsed.origin, "orchestrator");
      addCloudCheck(
        checks,
        "warning",
        "modal_orchestrator_url",
        "Orchestrator tunnel will be created automatically at worker start.",
        "",
        { automatic_tunnel: true, target: "loopback" },
      );
      return;
    }
    const origin = validateRemoteModalUrl(parsed.origin, "MODAL_ORCHESTRATOR_URL", env);
    if (new URL(origin).protocol !== "https:") {
      throw new Error("MODAL_ORCHESTRATOR_URL must use https for the cloud v1 preflight.");
    }
    addCloudCheck(checks, "ok", "modal_orchestrator_url", "Orchestrator base URL is a public HTTPS origin.", "", {
      scheme: "https",
      host: new URL(origin).hostname,
    });
  } catch (error) {
    addCloudCheck(
      checks,
      "failed",
      "modal_orchestrator_url",
      "Modal cannot reach the orchestrator.",
      safeProviderError(errorMessage(error)),
    );
  }
}

function addS3EndpointCheck(checks, env) {
  const endpoint = String(env.S3_ENDPOINT_URL ?? "").trim();
  if (!endpoint) {
    if (s3TunnelEnabled(env)) {
      addCloudCheck(
        checks,
        "warning",
        "s3_endpoint",
        "Local S3 endpoint will be exposed through an automatic tunnel at worker start.",
        "",
        { automatic_tunnel: true, target: DEFAULT_S3_ENDPOINT_URL },
      );
      return;
    }
    addCloudCheck(
      checks,
      "failed",
      "s3_endpoint",
      "S3 endpoint is not configured.",
      "Set S3_ENDPOINT_URL to a public HTTPS S3/R2 endpoint or enable MODEL_EXPRESS_MODAL_TUNNEL_S3 for local MinIO tunneling.",
    );
    return;
  }
  try {
    const parsed = parseHttpOrigin(endpoint, "S3_ENDPOINT_URL");
    if (parsed.port === "9001") {
      throw new Error("S3_ENDPOINT_URL must not point at the MinIO console port.");
    }
    if (isLoopbackHostname(parsed.hostname)) {
      if (!s3TunnelEnabled(env)) {
        throw new Error("S3_ENDPOINT_URL is local; Modal needs a public S3 endpoint or automatic S3 tunneling.");
      }
      validateCloudflaredTarget(parsed.origin, "s3");
      addCloudCheck(
        checks,
        "warning",
        "s3_endpoint",
        "Local S3 endpoint will be exposed through an automatic tunnel at worker start.",
        "",
        { automatic_tunnel: true, target: parsed.origin },
      );
      return;
    }
    const origin = validateRemoteModalUrl(endpoint, "S3_ENDPOINT_URL", env);
    const remote = new URL(origin);
    if (remote.protocol !== "https:") {
      throw new Error("S3_ENDPOINT_URL must use https for the cloud v1 preflight.");
    }
    addCloudCheck(checks, "ok", "s3_endpoint", "S3_ENDPOINT_URL is a public HTTPS origin.", "", {
      scheme: "https",
      host: remote.hostname,
    });
  } catch (error) {
    addCloudCheck(
      checks,
      "failed",
      "s3_endpoint",
      "S3_ENDPOINT_URL is not a public HTTPS URL reachable by Modal.",
      safeProviderError(errorMessage(error)),
    );
  }
}

function addModalS3EndpointCheck(checks, env) {
  const configured = firstNonEmpty(env.MODAL_S3_ENDPOINT_URL, env.MODEL_EXPRESS_MODAL_S3_ENDPOINT_URL);
  if (configured) {
    addCloudURLCheck(checks, "modal_s3_endpoint", "MODAL_S3_ENDPOINT_URL", configured, env);
    return;
  }
  const endpoint = String(env.S3_ENDPOINT_URL ?? "").trim();
  if (endpoint) {
    try {
      const parsed = parseHttpOrigin(endpoint, "S3_ENDPOINT_URL");
      if (!isLoopbackHostname(parsed.hostname)) {
        const origin = validateRemoteModalUrl(parsed.origin, "S3_ENDPOINT_URL", env);
        if (new URL(origin).protocol !== "https:") {
          throw new Error("S3_ENDPOINT_URL must use https for Modal workers.");
        }
        addCloudCheck(checks, "ok", "modal_s3_endpoint", "Modal workers will use S3_ENDPOINT_URL.", "", {
          scheme: "https",
          host: new URL(origin).hostname,
        });
        return;
      }
    } catch (error) {
      addCloudCheck(
        checks,
        "failed",
        "modal_s3_endpoint",
        "MODAL_S3_ENDPOINT_URL is not configured and S3_ENDPOINT_URL is not safe for Modal.",
        safeProviderError(errorMessage(error)),
      );
      return;
    }
  }
  if (s3TunnelEnabled(env)) {
    try {
      const targetUrl = endpoint ? parseHttpOrigin(endpoint, "S3_ENDPOINT_URL").origin : DEFAULT_S3_ENDPOINT_URL;
      validateCloudflaredTarget(targetUrl, "s3");
      addCloudCheck(
        checks,
        "warning",
        "modal_s3_endpoint",
        "S3 tunnel will be created automatically at worker start.",
        "",
        { automatic_tunnel: true },
      );
      return;
    } catch (error) {
      addCloudCheck(
        checks,
        "failed",
        "modal_s3_endpoint",
        "Automatic S3 tunnel target is not safe for Modal.",
        safeProviderError(errorMessage(error)),
      );
      return;
    }
  }
  addCloudCheck(
    checks,
    "failed",
    "modal_s3_endpoint",
    "Modal S3 endpoint is not configured.",
    "Set MODAL_S3_ENDPOINT_URL, use a public HTTPS S3_ENDPOINT_URL, or enable MODEL_EXPRESS_MODAL_TUNNEL_S3 for local MinIO tunneling.",
  );
}

function automaticTunnelsEnabled(env) {
  return !envFlagFrom(env, "MODEL_EXPRESS_DISABLE_AUTO_TUNNELS", false);
}

function s3TunnelEnabled(env) {
  return (
    envFlagFrom(env, "MODEL_EXPRESS_MODAL_TUNNEL_S3", false) ||
    envFlagFrom(env, "MODEL_EXPRESS_ALLOW_MODAL_MINIO_TUNNEL", false)
  );
}

function addCloudBucketCheck(checks, env) {
  const bucket = firstNonEmpty(env.S3_BUCKET, env.MODEL_EXPRESS_ARTIFACT_BUCKET);
  addCloudCheck(
    checks,
    bucket ? "ok" : "failed",
    "s3_bucket",
    bucket ? "S3 bucket is configured." : "S3 bucket is not configured.",
    bucket ? "" : "Set S3_BUCKET and MODEL_EXPRESS_ARTIFACT_BUCKET.",
    bucket ? { bucket } : undefined,
  );
}

function addStorageCredentialChecks(checks, env) {
  addCredentialCheck(checks, "s3_upload_credentials", env.AWS_ACCESS_KEY_ID, env.AWS_SECRET_ACCESS_KEY, "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to scoped upload credentials.", env);
  const modalAccessKey = firstNonEmpty(env.MODEL_EXPRESS_MODAL_AWS_ACCESS_KEY_ID, env.AWS_ACCESS_KEY_ID);
  const modalSecretKey = firstNonEmpty(env.MODEL_EXPRESS_MODAL_AWS_SECRET_ACCESS_KEY, env.AWS_SECRET_ACCESS_KEY);
  addCredentialCheck(
    checks,
    "modal_s3_credentials",
    modalAccessKey,
    modalSecretKey,
    "Set MODEL_EXPRESS_MODAL_AWS_ACCESS_KEY_ID and MODEL_EXPRESS_MODAL_AWS_SECRET_ACCESS_KEY to scoped Modal credentials.",
    env,
  );
}

function addCredentialCheck(checks, id, accessKey, secretKey, remediation, env) {
  const accessPresent = Boolean(String(accessKey ?? "").trim());
  const secretPresent = Boolean(String(secretKey ?? "").trim());
  if (!accessPresent || !secretPresent) {
    addCloudCheck(checks, "failed", id, "S3 credentials are incomplete.", remediation, {
      access_key_present: accessPresent,
      secret_key_present: secretPresent,
    });
    return;
  }
  if (
    String(accessKey).trim() === "model_express" &&
    String(secretKey).trim() === "model_express_password" &&
    !envFlagFrom(env, "MODEL_EXPRESS_ALLOW_MODAL_ROOT_STORAGE", false)
  ) {
    addCloudCheck(
      checks,
      "failed",
      id,
      "Cloud storage refuses default local MinIO root credentials.",
      "Use scoped S3 credentials. Set MODEL_EXPRESS_ALLOW_MODAL_ROOT_STORAGE=true only for an explicit advanced demo fallback.",
    );
    return;
  }
  addCloudCheck(checks, "ok", id, "Scoped S3 credentials are configured.", "", {
    access_key_present: true,
    secret_key_present: true,
  });
}

async function mergeBackendCloudPreflight(checks, { baseUrl, stage, live, env, fetchFn, automaticTunnels }) {
  if (typeof fetchFn !== "function") {
    addCloudCheck(checks, "failed", "backend_preflight", "Backend cloud preflight could not run.", "Start the orchestrator and run preflight again.");
    return;
  }
  try {
    const response = await fetchFn(new URL("/preflight/cloud", baseUrl).toString(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...apiTokenHeaders(env),
      },
      body: JSON.stringify({ stage, live, automatic_tunnels: Boolean(automaticTunnels) }),
    });
    const payload = await parseJSONResponse(response);
    if (payload && Array.isArray(payload.checks)) {
      for (const check of payload.checks) {
        checks.push(normalizeCloudCheck(check, "backend"));
      }
      return;
    }
    if (!response.ok) {
      addCloudCheck(
        checks,
        "failed",
        "backend_preflight",
        response.status === 401
          ? "Public orchestrator returned 401 for cloud preflight."
          : "Backend cloud preflight failed.",
        response.status === 401
          ? "Mission Control and the orchestrator are using different MODEL_EXPRESS_API_TOKEN values."
          : safeProviderError(payload?.error || response.statusText),
        { status: response.status },
      );
      return;
    }
    addCloudCheck(checks, "ok", "backend_preflight", "Backend cloud preflight endpoint is reachable.");
  } catch (error) {
    addCloudCheck(
      checks,
      "failed",
      "backend_preflight",
      "Backend cloud preflight endpoint is not reachable.",
      `Start the orchestrator at ${baseUrl} and run preflight again. ${safeProviderError(errorMessage(error))}`,
    );
  }
}

async function preflightPublicOrchestrator(checks, { env, fetchFn }) {
  const publicURL = firstNonEmpty(env.MODAL_ORCHESTRATOR_URL, env.MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL);
  if (!publicURL) {
    addCloudCheck(
      checks,
      "warning",
      "public_orchestrator",
      "Public orchestrator probe will run after the automatic tunnel is created at worker start.",
      "",
      { automatic_tunnel: true },
    );
    return;
  }
  if (typeof fetchFn !== "function") {
    addCloudCheck(checks, "failed", "public_orchestrator", "Public orchestrator check could not run.", "Start Mission Control with a fetch-capable Electron runtime.");
    return;
  }
  let origin;
  try {
    origin = validateRemoteModalUrl(publicURL, "MODAL_ORCHESTRATOR_URL", env);
    if (new URL(origin).protocol !== "https:") {
      throw new Error("MODAL_ORCHESTRATOR_URL must use https for the cloud v1 preflight.");
    }
  } catch (error) {
    addCloudCheck(checks, "failed", "public_orchestrator", "Modal cannot reach the orchestrator.", errorMessage(error));
    return;
  }
  try {
    const response = await fetchFn(new URL("/settings/automation", origin).toString(), {
      method: "GET",
      headers: apiTokenHeaders(env),
    });
    if (response.status === 401 || response.status === 403) {
      addCloudCheck(
        checks,
        "failed",
        "public_orchestrator",
        "Public orchestrator returned 401.",
        "Mission Control and the orchestrator are using different MODEL_EXPRESS_API_TOKEN values.",
        { status: response.status },
      );
      return;
    }
    if (!response.ok) {
      addCloudCheck(checks, "failed", "public_orchestrator", "Public orchestrator check failed.", safeProviderError(response.statusText), { status: response.status });
      return;
    }
    addCloudCheck(checks, "ok", "public_orchestrator", "Public orchestrator responded with the configured API token.");
  } catch (error) {
    addCloudCheck(checks, "failed", "public_orchestrator", "Modal cannot reach the orchestrator.", safeProviderError(errorMessage(error)));
  }
}

async function preflightS3RoundTrip(checks, { env, s3LiveCheck }) {
  if (typeof s3LiveCheck === "function") {
    try {
      await s3LiveCheck(env);
      addCloudCheck(checks, "ok", "s3_live", "S3 preflight write/read/delete succeeded.");
    } catch (error) {
      addCloudCheck(checks, "failed", "s3_live", "S3 preflight write/read/delete failed.", safeProviderError(errorMessage(error)));
    }
    return;
  }
  const bucket = firstNonEmpty(env.S3_BUCKET, env.MODEL_EXPRESS_ARTIFACT_BUCKET);
  if (!bucket) {
    addCloudCheck(checks, "failed", "s3_live", "S3 preflight cannot run without a bucket.", "Set S3_BUCKET and MODEL_EXPRESS_ARTIFACT_BUCKET.");
    return;
  }
  const key = `model-express/preflight/${crypto.randomUUID()}.txt`;
  const body = Buffer.from(`model-express-cloud-preflight ${new Date().toISOString()}\n`, "utf8");
  const client = createS3ClientFromEnv(env);
  let wrote = false;
  try {
    await client.send(new PutObjectCommand({ Bucket: bucket, Key: key, Body: body, ContentLength: body.byteLength, ContentType: "text/plain" }));
    wrote = true;
    const response = await client.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
    const read = await streamToBuffer(response.Body);
    if (!read.equals(body)) {
      throw new Error("S3 preflight read did not match written bytes.");
    }
    addCloudCheck(checks, "ok", "s3_live", "S3 preflight write/read/delete succeeded.", "", { key_prefix: "model-express/preflight/" });
  } catch (error) {
    addCloudCheck(checks, "failed", "s3_live", "S3 preflight write/read/delete failed.", safeProviderError(errorMessage(error)), { key_prefix: "model-express/preflight/" });
  } finally {
    if (wrote) {
      await client.send(new DeleteObjectCommand({ Bucket: bucket, Key: key })).catch(() => undefined);
    }
  }
}

function createS3ClientFromEnv(env) {
  const accessKeyId = String(env.AWS_ACCESS_KEY_ID ?? "").trim();
  const secretAccessKey = String(env.AWS_SECRET_ACCESS_KEY ?? "").trim();
  if (!accessKeyId || !secretAccessKey) {
    throw new Error("S3 credentials are required.");
  }
  return new S3Client({
    endpoint: resolveS3Endpoint({}, "storage", env),
    region: env.AWS_DEFAULT_REGION || "us-east-1",
    forcePathStyle: true,
    credentials: {
      accessKeyId,
      secretAccessKey,
    },
  });
}

function addCloudCheck(checks, status, id, message, remediation = "", metadata) {
  const check = {
    id,
    status,
    message,
  };
  if (remediation) {
    check.remediation = remediation;
  }
  if (metadata && Object.keys(metadata).length > 0) {
    check.metadata = metadata;
  }
  checks.push(check);
}

function normalizeCloudCheck(check, source) {
  return {
    id: source ? `${source}:${String(check.id ?? "check")}` : String(check.id ?? "check"),
    status: String(check.status ?? "failed"),
    message: String(check.message ?? "Cloud preflight check failed."),
    ...(check.remediation ? { remediation: String(check.remediation) } : {}),
    ...(check.metadata && typeof check.metadata === "object" ? { metadata: sanitizeCloudMetadata(check.metadata) } : {}),
  };
}

function sanitizeCloudMetadata(metadata) {
  const safe = {};
  for (const [key, value] of Object.entries(metadata)) {
    if (sensitiveLogKey(key)) {
      safe[key] = "[redacted]";
    } else {
      safe[key] = value;
    }
  }
  return safe;
}

async function parseJSONResponse(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return { error: text };
  }
}

function cloudProfile(env) {
  const value = String(env.MODEL_EXPRESS_V1_PROFILE ?? "").trim().toLowerCase().replace(/_/g, "-");
  if (value === "cloud-v1" || value === "v1-cloud") {
    return "cloud";
  }
  return value;
}

function normalizedProvider(value) {
  const provider = String(value ?? "").trim().toLowerCase().replace(/-/g, "_");
  return provider || "local";
}

function safeProviderError(value) {
  return safeLogText(String(value ?? "").trim() || "Provider check failed.");
}

function formatCloudPreflightError(preflight) {
  const failed = Array.isArray(preflight?.checks)
    ? preflight.checks.filter((check) => String(check.status ?? "").toLowerCase() === "failed")
    : [];
  const details = failed.slice(0, 4).map((check) => {
    const remediation = check.remediation ? ` ${check.remediation}` : "";
    return `${check.message}${remediation}`;
  });
  return ["Cloud Agentic Demo / Modal + OpenAI required.", ...details].join(" ");
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function datasetUploadPreflight(entries, archivePlan, options = {}, env = process.env) {
  return buildDatasetUploadPreflight(entries, archivePlan, {
    warnFileCount: options.warnFileCount ?? env.MODEL_EXPRESS_UPLOAD_WARN_FILE_COUNT,
    warnBytes: options.warnBytes ?? env.MODEL_EXPRESS_UPLOAD_WARN_BYTES,
    maxFileCount: options.maxFileCount ?? env.MODEL_EXPRESS_UPLOAD_MAX_FILE_COUNT,
    maxBytes: options.maxBytes ?? env.MODEL_EXPRESS_UPLOAD_MAX_BYTES,
  });
}

module.exports = {
  __test: {
    apiTokenHeaders,
    appLocalRuntimeEnv,
    appManagedLocalRuntimeEnabled,
    bootstrapAppLocalRuntime,
    championDemoRuntimeSnapshot,
    configuredLocalArtifactRoots,
    dockerComposeLocalServicesSpec,
    ensureAppLocalRuntime,
    ensureLocalApiTokenEnv,
    ensureLocalRuntimeConfig,
    ensureRemoteTrainingSession,
    ensureS3Bucket,
    externalDataMountPath,
    isLoopbackHostname,
    localConfigPath,
    missionControlEnv,
    parseCloudflaredTunnelUrl,
    predictChampionDemoLocal,
    preflightCloud,
    preflightDatasetFolder,
    rememberDatasetFolder,
    requireAuthenticatedOrchestratorExposure,
    remoteTrainingSessionActive,
    remoteTrainingSessionTtlMs,
    resolveDatasetFolderOption,
    resolveS3Endpoint,
    safeLogText,
    saveExportArtifact,
    selectedDatasetFolders,
    startDockerComposeLocalServices,
    stopChampionDemoRuntime,
    stopProjectTunnels,
    waitForMinioBucket,
    validateAppRequestPath,
    validateLocalArtifactPath,
    validateLocalExportArtifactPath,
    validateLocalPortableBundlePath,
    validateOrchestratorBaseUrl,
    validateOrchestratorRequest,
    validateRemoteModalUrl,
    validateS3ExportArtifactUri,
    validateUploadEndpoint,
  },
};
