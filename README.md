# Model Express

Model Express is a local-first training workbench for computer vision experiments. It helps you turn an image dataset into a repeatable experiment run: upload a dataset, profile it, generate candidate training plans, run experiments on local hardware or Modal GPUs, compare results, select a champion model, export it, and test predictions from the desktop UI.

The project is currently distributed as source code instead of a packaged executable. The setup is intentionally explicit: you run the support services, the orchestrator, and the Mission Control desktop app from the repo.

## What This Project Does

Model Express gives you an end-to-end loop for vision model development:

- Creates a project around a dataset and goal.
- Uploads an image folder into local object storage.
- Profiles the dataset for class balance, image counts, image dimensions, metadata, annotations, and possible preprocessing issues.
- Creates experiment plans from the dataset profile and project goal.
- Runs training jobs through Python workers.
- Supports local workers for single-machine runs and Modal workers for cloud GPU runs.
- Tracks training metrics, evaluations, and artifacts.
- Reviews completed experiments and selects a champion model.
- Exports the champion for downstream use.
- Provides a Mission Control UI for monitoring jobs, workers, plans, decisions, metrics, exports, and demo predictions.

The default first-run mode is conservative and local-first. Agentic/OpenAI and Modal cloud features are optional.

## How It Works

Model Express is split into four main pieces:

| Component | Location | Purpose |
| --- | --- | --- |
| Mission Control | `apps/mission-control` | Electron desktop UI. Starts workers, uploads datasets, shows projects, plans, jobs, results, exports, and demos. |
| Orchestrator | `services/orchestrator` | Go API server. Owns projects, datasets, jobs, workers, plans, automation settings, run state, champion selection, and DB persistence. |
| Worker | `services/worker` | Python training/runtime worker. Profiles datasets, trains models, reports metrics, evaluates runs, exports champions, and runs demo inference. |
| Local services | `infra/compose.yaml` | Postgres, MLflow, and MinIO. Postgres stores app state, MLflow tracks runs/artifacts, and MinIO stores uploaded datasets and generated artifacts. |

Basic flow:

```text
User creates project in Mission Control
        |
        v
Mission Control uploads dataset folder to MinIO
        |
        v
Orchestrator records project, dataset, jobs, plans, and state in Postgres
        |
        v
Python worker polls orchestrator for jobs
        |
        v
Worker profiles data, trains models, reports metrics/evaluations, exports artifacts
        |
        v
Mission Control displays progress, decisions, champion model, exports, and demo predictions
```

Optional cloud flow:

```text
Mission Control starts a Modal dispatcher worker
        |
        v
Mission Control creates temporary HTTPS tunnels for orchestrator and local MinIO
        |
        v
Modal GPU containers train remotely and call back to the local orchestrator
```

For Modal, the local tunnel target is HTTP, for example `http://127.0.0.1:8080`, but the URL passed to Modal must be public HTTPS, for example `https://...trycloudflare.com`.

## Repository Layout

```text
model-express/
+-- apps/mission-control/        Electron + React desktop UI
+-- services/orchestrator/       Go API server and Postgres-backed orchestration state
+-- services/worker/             Python worker package for profiling, training, export, and demo inference
+-- infra/                       Docker Compose definitions for Postgres, MLflow, and MinIO
+-- scripts/                     Development and smoke-test helpers
+-- schemas/                     Shared schema notes
+-- datasets/                    Local ignored dataset workspace placeholder
+-- artifacts/                   Local ignored logs/artifact workspace
+-- exports/                     Local ignored export workspace
+-- .env.v1.local.example        Recommended local-first environment profile
+-- .env.v1.cloud.example        Cloud/Modal-oriented environment profile
+-- .env.example                 Older/common environment example
```

## Prerequisites

Install these before starting from a fresh clone:

| Dependency | Why it is needed | Check command |
| --- | --- | --- |
| Git | Clone the repository. | `git --version` |
| Docker Desktop | Runs Postgres, MLflow, and MinIO through Docker Compose. | `docker version` |
| Docker Compose | Included with modern Docker Desktop. | `docker compose version` |
| Go | Runs the orchestrator. Use the version from `services/orchestrator/go.mod` or newer. | `go version` |
| Node.js + npm | Runs Mission Control in development mode. Node 22+ is recommended. | `node --version` and `npm --version` |
| Python 3.11+ | Runs the worker package. | `python --version` or `py -3.11 --version` |

Optional cloud dependencies:

| Dependency | Needed when |
| --- | --- |
| OpenAI API key | You want LLM/agent planning, visual analysis, memory embeddings, or agent review. |
| Modal account and auth | You want cloud GPU workers. |
| `cloudflared` on PATH | You want Mission Control to create automatic HTTPS tunnels for Modal workers against local services. |

## Dataset Format

For the smoothest first run, start with an image classification dataset laid out by class folder:

```text
my-dataset/
+-- class-a/
|   +-- image-001.jpg
|   +-- image-002.jpg
+-- class-b/
|   +-- image-001.jpg
|   +-- image-002.jpg
+-- class-c/
    +-- image-001.jpg
```

Use normal image files such as `.jpg`, `.jpeg`, `.png`, or `.webp`. Avoid starting with a huge dataset on the first run. A small smoke dataset with 2-5 classes and a few dozen images per class is enough to verify the system.

Object detection / YOLO-style datasets are supported by the worker path, but classification is the recommended first setup test because it has fewer moving pieces.

## Fresh Clone Setup

The steps below assume you are running from the repo root unless a command says otherwise.

### 1. Clone the repository

Replace the URL with your repo URL if you publish this under a different owner/name.

Windows PowerShell:

```powershell
git clone https://github.com/YOUR_USERNAME/model-express.git
cd model-express
```

macOS/Linux:

```bash
git clone https://github.com/YOUR_USERNAME/model-express.git
cd model-express
```

### 2. Create your local environment file

Start with the safe local profile. It disables autonomous execution and cloud features by default.

Windows PowerShell:

```powershell
Copy-Item .env.v1.local.example .env.local
```

macOS/Linux:

```bash
cp .env.v1.local.example .env.local
```

You can open `.env.local` and review it, but you should not need to edit it for a local-first run.

The important defaults are:

```env
MODEL_EXPRESS_ORCHESTRATOR_ADDR=127.0.0.1:8080
MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER=local
MODEL_EXPRESS_EXECUTION_PROFILE=safe-local
MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS=false
MODEL_EXPRESS_AUTO_EXECUTE_PLANS=false
MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS=false
MODEL_EXPRESS_MAX_AUTO_WORKERS=1
MODEL_EXPRESS_LOCAL_MAX_CONCURRENT_JOBS=1
MODEL_EXPRESS_LLM_ENABLED=false
MODEL_EXPRESS_AUTOML_ENABLED=false
```

Notes:

- `.env.local` is gitignored.
- The orchestrator and Mission Control load `.env` and `.env.local` automatically from the repo root.
- If `MODEL_EXPRESS_ENV_FILE` is set, the app loads only that file instead.

### 3. Start local support services

Start Docker Desktop first. Wait until Docker says it is running.

Then run:

```powershell
docker compose -f infra/compose.yaml -f infra/compose.local-safe.yaml up -d
```

This starts:

| Service | URL/port | Purpose |
| --- | --- | --- |
| Postgres + pgvector | `127.0.0.1:5432` | Orchestrator database. |
| MLflow | `http://127.0.0.1:5000` | Training run tracking. |
| MinIO API | `http://127.0.0.1:9000` | Dataset and artifact object storage. |

Check the services:

```powershell
docker compose -f infra/compose.yaml ps
```

If a container is still starting, wait a few seconds and run the command again.

### 4. Install the Python worker dependencies

Mission Control starts workers with `python -m worker.main` from `services/worker`. It first looks for `services/worker/.venv`, so create that virtual environment explicitly.

Windows PowerShell:

```powershell
cd services\worker
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -c "import worker; print('worker import ok')"
cd ..\..
```

macOS/Linux:

```bash
cd services/worker
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -c "import worker; print('worker import ok')"
cd ../..
```

If `python -m pip install -e .` takes a while, that is expected. The worker installs ML and training dependencies including PyTorch, torchvision, Ultralytics, MLflow, Modal, ONNX, and scikit-learn.

If Python cannot find the worker later, set `MODEL_EXPRESS_PYTHON` in `.env.local` to the full path of the venv Python executable.

Example for Windows:

```env
MODEL_EXPRESS_PYTHON=C:\Users\you\path\to\model-express\services\worker\.venv\Scripts\python.exe
```

### 5. Install Mission Control dependencies

```powershell
cd apps\mission-control
npm install
cd ..\..
```

macOS/Linux:

```bash
cd apps/mission-control
npm install
cd ../..
```

### 6. Start the orchestrator

Open a new terminal from the repo root.

```powershell
cd services\orchestrator
go run ./cmd/orchestrator
```

macOS/Linux:

```bash
cd services/orchestrator
go run ./cmd/orchestrator
```

Expected behavior:

- The orchestrator loads `.env.local` from the repo root.
- It connects to Postgres.
- It applies embedded migrations automatically.
- It listens on `127.0.0.1:8080` by default.

You should see a log line similar to:

```text
starting orchestrator addr=127.0.0.1:8080
```

Keep this terminal open.

In another terminal, verify health:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/healthz
```

macOS/Linux:

```bash
curl http://127.0.0.1:8080/healthz
```

### 7. Start Mission Control

Open another terminal from the repo root.

Windows PowerShell:

```powershell
cd apps\mission-control
npm run dev
```

macOS/Linux:

```bash
cd apps/mission-control
npm run dev
```

This starts Vite and opens the Electron desktop app. Leave this terminal open while using the app.

### 8. Optional: run the API smoke script

With Docker services and the orchestrator running, you can run the lightweight smoke script from the repo root:

```powershell
.\scripts\orchestrator_smoke.ps1
```

This checks health, creates a project, creates a queued job, registers a worker, reports a fake metric, and reads the resulting state. It does not train a real model.

## How To Use The System

### 1. Start the app stack

Every time you want to use Model Express locally:

1. Start Docker Desktop.
2. Start support services:

   ```powershell
   docker compose -f infra/compose.yaml -f infra/compose.local-safe.yaml up -d
   ```

3. Start the orchestrator in one terminal:

   ```powershell
   cd services\orchestrator
   go run ./cmd/orchestrator
   ```

4. Start Mission Control in another terminal:

   ```powershell
   cd apps\mission-control
   npm run dev
   ```

If the app cannot connect, check that `http://127.0.0.1:8080/healthz` responds.

### 2. Create a project and upload a dataset

In Mission Control:

1. Click the new project / new mission control.
2. Enter a mission name.
   Example: `Surface defect classifier`.
3. Enter a goal.
   Example: `Optimize balanced accuracy while keeping inference lightweight`.
4. Click `Choose Folder & Upload`.
5. Select your dataset folder.
6. Review the dataset preflight result.
   - If it says ready, continue.
   - If it warns about file count, size, or invalid files, decide whether to fix the dataset or continue.
7. Click `Create Project`.

What happens behind the scenes:

1. Mission Control creates the project through the orchestrator.
2. Mission Control zips the selected dataset folder.
3. Mission Control uploads the ZIP to local MinIO.
4. Mission Control records the dataset in the orchestrator.
5. Mission Control starts a profiling worker.
6. The worker profiles the dataset and reports the result back to the orchestrator.

Wait until the dataset status becomes `PROFILED`. The Datasets view will then show class balance, counts, metadata status, and dataset intelligence.

### 3. Review the dataset

Open the Datasets tab and inspect:

- Dataset status.
- Class distribution.
- Image count and size information.
- Metadata import status.
- Visual analysis coverage if enabled.
- Warnings about corrupted images, imbalance, annotations, or preprocessing needs.

Do not start with large cloud runs until the dataset profile looks sane.

### 4. Review or execute an experiment plan

Open the Mission or In Depth plan area.

A plan contains:

- Target metric.
- Recommended worker count.
- Estimated runtime.
- Candidate experiments.
- Model choices.
- Preprocessing and augmentation settings.
- Backend validation warnings.

For local-first usage:

1. Keep automation settings conservative.
2. Review the plan first.
3. Click `Execute Plan` when you are ready.
4. Mission Control queues training jobs and starts local worker processes.
5. Watch jobs move from queued to assigned/running/completed.

If workers are not running, click `Resume Work`. Mission Control will try to restart worker processes for open project work.

### 5. Monitor training

During execution, watch these areas:

- Activity stream: high-level run progress.
- Jobs: queued, running, succeeded, failed jobs.
- Workers: active local or Modal workers.
- Metrics: epoch metrics and final metrics.
- Evaluations: run-level evaluation summaries.
- Agent decisions: review decisions and follow-up reasoning when agent features are enabled.

For local runs, training speed depends on your machine. CPU-only runs can be slow. Start with small datasets and low worker counts.

### 6. Review completed runs and select a champion

After jobs complete, Model Express compares successful runs and records champion information. The champion view shows:

- Selected model/job.
- Primary score.
- Supporting metrics.
- Selection reason.
- Comparison against other runs.
- Deployment/export status.

If no champion appears, check:

- Did at least one training job succeed?
- Did the worker report a training run summary and evaluation?
- Are jobs still running or queued?
- Did the plan get cancelled or fail?

### 7. Schedule follow-up experiments

If you want another round:

1. Click `Review Experiments` to have the reviewer inspect completed runs.
2. If the reviewer recommends more work, click `Schedule Follow-up`.
3. Mission Control queues a follow-up plan.
4. Execute the follow-up plan.

In the safe local profile, auto-followups and auto-execution are off by default. You stay in control.

### 8. Export and test the champion

When a champion is available:

1. Open the Test / Export view.
2. Request an ONNX export.
3. Mission Control queues the export job and starts a worker if needed.
4. Wait for export status to become ready.
5. Use the demo prediction controls to run sample images or a custom image.
6. Save/export the portable bundle when available.

The demo path can use browser ONNX runtime for compatible exports, or the local Python worker runtime for export formats that need Python-side inference.

## Optional Cloud / Modal Setup

Use the local setup first. Add Modal only after a local project can upload, profile, and run at least a small job.

Cloud mode needs more moving pieces:

- Modal credentials.
- OpenAI key if using agent/LLM features.
- Public HTTPS callback URL for the orchestrator.
- Public HTTPS S3 endpoint or a temporary tunnel for local MinIO.
- `cloudflared` installed if you want Mission Control automatic tunnels.

### Modal prerequisites

1. Create a Modal account.
2. Authenticate Modal locally using Modal's setup flow, or set:

   ```env
   MODAL_TOKEN_ID=replace-with-modal-token-id
   MODAL_TOKEN_SECRET=replace-with-modal-token-secret
   ```

3. Install `cloudflared` and ensure this works:

   ```powershell
   cloudflared --version
   ```

4. Add OpenAI if using LLM features:

   ```env
   OPENAI_API_KEY=replace-with-openai-api-key
   ```

### Use the cloud example profile

Copy the cloud profile only when you are ready to test Modal:

Windows PowerShell:

```powershell
Copy-Item .env.v1.cloud.example .env.v1.cloud
```

macOS/Linux:

```bash
cp .env.v1.cloud.example .env.v1.cloud
```

Then edit `.env.v1.cloud` and fill in at least:

```env
MODAL_TOKEN_ID=replace-with-modal-token-id
MODAL_TOKEN_SECRET=replace-with-modal-token-secret
OPENAI_API_KEY=replace-with-openai-api-key
```

Start the orchestrator and Mission Control with that env file selected:

Windows PowerShell:

```powershell
$env:MODEL_EXPRESS_ENV_FILE='.env.v1.cloud'
```

macOS/Linux:

```bash
export MODEL_EXPRESS_ENV_FILE=.env.v1.cloud
```

Then start the orchestrator and Mission Control from terminals that have that variable set.

### Automatic tunnels

When Modal workers are selected and no explicit public URLs are configured, Mission Control can create temporary Cloudflare tunnels:

- Local orchestrator target: `http://127.0.0.1:8080`
- Local MinIO target: `http://127.0.0.1:9000`
- Modal receives public HTTPS URLs generated by `cloudflared`

Do not expose Postgres, MLflow, or the MinIO console to Modal. The app is designed to tunnel only the orchestrator API and MinIO API when needed.

## Configuration Guide

Start with `.env.v1.local.example`. These are the settings most users should understand first:

| Variable | First-run value | Meaning |
| --- | --- | --- |
| `MODEL_EXPRESS_ORCHESTRATOR_ADDR` | `127.0.0.1:8080` | Address for the Go API server. |
| `MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER` | `local` | Use local Python workers by default. |
| `MODEL_EXPRESS_EXECUTION_PROFILE` | `safe-local` | Conservative single-machine behavior. |
| `GPU_TYPE` | `local` | Worker type label for local workers. |
| `MODEL_EXPRESS_MAX_AUTO_WORKERS` | `1` | Avoids launching too many workers on a laptop. |
| `MODEL_EXPRESS_LOCAL_MAX_CONCURRENT_JOBS` | `1` | One local training job at a time. |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | Local MLflow server. |
| `S3_ENDPOINT_URL` | `http://127.0.0.1:9000` | Local MinIO API. |
| `S3_BUCKET` | `model-express` | Bucket used for datasets/artifacts. |
| `MODEL_EXPRESS_LLM_ENABLED` | `false` | Keeps OpenAI/agent features off for first run. |

Do not start by copying every variable from a development `.env.local`. Most knobs are advanced tuning flags and are not needed for a new user.

## Useful Commands

Start services:

```powershell
docker compose -f infra/compose.yaml -f infra/compose.local-safe.yaml up -d
```

Stop services:

```powershell
docker compose -f infra/compose.yaml -f infra/compose.local-safe.yaml down
```

Check orchestrator health:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/healthz
```

Run orchestrator tests:

```powershell
cd services\orchestrator
go test ./...
```

Build Mission Control:

```powershell
cd apps\mission-control
npm run build
```

Run worker Modal provider tests:

```powershell
cd services\worker
python -m pytest tests/test_training_modal_provider.py -q
```

Run API smoke test:

```powershell
.\scripts\orchestrator_smoke.ps1
```

## Troubleshooting

### Orchestrator fails to start with a Postgres connection error

Postgres is not running or Docker Desktop is not ready.

Fix:

```powershell
docker compose -f infra/compose.yaml -f infra/compose.local-safe.yaml up -d
docker compose -f infra/compose.yaml ps
```

Then restart the orchestrator.

### Mission Control opens but cannot load projects

Check the orchestrator health endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/healthz
```

If that fails, restart the orchestrator. If it succeeds, make sure Mission Control is using `http://127.0.0.1:8080` as its base URL.

### Worker does not start

The most common cause is missing Python worker dependencies.

Check:

```powershell
cd services\worker
.\.venv\Scripts\Activate.ps1
python -c "import worker; print('worker import ok')"
```

If that fails, reinstall:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

### Dataset upload fails

Check that MinIO is running:

```powershell
docker compose -f infra/compose.yaml ps
```

Also check dataset size and file count. Start with a small dataset first.

### Modal job says localhost or HTTP is not allowed

Modal cannot call `http://localhost:8080` on your computer. That address is only valid as the local tunnel target.

For Modal workers, the callback URL must be public HTTPS. Use Mission Control automatic tunnels with `cloudflared`, or set:

```env
MODAL_ORCHESTRATOR_URL=https://your-public-orchestrator-url.example.com
MODAL_S3_ENDPOINT_URL=https://your-public-s3-url.example.com
```

### Modal preflight fails because `cloudflared` is missing

Install `cloudflared` and make sure `cloudflared --version` works. If it is installed but not on PATH, set:

```env
MODEL_EXPRESS_CLOUDFLARED_PATH=C:\path\to\cloudflared.exe
```

### Local training is slow

That is expected on CPU or low-end GPUs. Use a smaller dataset, fewer epochs, one worker, and a lightweight model for the first run. Modal GPU workers are the intended path for faster cloud execution.

## Development Notes

- Generated datasets, artifacts, exports, local DB backups, logs, `.env.local`, virtual environments, and node modules are gitignored.
- The root README should stay user-facing. Planning docs, agent context dumps, and private release notes should not be part of the public release surface.
- Before publishing, replace `YOUR_USERNAME/model-express.git` with the final public repository URL.
- Add screenshots later near the top of the README after the project summary.

## License

Add the project license before public release.
