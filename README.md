# Model Express

Model Express is a cloud-first training workbench for computer vision experiments. The intended way to use it is from the desktop Mission Control app while training jobs run on Modal GPUs.

You bring an image dataset and a goal. Model Express uploads the dataset, profiles it, builds experiment plans, launches cloud training jobs on Modal, compares the results, selects a champion model, exports it, and lets you test predictions from the UI.

This project is distributed as source code instead of a packaged executable. That is intentional for now: setup is explicit, the moving pieces are visible, and users can inspect or modify the system before running cloud jobs.

Add screenshots here before public release.

This is an early source release, not a bug-free commercial product. If you encounter a reproducible bug, open an issue in the repository's GitHub Issues tab with your operating system, setup path, what you expected, what happened, and the smallest reproduction/log snippet you can share. For feature requests or workflow ideas, start a GitHub Discussion if Discussions are enabled; otherwise open an issue with `Feature request:` in the title and describe the use case, why it matters, and the outcome you want. For usage questions, use GitHub Discussions if enabled. Do not include API keys, Modal tokens, `.env` contents, tunnel URLs, or private dataset files.

## What This Project Does

Model Express gives you an end-to-end cloud training loop for vision model development:

- Creates a project around a dataset and goal.
- Uploads an image folder into local object storage.
- Profiles the dataset for class balance, image counts, image dimensions, metadata, annotations, and possible preprocessing issues.
- Creates experiment plans from the dataset profile and project goal.
- Runs training jobs on Modal GPU workers.
- Tracks training metrics, evaluations, decisions, and artifacts.
- Reviews completed experiments and selects a champion model.
- Exports the champion for downstream use.
- Provides a Mission Control UI for monitoring jobs, workers, plans, decisions, metrics, exports, and demo predictions.

Local-only execution exists for smoke tests and development, but it is not the main product path. CPU or laptop training is usually slow and can hide the value of the system. For normal usage, use the cloud profile and Modal.

## How It Works

Model Express has a local control plane and a cloud execution plane.

| Component | Location | Purpose |
| --- | --- | --- |
| Mission Control | `apps/mission-control` | Electron desktop UI. Uploads datasets, starts workers, manages preflight, shows projects, plans, jobs, results, exports, and demos. |
| Orchestrator | `services/orchestrator` | Go API server. Owns projects, datasets, jobs, workers, plans, automation settings, run state, champion selection, and DB persistence. |
| Worker / Dispatcher | `services/worker` | Python runtime. In cloud mode it starts a Modal dispatcher, submits Modal jobs, and reports results back to the orchestrator. |
| Modal GPU workers | Modal cloud | Remote containers that run the actual training, profiling, evaluation, and export work. |
| Local services | `infra/compose.yaml` | Postgres, MLflow, and MinIO. Postgres stores app state, MLflow tracks runs/artifacts, and MinIO stores uploaded datasets and generated artifacts. |

Cloud-first flow:

```text
User creates project in Mission Control
        |
        v
Mission Control uploads the dataset folder to local MinIO
        |
        v
Orchestrator records project, dataset, jobs, plans, and state in Postgres
        |
        v
Mission Control starts a Modal dispatcher worker
        |
        v
Mission Control creates temporary HTTPS tunnels for orchestrator and MinIO
        |
        v
Modal GPU containers train remotely and call back to the local orchestrator
        |
        v
Mission Control displays progress, metrics, decisions, champion model, exports, and demo predictions
```

The local orchestrator and MinIO services listen on local HTTP URLs such as `http://127.0.0.1:8080` and `http://127.0.0.1:9000`. Modal cannot reach those directly. In cloud mode, Mission Control creates public HTTPS tunnel URLs and passes those HTTPS URLs to Modal.

For Modal, this distinction matters:

```text
Local tunnel target:  http://127.0.0.1:8080
Modal callback URL:  https://...trycloudflare.com
```

The first URL is allowed as the local target. The second URL is the one Modal workers must receive.

## Repository Layout

```text
model-express/
+-- apps/mission-control/        Electron + React desktop UI
+-- services/orchestrator/       Go API server and Postgres-backed orchestration state
+-- services/worker/             Python worker package for Modal dispatch, profiling, training, export, and demo inference
+-- infra/                       Docker Compose definitions for Postgres, MLflow, and MinIO
+-- scripts/                     Development and smoke-test helpers
+-- schemas/                     Shared schema notes
+-- datasets/                    Local ignored dataset workspace placeholder
+-- artifacts/                   Local ignored logs/artifact workspace
+-- exports/                     Local ignored export workspace
+-- .env.v1.cloud.example        Recommended cloud-first environment profile
+-- .env.v1.local.example        Local-only fallback profile for smoke tests and development
+-- .env.example                 Older/common environment example
```

## Prerequisites

Install these before starting from a fresh clone.

| Dependency | Why it is needed | Check command |
| --- | --- | --- |
| Git | Clone the repository. | `git --version` |
| Docker Desktop | Runs Postgres, MLflow, and MinIO through Docker Compose. | `docker version` |
| Docker Compose | Included with modern Docker Desktop. | `docker compose version` |
| Go | Runs the orchestrator. Use the version from `services/orchestrator/go.mod` or newer. | `go version` |
| Node.js + npm | Runs Mission Control in development mode. Node 22+ is recommended. | `node --version` and `npm --version` |
| Python 3.11+ | Runs the worker/Modal dispatcher package. | `python --version` or `py -3.11 --version` |
| Modal account | Runs cloud GPU jobs. | Modal dashboard account access |
| `cloudflared` | Lets Mission Control create HTTPS tunnels for Modal callbacks. | `cloudflared --version` |
| OpenAI API key | Enables the agentic planning/review path used by the cloud profile. | Key available from your OpenAI account |

Modal's own getting started docs currently recommend installing the Modal Python package and running `modal setup` or `python -m modal setup` to authenticate: https://modal.com/docs/guide

## Dataset Format

Model Express supports image classification datasets and YOLO-style object detection datasets. For the first Modal smoke test, use a small dataset so you can verify upload, profiling, planning, Modal dispatch, metrics, champion selection, and export before spending more GPU budget.

Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.

### Image Classification: Direct Class Folders

The simplest classification layout is one folder per class:

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

Each immediate child folder is treated as a class name. Use readable folder names because those names become class labels in the profile and training run.

### Image Classification: Explicit Train/Val/Test Splits

Classification datasets can also be split first, then class folders inside each split:

```text
my-dataset/
+-- train/
|   +-- class-a/
|   |   +-- image-001.jpg
|   +-- class-b/
|       +-- image-001.jpg
+-- val/
|   +-- class-a/
|   |   +-- image-002.jpg
|   +-- class-b/
|       +-- image-002.jpg
+-- test/
    +-- class-a/
    |   +-- image-003.jpg
    +-- class-b/
        +-- image-003.jpg
```

Accepted split folder names:

| Canonical split | Accepted folder names |
| --- | --- |
| `train` | `train`, `training` |
| `val` | `val`, `valid`, `validation`, `dev` |
| `test` | `test`, `testing`, `holdout`, `heldout` |

The same class names should appear consistently across splits. Missing classes in a split are allowed, but they may lead to weaker validation or test results.

### Image Classification: Common Image Root or Wrapper Folder

Model Express can also detect a class-folder dataset under a common image root:

```text
my-dataset/
+-- images/
    +-- class-a/
    |   +-- one.jpg
    +-- class-b/
        +-- two.jpg
```

Common image root folder names include `images`, `image`, `imgs`, `img`, `jpegimages`, and `data`.

It can also unwrap a single top-level wrapper folder, which is common in downloaded datasets:

```text
downloaded-dataset/
+-- dataset-name/
    +-- images/
        +-- class-a/
        +-- class-b/
```

Avoid naming class folders `labels`, `annotations`, `metadata`, `splits`, `parts`, `bboxes`, or similar metadata names. Those are intentionally treated as annotation/metadata folders instead of class folders.

### YOLO Object Detection: Recommended Layout

For object detection training, use a standard YOLO dataset with a data config file plus parallel `images/` and `labels/` folders:

```text
my-yolo-dataset/
+-- data.yaml
+-- images/
|   +-- train/
|   |   +-- image-001.jpg
|   |   +-- image-002.jpg
|   +-- val/
|   |   +-- image-003.jpg
|   +-- test/
|       +-- image-004.jpg
+-- labels/
    +-- train/
    |   +-- image-001.txt
    |   +-- image-002.txt
    +-- val/
    |   +-- image-003.txt
    +-- test/
        +-- image-004.txt
```

A minimal `data.yaml` should look like this:

```yaml
path: .
train: images/train
val: images/val
test: images/test
nc: 2
names: [cat, dog]
```

`test` is optional, but `train` and `val` are required for training. `valid` is accepted as an alias for `val` in YOLO configs.

Accepted YOLO config filenames:

- `data.yaml`
- `data.yml`
- `dataset.yaml`
- `dataset.yml`

The config can be at the dataset root or nested inside the uploaded folder. For least confusion, put it at the dataset root and select that root folder during upload.

### YOLO Label Files

Each YOLO label file should match the image filename stem:

```text
images/train/image-001.jpg
labels/train/image-001.txt
```

Each non-empty label line should use normalized YOLO box format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Example:

```text
0 0.5000 0.5000 0.2500 0.3000
1 0.2500 0.2500 0.1000 0.1000
```

Rules:

- `class_id` is zero-based and must match the order in `names`.
- Box coordinates should be normalized from `0.0` to `1.0`.
- `width` and `height` must be greater than zero.
- Empty `.txt` files are allowed for images with no objects.
- If an image has objects, keep the label file in the matching `labels/<split>/` folder.

### YOLO Config Variants That Work

Class names can be written as an inline list:

```yaml
names: [cat, dog]
```

Or as a YAML list:

```yaml
names:
  - cat
  - dog
```

Or as an id-to-name mapping:

```yaml
names:
  0: cat
  1: dog
```

YOLO split paths can be relative to `path`, relative to the config file, or relative to the uploaded dataset root. The worker also tries to normalize stale absolute paths from downloaded archives when the matching folders exist inside the uploaded dataset.

Do not use YOLO config paths that point outside the uploaded dataset, use parent traversal such as `..`, or reference remote URIs such as `s3://`, `gs://`, `http://`, or `https://`. Modal training expects the dataset files to be inside the uploaded dataset materialization.

### YOLO Profiling vs YOLO Training

Model Express can often detect YOLO labels even when no data config exists. That is useful for profiling, but it is not enough for actual YOLO training.

For Modal YOLO training, include a valid `data.yaml`, `data.yml`, `dataset.yaml`, or `dataset.yml` with:

- `train`
- `val` or `valid`
- `names` or `nc`

COCO JSON, Pascal VOC XML, and other annotation files may be detected as metadata, but the YOLO training path expects YOLO-format `.txt` labels and a YOLO data config. Convert those datasets to YOLO format before using them for object detection training.

## Fresh Clone Setup: Cloud / Modal Path

The steps below are the recommended setup path. They configure Model Express to use Modal by default.

Run commands from the repo root unless a step says otherwise.

### 1. Clone the repository

Replace the URL with the final public repo URL before publishing.

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

### 2. Create the cloud environment file

Copy the cloud-first profile:

Windows PowerShell:

```powershell
Copy-Item .env.v1.cloud.example .env.v1.cloud
```

macOS/Linux:

```bash
cp .env.v1.cloud.example .env.v1.cloud
```

Open `.env.v1.cloud` in an editor.

At minimum, do two things:

1. Set a real OpenAI key:

   ```env
   OPENAI_API_KEY=replace-with-openai-api-key
   ```

2. Choose one Modal authentication method.

Option A, env tokens:

```env
MODAL_TOKEN_ID=your-real-modal-token-id
MODAL_TOKEN_SECRET=your-real-modal-token-secret
```

Option B, Modal CLI config:

```env
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
```

Then authenticate after the worker virtual environment is installed in step 7. Modal will create a user config such as `C:\Users\you\.modal.toml` on Windows or `~/.modal.toml` on macOS/Linux.

Do not leave placeholder values in `.env.v1.cloud`. Placeholder strings are still non-empty strings, and Modal/OpenAI clients may try to use them as real credentials.

### 3. Review the cloud defaults

The cloud profile should keep these values:

```env
MODEL_EXPRESS_V1_PROFILE=cloud
MODEL_EXPRESS_EXECUTION_PROFILE=fast-remote
MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER=modal
MODEL_EXPRESS_DEFAULT_GPU_TYPE=T4
MODEL_EXPRESS_MODAL_DEFAULT_GPU_TYPE=T4
GPU_TYPE=modal
MODEL_EXPRESS_MODAL_TUNNEL_S3=true
MODEL_EXPRESS_AGENT_MODE=autonomous
MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS=true
MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS=true
MODEL_EXPRESS_AUTO_EXECUTE_PLANS=true
MODEL_EXPRESS_BUDGET_CAP_USD=5
```

For a first public run, keep the budget cap low. Increase it only after the full path works on a small dataset.

### 4. Install `cloudflared`

Mission Control uses `cloudflared` to create temporary HTTPS URLs that Modal can call back into.

After installing it, verify:

Windows PowerShell:

```powershell
cloudflared --version
```

macOS/Linux:

```bash
cloudflared --version
```

If `cloudflared` is installed but not on PATH, set this in `.env.v1.cloud`:

```env
MODEL_EXPRESS_CLOUDFLARED_PATH=C:\path\to\cloudflared.exe
```

Use the real path on your machine.

### 5. Start local support services

Even in cloud mode, the local app still needs Postgres, MLflow, and MinIO.

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

### 6. Install the Python worker / Modal dispatcher dependencies

Mission Control starts a Python worker process even for cloud mode. In cloud mode that process acts as the Modal dispatcher.

Create the worker virtual environment explicitly.

Windows PowerShell:

```powershell
cd services\worker
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -c "import worker; print('worker import ok')"
python -c "import modal; print('modal import ok')"
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
python -c "import modal; print('modal import ok')"
cd ../..
```

If `python -m pip install -e .` takes a while, that is expected. The worker installs ML and training dependencies including PyTorch, torchvision, Ultralytics, MLflow, Modal, ONNX, and scikit-learn.

If Mission Control later cannot find Python, set `MODEL_EXPRESS_PYTHON` in `.env.v1.cloud` to the full path of the venv Python executable.

Example for Windows:

```env
MODEL_EXPRESS_PYTHON=C:\Users\you\path\to\model-express\services\worker\.venv\Scripts\python.exe
```

### 7. Verify Modal auth from the worker venv

From the activated worker venv, run:

```powershell
python -m modal --version
```

If you use Modal CLI config instead of env tokens, run:

```powershell
python -m modal setup
```

That command should create or update a Modal user config such as `C:\Users\you\.modal.toml` on Windows or `~/.modal.toml` on macOS/Linux.

If you use env tokens instead, make sure `.env.v1.cloud` contains real token values and not placeholders, then skip `python -m modal setup`.

### 8. Install Mission Control dependencies

Windows PowerShell:

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

### 9. Start the orchestrator with the cloud profile

Open a new terminal from the repo root.

Windows PowerShell:

```powershell
$env:MODEL_EXPRESS_ENV_FILE='.env.v1.cloud'
cd services\orchestrator
go run ./cmd/orchestrator
```

macOS/Linux:

```bash
export MODEL_EXPRESS_ENV_FILE=.env.v1.cloud
cd services/orchestrator
go run ./cmd/orchestrator
```

Expected behavior:

- The orchestrator loads `.env.v1.cloud` from the repo root.
- It connects to Postgres.
- It applies embedded migrations automatically.
- It listens on `127.0.0.1:8080` by default.
- It reports the cloud profile during cloud preflight.

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

### 10. Start Mission Control with the same cloud profile

Open another terminal from the repo root.

Windows PowerShell:

```powershell
$env:MODEL_EXPRESS_ENV_FILE='.env.v1.cloud'
cd apps\mission-control
npm run dev
```

macOS/Linux:

```bash
export MODEL_EXPRESS_ENV_FILE=.env.v1.cloud
cd apps/mission-control
npm run dev
```

This starts Vite and opens the Electron desktop app. Leave this terminal open while using the app.

The environment variable matters. If you forget `MODEL_EXPRESS_ENV_FILE=.env.v1.cloud`, the app may run with local defaults instead of the Modal cloud profile.

### 11. Run cloud preflight before the first real dataset

Mission Control runs cloud checks before cloud-sensitive operations. If you see a preflight panel, fix failed checks before launching jobs.

The cloud profile expects:

- `MODEL_EXPRESS_V1_PROFILE=cloud`
- `MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER=modal`
- Modal authentication from env tokens or `~/.modal.toml`
- OpenAI key configured for the agentic workflow
- `cloudflared` available for automatic tunnels
- Public HTTPS tunnel URLs created for Modal callbacks
- S3/MinIO access available to Modal through the automatic MinIO API tunnel

Do not manually set `MODAL_ORCHESTRATOR_URL=http://localhost:8080`. Modal requires a public HTTPS URL. Let Mission Control create the tunnel unless you are intentionally using your own public HTTPS endpoint.

### 12. Optional: run the API smoke script

With Docker services and the orchestrator running, you can run the lightweight smoke script from the repo root:

```powershell
.\scripts\orchestrator_smoke.ps1
```

This checks health, creates a project, creates a queued job, registers a worker, reports a fake metric, and reads the resulting state. It does not train a real model and does not prove Modal training works. It only verifies the local API path.

## How To Use The System

### 1. Start the cloud stack

Every time you want to use Model Express in the intended mode:

1. Start Docker Desktop.
2. Start support services:

   ```powershell
   docker compose -f infra/compose.yaml -f infra/compose.local-safe.yaml up -d
   ```

3. Start the orchestrator in one terminal:

   ```powershell
   $env:MODEL_EXPRESS_ENV_FILE='.env.v1.cloud'
   cd services\orchestrator
   go run ./cmd/orchestrator
   ```

4. Start Mission Control in another terminal:

   ```powershell
   $env:MODEL_EXPRESS_ENV_FILE='.env.v1.cloud'
   cd apps\mission-control
   npm run dev
   ```

5. Confirm the orchestrator health endpoint responds:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8080/healthz
   ```

If the app behaves like local mode, close both terminals and restart them with `MODEL_EXPRESS_ENV_FILE=.env.v1.cloud` set.

### 2. Create a project and upload a dataset

In Mission Control:

1. Click `New Project`.
2. Enter a mission name.
   Example: `Surface defect classifier`.
3. Enter a goal.
   Example: `Optimize balanced accuracy while keeping inference lightweight`.
4. Click `Choose Folder & Upload`.
5. Select your dataset folder.
6. Review the dataset preflight result.
   - If it says ready, continue.
   - If it warns about file count, size, or invalid files, fix the dataset before spending Modal budget.
7. Click `Create Project`.

What happens behind the scenes:

1. Mission Control creates the project through the orchestrator.
2. Mission Control zips the selected dataset folder.
3. Mission Control uploads the ZIP to local MinIO.
4. Mission Control records the dataset in the orchestrator.
5. Mission Control starts cloud preflight as needed.
6. Mission Control starts the Modal dispatcher worker.
7. The dispatcher submits profiling/training work to Modal.
8. Modal workers report results back through the public HTTPS orchestrator tunnel.

Wait until the dataset status becomes `PROFILED`. The Datasets view will then show class balance, counts, metadata status, and dataset intelligence.

### 3. Review the dataset before spending more cloud budget

Open the Datasets tab and inspect:

- Dataset status.
- Class distribution.
- Image count and size information.
- Metadata import status.
- Visual analysis coverage if enabled.
- Warnings about corrupted images, imbalance, annotations, or preprocessing needs.

Do not start larger Modal runs until the dataset profile looks sane. Bad labels or broken folder structure will waste GPU time.

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

In the cloud profile, Modal should be the default training provider.

To run the plan:

1. Confirm the plan is using Modal or a Modal GPU type.
2. Confirm the budget cap is acceptable.
3. Click `Execute Plan`.
4. Mission Control runs cloud preflight if needed.
5. Mission Control creates HTTPS tunnels for the orchestrator and MinIO API.
6. Mission Control starts the Modal dispatcher.
7. The dispatcher submits remote training jobs to Modal.
8. Watch jobs move from queued to assigned/running/completed.

If workers are not running, click `Resume Work`. Mission Control will try to restart the Modal dispatcher for open project work.

### 5. Monitor Modal training

During execution, watch these areas:

- Activity stream: high-level run progress.
- Jobs: queued, running, succeeded, failed jobs.
- Workers: local dispatcher and remote Modal activity.
- Metrics: epoch metrics and final metrics.
- Evaluations: run-level evaluation summaries.
- Agent decisions: review decisions and follow-up reasoning.
- Budget/cost indicators: make sure the run stays within your intended cap.

Training speed and cost depend on dataset size, model choice, GPU type, and number of follow-up rounds. Start with T4 and a small dataset before using larger GPUs.

### 6. Review completed runs and select a champion

After jobs complete, Model Express compares successful runs and records champion information. The champion view shows:

- Selected model/job.
- Primary score.
- Supporting metrics.
- Selection reason.
- Comparison against other runs.
- Deployment/export status.

If no champion appears, check:

- Did at least one Modal training job succeed?
- Did the worker report a training run summary and evaluation?
- Are jobs still running or queued?
- Did cloud preflight fail?
- Did the plan get cancelled or fail?

### 7. Let the agent schedule follow-up experiments

In the cloud profile, follow-up automation is enabled by default:

```env
MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS=true
MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS=true
MODEL_EXPRESS_AUTO_EXECUTE_PLANS=true
MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS=5
```

You can still review what happened in the UI.

Manual flow:

1. Click `Review Experiments` to have the reviewer inspect completed runs.
2. If the reviewer recommends more work, click `Schedule Follow-up`.
3. Mission Control queues a follow-up plan.
4. Execute the follow-up plan, or let cloud automation execute it when enabled.

For first-time users, keep `MODEL_EXPRESS_BUDGET_CAP_USD` low and raise it only after the workflow is proven.

### 8. Export and test the champion

When a champion is available:

1. Open the `Test / Export` view.
2. Request an ONNX export.
3. Mission Control queues the export job.
4. The Modal dispatcher starts export work when needed.
5. Wait for export status to become ready.
6. Use the demo prediction controls to run sample images or a custom image.
7. Save/export the portable bundle when available.

The demo path can use browser ONNX runtime for compatible exports, or the local Python worker runtime for export formats that need Python-side inference.

## Local-Only Mode

Local-only mode is available, but it is not the recommended user path.

Use local-only mode only when:

- You are developing the app.
- You need a quick smoke test without a Modal account.
- You are debugging local worker behavior.
- You intentionally want to avoid cloud cost.

To run local-only mode, copy the local profile:

Windows PowerShell:

```powershell
Copy-Item .env.v1.local.example .env.local
```

macOS/Linux:

```bash
cp .env.v1.local.example .env.local
```

Then start Docker services, the orchestrator, and Mission Control without setting `MODEL_EXPRESS_ENV_FILE`.

Local mode defaults to:

```env
MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER=local
MODEL_EXPRESS_EXECUTION_PROFILE=safe-local
MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS=false
MODEL_EXPRESS_AUTO_EXECUTE_PLANS=false
MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS=false
```

Expect local training to be slow on CPU-only machines. It is not the path this README optimizes for.

## Cloud Configuration Guide

Start with `.env.v1.cloud.example`. These are the settings most users should understand first:

| Variable | Recommended value | Meaning |
| --- | --- | --- |
| `MODEL_EXPRESS_V1_PROFILE` | `cloud` | Enables the cloud profile expected by preflight. |
| `MODEL_EXPRESS_ORCHESTRATOR_ADDR` | `127.0.0.1:8080` | Local address for the Go API server. |
| `MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE` | `true` | Allows authenticated public tunnel callbacks. |
| `MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER` | `modal` | Uses Modal workers by default. |
| `MODEL_EXPRESS_EXECUTION_PROFILE` | `fast-remote` | Cloud-oriented execution profile. |
| `MODEL_EXPRESS_DEFAULT_GPU_TYPE` | `T4` | Conservative first GPU choice. |
| `MODEL_EXPRESS_MODAL_DEFAULT_GPU_TYPE` | `T4` | Modal default GPU choice. |
| `MODEL_EXPRESS_MODAL_TUNNEL_S3` | `true` | Allows Mission Control to tunnel local MinIO API to Modal. |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | real values or blank | Use real env tokens, or leave blank when using Modal CLI config. |
| `OPENAI_API_KEY` | real value | Enables agentic planning/review and visual analysis. |
| `MODEL_EXPRESS_BUDGET_CAP_USD` | `5` for first run | Keeps the first cloud run bounded. |
| `MODEL_EXPRESS_MAX_AUTO_WORKERS` | `1` for first run | Avoids launching too much cloud work at once. |
| `MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS` | `5` by default | Controls autonomous follow-up depth. |

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

Start orchestrator in cloud mode:

```powershell
$env:MODEL_EXPRESS_ENV_FILE='.env.v1.cloud'
cd services\orchestrator
go run ./cmd/orchestrator
```

Start Mission Control in cloud mode:

```powershell
$env:MODEL_EXPRESS_ENV_FILE='.env.v1.cloud'
cd apps\mission-control
npm run dev
```

Check Modal from the worker venv:

```powershell
cd services\worker
.\.venv\Scripts\Activate.ps1
python -m modal --version
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

### The app is using local mode instead of Modal

Most likely `MODEL_EXPRESS_ENV_FILE=.env.v1.cloud` was not set in the terminal that started the orchestrator or Mission Control.

Fix:

1. Stop the orchestrator.
2. Stop Mission Control.
3. Restart both terminals with:

   ```powershell
   $env:MODEL_EXPRESS_ENV_FILE='.env.v1.cloud'
   ```

4. Start the orchestrator and Mission Control again.

### Modal authentication fails

Model Express checks Modal auth in this order:

1. `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
2. `MODAL_PROFILE`
3. `MODAL_CONFIG_PATH`
4. `~/.modal.toml`

Fix options:

- Put real token values in `.env.v1.cloud`.
- Or run `python -m modal setup` and blank out `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in `.env.v1.cloud`.
- Do not leave `replace-with-modal-token-id` or `replace-with-modal-token-secret` as values.

### Modal preflight fails because `cloudflared` is missing

Install `cloudflared` and make sure this works:

```powershell
cloudflared --version
```

If it is installed but not on PATH, set:

```env
MODEL_EXPRESS_CLOUDFLARED_PATH=C:\path\to\cloudflared.exe
```

### Modal job says localhost or HTTP is not allowed

Modal cannot call `http://localhost:8080` on your computer. That address is only valid as the local tunnel target.

For Modal workers, the callback URL must be public HTTPS. Use Mission Control automatic tunnels with `cloudflared`, or set:

```env
MODAL_ORCHESTRATOR_URL=https://your-public-orchestrator-url.example.com
MODAL_S3_ENDPOINT_URL=https://your-public-s3-url.example.com
```

Do not set either of those to `http://localhost:...`.

### Modal cannot reach MinIO or artifacts fail to upload

The cloud profile expects Mission Control to tunnel local MinIO's API port to Modal:

```env
MODEL_EXPRESS_MODAL_TUNNEL_S3=true
S3_ENDPOINT_URL=http://127.0.0.1:9000
```

Check:

- Docker services are running.
- MinIO is reachable locally at `http://127.0.0.1:9000`.
- `cloudflared` is installed.
- `MODEL_EXPRESS_MODAL_TUNNEL_S3=true` is present in `.env.v1.cloud`.
- You did not manually set `MODAL_S3_ENDPOINT_URL` to a local HTTP URL.

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

### Worker or Modal dispatcher does not start

The most common cause is missing Python worker dependencies.

Check:

```powershell
cd services\worker
.\.venv\Scripts\Activate.ps1
python -c "import worker; print('worker import ok')"
python -c "import modal; print('modal import ok')"
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

### Modal costs are higher than expected

Lower the first-run budget and parallelism:

```env
MODEL_EXPRESS_BUDGET_CAP_USD=2
MODEL_EXPRESS_MAX_AUTO_WORKERS=1
MODEL_EXPRESS_MODAL_DISPATCHER_SLOTS=1
MODEL_EXPRESS_MODAL_BATCH_MAX_TRIALS=1
MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS=1
```

Use T4 for initial tests before trying larger GPUs.

## Development Notes

- Generated datasets, artifacts, exports, local DB backups, logs, `.env.local`, `.env.v1.cloud`, virtual environments, and node modules are gitignored.
- The root README should stay user-facing and cloud-first.
- Planning docs, agent context dumps, and private release notes should not be part of the public release surface.
- Before publishing, replace `YOUR_USERNAME/model-express.git` with the final public repository URL.
- Add screenshots near the top of the README after the project summary.

## License

MIT License

Copyright (c) 2026 Model Express contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
