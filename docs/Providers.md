# Model Express Provider Guide

> Last updated: June 23, 2026  
> This guide explains the external runtimes Model Express can use: local services, workers, remote GPU training, LLM agents, embeddings, object storage, tunnels, and export/demo runtimes.

Model Express is designed to work in two main modes:

1. **Safe local demo** — local orchestrator, local Postgres/MinIO, local worker, LLMs off by default.
2. **Cloud agentic demo** — local Mission Control and orchestrator, Modal GPU workers, OpenAI agent runtime, S3-compatible storage, and optional automatic tunnels.

The provider system is intentionally simple: providers supply capabilities, but the Go backend still validates and schedules all executable work.

## Recommended Setup Order

| Step | Provider / Runtime | Required? | Unlocks |
| --- | --- | --- | --- |
| 1 | Docker + local Postgres/MinIO | Recommended for local demos | Durable control-plane state and local S3-compatible dataset/artifact storage |
| 2 | Local Python worker | Recommended | Dataset profiling, local smoke jobs, export/demo jobs, worker contract testing |
| 3 | OpenAI | Optional, required for agentic cloud demo | Experiment Planner, visual analysis, structured JSON recommendations, usage telemetry |
| 4 | Modal | Optional, required for real remote GPU training | Torchvision classifier training, Ultralytics YOLO detector training, remote profiling/export paths |
| 5 | S3-compatible storage | Required for dataset/artifact exchange | Local MinIO, AWS S3, Cloudflare R2, or another compatible object store |
| 6 | OpenAI embeddings | Optional | Vector memory retrieval for compact agent memory cards |
| 7 | ONNX Runtime Web / local demo runtime | Optional but useful | Browser-side champion demo inference when compatible artifacts exist |
| 8 | Cloudflare tunnels / public HTTPS endpoints | Optional, useful for Modal with local services | Allows Modal workers to call the local orchestrator and local MinIO safely |

Model Express does not require every provider for every workflow. For a recruiter-facing demo, the safest path is to show a local profile first, then show the cloud profile as the “real GPU + LLM automation” path.

## Quick Profiles

### Safe Local Profile

Use this when you want a predictable single-machine demo with no paid LLM or remote GPU requirement.

```bash
MODEL_EXPRESS_ENV_FILE=.env.v1.local
MODEL_EXPRESS_ORCHESTRATOR_ADDR=127.0.0.1:8080
MODEL_EXPRESS_ALLOW_LAN=false

MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER=local
MODEL_EXPRESS_EXECUTION_PROFILE=safe-local
GPU_TYPE=local

MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS=false
MODEL_EXPRESS_AUTO_EXECUTE_PLANS=false
MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS=false
MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS=1

S3_ENDPOINT_URL=http://127.0.0.1:9000
S3_BUCKET=model-express
AWS_DEFAULT_REGION=us-east-1

MODEL_EXPRESS_LLM_ENABLED=false
MODEL_EXPRESS_AUTOML_ENABLED=false
MODEL_EXPRESS_VISUAL_ANALYSIS_AUTOMATION=false
```

What this gives you:

- Local Mission Control.
- Local orchestrator.
- Local Postgres and MinIO through Docker Compose.
- Local dataset upload and profiling.
- Manual review flow.
- Low-risk demos without secret setup.

Tradeoff: local mode is not the main path for real GPU training. It is best for smoke tests, UI walkthroughs, and proving the orchestration contract.

### Cloud Agentic Profile

Use this when you want to demonstrate the full agentic loop with OpenAI and Modal.

```bash
MODEL_EXPRESS_ENV_FILE=.env.v1.cloud
MODEL_EXPRESS_V1_PROFILE=cloud
MODEL_EXPRESS_EXECUTION_PROFILE=fast-remote

MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER=modal
MODAL_TOKEN_ID=replace-with-modal-token-id
MODAL_TOKEN_SECRET=replace-with-modal-token-secret
MODEL_EXPRESS_DEFAULT_GPU_TYPE=T4
MODEL_EXPRESS_MODAL_DEFAULT_GPU_TYPE=T4
MODAL_GPU_TYPE=T4

OPENAI_API_KEY=replace-with-openai-api-key
MODEL_EXPRESS_LLM_ENABLED=true
MODEL_EXPRESS_LLM_PROVIDER=openai
MODEL_EXPRESS_LLM_MODEL=gpt-5.4-mini
MODEL_EXPRESS_LLM_API_STYLE=responses
MODEL_EXPRESS_LLM_STORED_RESPONSES=true
MODEL_EXPRESS_LLM_REASONING_EFFORT=medium
MODEL_EXPRESS_LLM_MAX_TOOL_ROUNDS=4

MODEL_EXPRESS_AGENT_MODE=autonomous
MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS=true
MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS=true
MODEL_EXPRESS_AUTO_EXECUTE_PLANS=true
MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS=5
MODEL_EXPRESS_BUDGET_CAP_USD=5
```

What this gives you:

- OpenAI-backed planning and visual-analysis agents.
- Modal-backed GPU training.
- Autonomous follow-up scheduling within backend limits.
- Usage/cost telemetry for LLM and embedding calls where available.
- A more realistic end-to-end release demo.

Tradeoff: this profile requires external credentials and can incur provider costs. Keep budgets and follow-up limits conservative for demos.

## Provider Summary

| Provider | Category | Used for | Default stance |
| --- | --- | --- | --- |
| Local worker | Execution | Profiling, smoke training, exports, demos | On for local demo |
| Modal | Remote GPU | Real classifier/detector training and remote execution | Optional, cloud profile |
| OpenAI | LLM | Planner, visual analysis, JSON recommendations | Optional, cloud profile |
| OpenAI-compatible | LLM | Hosted/self-managed APIs with OpenAI-like JSON behavior | Advanced/experimental |
| Local LLM | LLM | Future/local experimentation surface | Present as config boundary, not the main release path |
| OpenAI embeddings | Memory | Compact vector retrieval for agent memory | Optional |
| MinIO | Storage | Local S3-compatible storage | Local default |
| AWS S3 / R2 / compatible | Storage | Cloud dataset/artifact exchange | Bring-your-own storage |
| ONNX Runtime Web | Demo runtime | Browser-side champion inference | Optional but useful |
| Cloudflare tunnel / public HTTPS | Connectivity | Allows Modal to reach local orchestrator/storage | Optional, must be authenticated |

## Local Runtime Provider

The local runtime is the easiest way to run Model Express without external infrastructure.

It includes:

- Postgres for durable state.
- MinIO for local S3-compatible object storage.
- A local orchestrator bound to `127.0.0.1:8080` by default.
- A local Python worker managed manually or by Mission Control worker requirements.

Mission Control can start the local Docker Compose services for Postgres and MinIO. It also generates local runtime credentials and a local API token in app-local user data when needed.

### Key variables

```bash
DATABASE_URL=postgres://model_express:model_express@127.0.0.1:5432/model_express?sslmode=disable
S3_ENDPOINT_URL=http://127.0.0.1:9000
S3_BUCKET=model-express
MODEL_EXPRESS_ARTIFACT_BUCKET=model-express
MODEL_EXPRESS_ARTIFACT_PREFIX=model-express/artifacts
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### Best for

- Local recruiter demo.
- UI walkthroughs.
- Dataset upload and profiling demonstration.
- Safe manual-mode testing.
- Development without paid providers.

### Notes

- Use the MinIO API port, not the console port.
- Mission Control rejects dangerous dataset selections such as filesystem roots, home folders, and system directories.
- Non-loopback orchestrator and storage origins require explicit allowlists or tunnel mode.

## Local Worker Provider

The local worker is the base execution runtime. It registers with the orchestrator, polls for jobs, runs the assigned template, reports metrics/results, and marks jobs complete or failed.

### Key variables

```bash
ORCHESTRATOR_URL=http://localhost:8080
PROJECT_ID=optional-project-id
GPU_TYPE=local
MODEL_EXPRESS_WORKER_POLL_INTERVAL_SECONDS=5
MODEL_EXPRESS_WORKER_IDLE_LOG_SECONDS=60
```

### What it unlocks

- Dataset profiling.
- Local contract-style training jobs.
- Dataset visual jobs where configured.
- Champion export jobs when artifacts are available locally.
- Champion demo prediction jobs.
- Visual exemplar generation.

### Trust boundary

The worker does not decide what work exists. It polls the backend, receives one assigned job, executes that job, and reports back. Job failures are reported with retry-aware metadata when possible.

## Modal Provider

Modal is the primary remote GPU execution provider for real training runs.

Model Express uses Modal for:

- Torchvision classifier training.
- Ultralytics YOLO detector training.
- Remote dataset materialization.
- Remote profiling paths.
- Export paths that need training artifacts.
- Resource-aware execution telemetry.

### Required variables

```bash
MODAL_TOKEN_ID=replace-with-modal-token-id
MODAL_TOKEN_SECRET=replace-with-modal-token-secret
MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER=modal
MODEL_EXPRESS_DEFAULT_GPU_TYPE=T4
MODEL_EXPRESS_MODAL_DEFAULT_GPU_TYPE=T4
MODAL_GPU_TYPE=T4
```

### Optional resource variables

```bash
MODEL_EXPRESS_MODAL_DISPATCHER_SLOTS=1
MODEL_EXPRESS_MODAL_DISPATCHER_IDLE_POLL_SLOTS=1
MODEL_EXPRESS_MODAL_BATCH_MAX_TRIALS=2
MODEL_EXPRESS_MODAL_TRAINING_TIMEOUT_SECONDS=21600
MODEL_EXPRESS_MODAL_MATERIALIZATION_TIMEOUT_SECONDS=3600
MODEL_EXPRESS_MODAL_PROFILE_TIMEOUT_SECONDS=3600
MODEL_EXPRESS_MODAL_COST_SENSITIVE_DEFAULTS=false
```

### Supported GPU ladder

The Modal resource resolver recognizes this GPU ladder:

```text
T4 → L4 → A10 → L40S → A100-40GB → A100-80GB
```

Jobs carry requested/effective GPU, memory, batch-size, resource-attempt, and resource-signature metadata. OOM-like failures are classified as GPU CUDA, system memory, or unknown memory failures when possible.

### Supported training families

For image classification, the backend-supported catalog includes:

- `mobilenet_v3_small`
- `mobilenet_v3_large`
- `efficientnet_b0`
- `efficientnet_b1`
- `efficientnet_b2`
- `regnet_y_400mf`
- `resnet18`
- `resnet34`
- `convnext_tiny`
- `swin_t`
- `vit_b_16`

For YOLO object-detection datasets, the backend adds:

- `yolo11n.pt`
- `yolo11s.pt`
- `yolo11m.pt`
- `yolo11l.pt`
- `yolo11x.pt`

YOLO detector jobs require YOLO evidence in the dataset profile or metadata summary. Classifier models are rejected for object-detection datasets when the backend detects YOLO task evidence.

### Best for

- Real GPU-backed demos.
- Showing that the architecture can leave the local machine without rewriting the app.
- Training practical champion candidates.
- Demonstrating cost/resource telemetry and failure handling.

## OpenAI Provider

OpenAI is the main LLM provider for the release-oriented agentic workflow.

It is used for:

- Experiment Planner recommendations.
- Visual dataset analysis when enabled.
- Structured JSON outputs.
- Stored response workflows.
- Usage telemetry when returned by the provider.
- Bounded tool-round workflows for agent information gathering.

### Required variables

```bash
OPENAI_API_KEY=replace-with-openai-api-key
MODEL_EXPRESS_LLM_ENABLED=true
MODEL_EXPRESS_LLM_PROVIDER=openai
MODEL_EXPRESS_LLM_MODEL=gpt-5.4-mini
MODEL_EXPRESS_LLM_API_STYLE=responses
MODEL_EXPRESS_LLM_STORED_RESPONSES=true
```

You may also use `MODEL_EXPRESS_LLM_API_KEY` instead of `OPENAI_API_KEY` when you want a Model Express-specific key.

### Optional reasoning/tool variables

```bash
MODEL_EXPRESS_LLM_REASONING_EFFORT=medium
MODEL_EXPRESS_LLM_PLATEAU_REASONING_EFFORT=high
MODEL_EXPRESS_LLM_MAX_TOOL_ROUNDS=4
MODEL_EXPRESS_LLM_TIMEOUT_SECONDS=1200
MODEL_EXPRESS_LLM_MAX_RETRIES=2
```

### Visual LLM variables

```bash
MODEL_EXPRESS_VISUAL_ANALYSIS_ENABLED=true
MODEL_EXPRESS_VISUAL_LLM_ENABLED=true
MODEL_EXPRESS_VISUAL_LLM_PROVIDER=openai
MODEL_EXPRESS_VISUAL_LLM_MODEL=gpt-5.4-mini
MODEL_EXPRESS_VISUAL_LLM_API_STYLE=responses
MODEL_EXPRESS_VISUAL_LLM_REASONING_EFFORT=medium
MODEL_EXPRESS_VISUAL_LLM_MAX_TOOL_ROUNDS=3
```

### Safety model

OpenAI output is never treated as direct execution. Agent output must be structured JSON and must pass backend validation before it can schedule work or select a champion.

In other words: OpenAI can recommend. The orchestrator decides.

## OpenAI-Compatible Provider

Model Express includes an OpenAI-compatible provider boundary for hosted or self-managed APIs that expose similar JSON behavior.

### Variables

```bash
MODEL_EXPRESS_LLM_ENABLED=true
MODEL_EXPRESS_LLM_PROVIDER=openai_compatible
MODEL_EXPRESS_LLM_BASE_URL=https://your-compatible-provider.example/v1
MODEL_EXPRESS_LLM_API_KEY=replace-with-provider-key
MODEL_EXPRESS_LLM_MODEL=replace-with-model-name
MODEL_EXPRESS_LLM_API_STYLE=chat_completions
```

### Best for

- Testing an OpenAI-compatible gateway.
- Hosted open-source model APIs.
- Provider portability experiments.

### Caveats

This is an advanced path. The release-grade path is OpenAI with the Responses-style workflow, because cloud preflight expects that configuration for the full cloud agentic demo.

## Local LLM Provider

The config model has a `local` provider option, but the current release story should treat local LLMs as experimental unless you have a compatible local endpoint and have verified JSON quality.

Use the local option only when you are intentionally testing local model behavior. The same backend validation still applies, but weaker JSON compliance can reduce planner reliability.

## Embeddings and Memory Retrieval

Embeddings are optional. They are used for compact agent memory retrieval, not for vectorizing raw prompts or full datasets.

### Variables

```bash
MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED=true
MODEL_EXPRESS_MEMORY_RETRIEVAL_LOG_ONLY=false
MODEL_EXPRESS_MEMORY_RETRIEVAL_MAX_CARDS=10
MODEL_EXPRESS_MEMORY_RETRIEVAL_MIN_SCORE=0.60
MODEL_EXPRESS_MEMORY_CROSS_PROJECT_ENABLED=false

MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED=true
MODEL_EXPRESS_MEMORY_EMBEDDING_PROVIDER=openai
MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL=text-embedding-3-small
MODEL_EXPRESS_MEMORY_EMBEDDING_API_KEY=
MODEL_EXPRESS_MEMORY_EMBEDDING_DIMENSIONS=1536
MODEL_EXPRESS_MEMORY_EMBEDDING_MAX_CALLS_PER_DAY=100
MODEL_EXPRESS_MEMORY_EMBEDDING_DAILY_BUDGET_USD=1.00
```

### What gets indexed

- Strategy scorecards.
- Distilled planning/training memories.
- Dataset profile fingerprints.
- Accepted visual-analysis cards.
- Preprocessing hypotheses.

### What does not get indexed by default

- Raw prompts.
- Raw LLM outputs.
- Full invocation contexts.
- Full epoch arrays.
- Full manifests.
- Image URIs.
- Unbounded JSON payloads.

That makes retrieval useful for planning while keeping the memory surface bounded and auditable.

## S3-Compatible Storage Provider

Model Express uses S3-compatible object storage for dataset archives and artifacts.

### Local MinIO variables

```bash
S3_ENDPOINT_URL=http://127.0.0.1:9000
S3_BUCKET=model-express
MODEL_EXPRESS_ARTIFACT_BUCKET=model-express
MODEL_EXPRESS_ARTIFACT_PREFIX=model-express/artifacts
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

### Modal storage variables

```bash
MODEL_EXPRESS_MODAL_AWS_ACCESS_KEY_ID=...
MODEL_EXPRESS_MODAL_AWS_SECRET_ACCESS_KEY=...
MODEL_EXPRESS_MODAL_TUNNEL_S3=true
MODAL_S3_ENDPOINT_URL=optional-public-s3-origin
MODEL_EXPRESS_MODAL_S3_ENDPOINT_URL=optional-public-s3-origin
```

### Bring-your-own storage

You can point the storage config at AWS S3, Cloudflare R2, or another compatible object store. For remote workers, the endpoint must be reachable by the remote runtime.

### Safety rules

- MinIO console ports should not be used as S3 API endpoints.
- Remote upload/storage origins require explicit allowlists or a controlled tunnel.
- Local storage credentials can be generated and stored in app-local user data for demos.

## Tunnels and Exposed Orchestrator Mode

When Modal workers need to call a local orchestrator or local MinIO, those services need public HTTPS origins. Model Express supports a tunnel-oriented setup for that workflow.

### Variables

```bash
MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE=true
MODEL_EXPRESS_ALLOW_LAN=true
MODEL_EXPRESS_API_TOKEN=generated-or-configured-secret
MODAL_ORCHESTRATOR_URL=optional-public-orchestrator-origin
MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL=optional-public-orchestrator-origin
MODEL_EXPRESS_MODAL_TUNNEL_S3=true
MODAL_S3_ENDPOINT_URL=optional-public-s3-origin
```

### Security expectation

If the orchestrator is exposed beyond loopback, an API token is required. Mission Control sends both bearer-style and Model Express-specific token headers for authenticated orchestrator calls.

For release demos, prefer automatic tunnel setup through Mission Control or explicitly configured public origins. Avoid binding the orchestrator to a LAN or public interface without a token.

## ONNX Runtime and Demo Providers

Model Express supports champion demo surfaces through exported artifacts and local/browser inference when possible.

The UI includes ONNX Runtime Web for browser-side inference. Worker-backed demo inference can also report predictions through the backend when browser execution is unavailable.

### Export/demo outputs

Champion export metadata can describe:

- Model kind.
- Task type.
- Input shape.
- Class labels.
- Preprocessing contract.
- Postprocessing contract.
- Runtime/default runtime.
- Confidence, IoU, and max-detection defaults for detector models.
- Latency budget/profile fields when known.

For ONNX exports, Model Express can attach a portable inference bundle. The bundle is designed to run without the orchestrator, Mission Control, Postgres, or Model Express packages.

### Trust rule

Demo prediction requests create durable audit rows. Missing artifacts, missing dependencies, or unavailable runtimes are recorded as unavailable or failed states. The backend does not fabricate predictions.

## AutoML

AutoML is not a separate external provider. It is a backend-owned hyperparameter suggestion layer.

By default, AutoML is disabled:

```bash
MODEL_EXPRESS_AUTOML_ENABLED=false
```

When enabled, the backend can generate concrete suggestions for supported knobs such as learning rate, weight decay, batch size, epochs, optimizer/scheduler parameters, dropout, label smoothing, gradient clipping, and selected augmentation/class-balancing numeric fields.

The LLM remains responsible for strategy. AutoML only samples within backend-supported, backend-validated search spaces.

## Cloud Preflight Checklist

The cloud agentic profile has a preflight path that checks whether the important providers are configured correctly.

It validates:

- Cloud v1 profile is enabled.
- Default training provider is Modal.
- OpenAI provider is enabled.
- Responses-style API is configured.
- Stored Responses are enabled.
- LLM model and API key are present.
- Modal authentication is configured.
- Public orchestrator access or automatic tunnels are configured.
- Public S3 access or S3 tunneling is configured.
- API token requirements are satisfied for public exposure.
- S3 bucket and credentials are present.
- Memory retrieval settings are coherent.

Use preflight before a recorded demo or a live walkthrough. It is much better to fail fast with remediation than to discover missing provider credentials halfway through an autonomous training loop.

## Provider Cost Notes

Provider pricing changes over time, so this repo should avoid hardcoding pricing in documentation. Instead:

- Keep external provider dashboards as the source of truth for pricing.
- Use conservative follow-up limits.
- Keep `MODEL_EXPRESS_BUDGET_CAP_USD` low for demos.
- Prefer T4/L4-style defaults before trying larger GPUs.
- Enable embeddings only when memory retrieval is part of the demo.
- Watch Model Express telemetry for token usage, cached input, reasoning tokens, embedding calls, and cost estimates when available.

The system is designed so provider costs are visible and bounded rather than hidden.

## Choosing a Provider Profile

### Choose safe local when...

- You want a zero-secret walkthrough.
- You want to show Mission Control and the backend architecture.
- You are validating dataset upload/profile/plan/champion UI behavior.
- You do not need real GPU training.

### Choose cloud agentic when...

- You want to show real remote training.
- You want autonomous follow-up planning.
- You want OpenAI + Modal + storage integration in one flow.
- You are comfortable managing API keys and provider costs.

### Choose bring-your-own storage when...

- Modal should not rely on a tunnel to local MinIO.
- You already have S3/R2-compatible infrastructure.
- You want cleaner remote-worker access to datasets and artifacts.

## Provider Boundary Summary

Model Express can use several external providers, but the provider boundary never replaces backend control:

```text
Provider supplies capability
        ↓
Backend validates whether that capability can be used
        ↓
Worker executes a bounded job
        ↓
Backend stores the result and audit trail
```

That is the core release message: Model Express is provider-flexible, but execution remains deterministic, auditable, and backend-governed.
