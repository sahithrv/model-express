# Model Express Architecture

> Last updated: June 23, 2026  
> This document is intentionally high-level. It explains the system shape, safety boundaries, and major runtime flows without trying to duplicate every implementation detail.

Model Express is an agentic computer-vision experiment platform. It gives an operator a desktop-first Mission Control app for uploading image datasets, profiling them, running backend-validated experiment plans, reviewing model performance, selecting a champion, and exporting or demoing the winning model.

The most important design choice is the control boundary:

```text
LLMs propose structured JSON
        ↓
Go orchestrator validates, stores, ranks, and schedules
        ↓
Python workers execute bounded jobs
        ↓
Metrics, evaluations, exports, and audit events flow back to the backend
```

LLMs do not directly create workers, mutate datasets, choose arbitrary filesystem paths, enqueue jobs, export models, or run inference. They produce recommendations. The Go backend is the execution gate.

## High-Level Flow

```text
Operator
  │
  ▼
Mission Control
Electron + Vite + React desktop UI
  │
  ├── selects a local image dataset
  ├── streams a zip archive to S3-compatible storage
  ├── creates projects, datasets, plans, and worker requirements
  └── renders runs, agents, champions, exports, and demos
  │
  ▼
Go Orchestrator
Gin API + Postgres control plane
  │
  ├── validates requests and agent output
  ├── persists projects, datasets, plans, jobs, workers, metrics, memory, and events
  ├── creates dataset profiling and training jobs
  ├── finalizes LLM planner proposals
  ├── tracks worker leases and retries
  └── exposes SSE/event refresh surfaces for Mission Control
  │
  ▼
Python Workers
Local worker or Modal-backed worker runtime
  │
  ├── register and poll for work
  ├── profile datasets and metadata sidecars
  ├── run local contract/smoke jobs or Modal GPU training
  ├── report epoch metrics, summaries, and evaluations
  ├── generate visual exemplars
  └── export or demo champion models
  │
  ▼
Champion + Export Layer
ONNX/TorchScript/checkpoint artifacts, portable inference bundle, demo predictions
```

## Repository Layout

```text
model-express/
├── apps/
│   └── mission-control/              # Electron/Vite/React operator application
│       ├── electron/                 # desktop shell, IPC bridge, dataset upload, workers, local runtime, tunnels
│       └── src/                      # project detail UI, plans, runs, agents, telemetry, champion demo/export views
│
├── services/
│   ├── orchestrator/                 # Go control plane
│   │   ├── cmd/orchestrator/         # process entrypoint, env loading, Postgres connection, HTTP server
│   │   └── internal/
│   │       ├── api/                  # routes, validation, plan execution, agents, workers, champions, exports
│   │       ├── store/                # Postgres + in-memory store contract
│   │       ├── agents/               # Training Monitor and Experiment Planner logic
│   │       ├── llm/                  # OpenAI/OpenAI-compatible JSON client and provider config
│   │       ├── memory/               # agent memory, embeddings, retrieval telemetry
│   │       ├── automl/               # backend-owned hyperparameter suggestion/trial records
│   │       ├── datasets/             # profile and normalized metadata models
│   │       ├── jobs/                 # job and epoch metric models
│   │       ├── plans/                # experiment plan models
│   │       ├── runs/                 # summaries, evaluations, champions, exports, demo predictions
│   │       └── workers/              # worker registration, heartbeat, and assignment models
│   │
│   └── worker/                       # Python execution runtime
│       ├── worker/main.py            # worker registration, polling loop, execution/failure reporting
│       ├── worker/jobs.py            # job template dispatch
│       ├── worker/training/          # local/Modal training, preprocessing, resource selection, YOLO support
│       ├── worker/exporting/         # champion export, demo inference, portable bundle generation
│       ├── worker/datasets/          # profiling and metadata discovery helpers
│       └── tests/                    # worker contract and helper tests
│
├── infra/                            # local Postgres + MinIO compose runtime
├── docs/                             # design notes, plans, and internal ground-truth documentation
├── plans/                            # UI/product planning docs
└── scripts/                          # local cleanup and operational helpers
```

## Core Architectural Principles

### 1. LLMs are advisors, not executors

Model Express is agentic, but it is not an LLM free-for-all. The Training Monitor and Experiment Planner can analyze run history and propose actions, but executable work must pass through deterministic backend validation.

For follow-up experiments, the planner produces candidate hypotheses and complete structured experiment configs. The backend finalizer ranks, filters, rejects, or schedules those candidates according to supported models, dataset/task compatibility, duplicate prevention, cost settings, automation mode, and project trajectory rules.

### 2. The Go backend owns the control plane

The orchestrator is the source of truth for projects, datasets, plans, jobs, workers, run summaries, evaluations, agent invocations, memory, decisions, scorecards, worker requirements, execution events, champions, exports, and demo predictions.

The UI renders backend state. Workers report results. Agents write structured recommendations. None of those layers independently decide what is true.

### 3. Desktop-first, cloud-assisted

The product is designed to feel like a local desktop application while still supporting real GPU work through cloud providers. Mission Control can manage local runtime services, local dataset selection, local S3-compatible storage, local workers, and public tunnel configuration for Modal-backed remote execution.

That keeps the demo approachable while avoiding a fake “all-cloud” architecture that would be harder to run locally.

### 4. Store-first reliability

Model Express favors Postgres, row locks, leases, idempotent writes, durable events, and SSE refresh hints before introducing heavier distributed infrastructure. For the current scale—desktop operator, small-to-moderate worker count, and training jobs that take minutes—Postgres is enough of a coordination layer.

The design intentionally avoids Kafka, Redis, NATS, WebSockets, and workflow engines until the existing Postgres + leases + SSE architecture is no longer sufficient.

### 5. Failure states are product states

The system does not pretend a model was exported when no artifact exists. It does not fabricate demo predictions when runtime dependencies or model artifacts are missing. It records pending, failed, unavailable, rejected, and runtime-unavailable states so operators can understand what happened.

That is a major trust feature: Model Express should be able to explain both successful automation and blocked automation.

## Major Components

### Mission Control

Mission Control is the operator-facing desktop app. It is built with Electron, Vite, React, and TypeScript.

It is responsible for:

- Creating projects and datasets.
- Selecting local dataset folders through the desktop picker.
- Streaming dataset archives to S3-compatible storage.
- Starting local Postgres/MinIO services when using the local runtime profile.
- Supervising local workers requested by the backend.
- Managing public tunnel configuration for Modal-backed work.
- Rendering plans, jobs, metrics, evaluations, agent decisions, memory, scorecards, worker requirements, execution events, telemetry, champions, exports, and demo predictions.
- Running browser-side champion demo inference when a compatible ONNX artifact is available.

Mission Control uses a guarded IPC bridge. Renderer requests must use allowed HTTP methods, safe app paths, JSON-serializable bodies, and approved orchestrator/storage origins. Non-loopback origins require explicit configuration.

### Orchestrator

The orchestrator is the Go control plane. It exposes the HTTP API, persists durable state, validates all executable requests, and manages job scheduling.

Its responsibilities include:

- Project and dataset lifecycle.
- Dataset profile and normalized metadata import storage.
- Plan creation, validation, execution, cancellation, and follow-up scheduling.
- Worker registration, heartbeat, polling, and lease recovery.
- Epoch metric ingestion and run summary/evaluation storage.
- Agent invocation, decision, memory, embedding, and telemetry records.
- AutoML study, suggestion, and trial records.
- Champion selection, export records, demo prediction audit rows, and feedback.
- Execution events and server-sent event streams for UI refresh.

The orchestrator uses Gin for the API layer and Postgres for durable state. An in-memory store mirrors the core behavior for tests and local development paths.

### Workers

Workers are Python processes that register with the orchestrator and poll for jobs.

A worker can handle several job categories:

- Dataset profiling.
- Dataset visual analysis.
- Local training contract/smoke jobs.
- Modal-dispatched training jobs.
- Epoch metric reporting.
- Training run summaries and evaluations.
- Visual exemplar generation.
- Champion export.
- Champion demo prediction.

Workers do not schedule their own experiments. They execute jobs assigned by the backend and report results through backend APIs.

### Postgres

Postgres is the coordination and audit layer. It stores durable rows for projects, datasets, plans, jobs, workers, metrics, summaries, evaluations, agent state, automation settings, scorecards, AutoML records, champions, exports, demo predictions, worker requirements, and execution events.

Current reliability features include:

- Transaction-protected job assignment using row locks.
- Job attempt and lease metadata.
- Recovery paths for expired non-terminal jobs.
- Idempotent epoch metric writes keyed by job and epoch.
- Idempotent champion export behavior for repeated champion/format requests.
- Durable demo prediction audit rows.
- Additive normalized metadata imports.
- Compact memory embeddings with usage telemetry and budget caps.

### Object Storage

Datasets and artifacts are stored through an S3-compatible interface. In the local profile, Mission Control can manage MinIO. In a cloud-oriented profile, the same interface can point to a public S3-compatible endpoint, or Mission Control can expose local MinIO through a controlled tunnel for Modal workers.

The storage boundary matters because workers may execute outside the local machine. Remote workers need a reachable dataset archive and artifact destination.

### Modal

Modal is the current remote GPU training provider. Model Express uses Modal to run real training jobs for supported Torchvision classifiers and Ultralytics YOLO detectors. Local mode remains useful for fast smoke tests and demos, but Modal is the main path for real remote GPU execution.

The Modal path carries explicit resource metadata: requested GPU, effective GPU, batch size, memory, resource attempt, and failure class. This gives the backend and UI enough information to explain cost/runtime/resource tradeoffs.

### LLM Runtime

The LLM runtime supports OpenAI-style JSON generation through OpenAI or OpenAI-compatible providers. It supports both Chat Completions-style and Responses-style calls, usage telemetry, stored response workflows, and bounded information-tool rounds.

The LLM runtime is used by specialized agents, not by workers directly. The output contract is structured JSON that must validate before it can affect stored recommendations or executable work.

## Data Model Overview

The internal data model is broad, but it can be understood in a few groups:

| Area | Durable concepts |
| --- | --- |
| Project setup | projects, datasets, dataset profiles, dataset metadata imports, metadata sources, classes, manifest records, annotations, splits |
| Execution | experiment plans, planned experiments, jobs, worker requirements, workers, execution events |
| Training output | epoch metrics, run summaries, run evaluations, confusion/per-class metrics, diagnostics |
| Agents | invocations, decisions, memory records, embeddings, retrieval usage, retrieval query cache |
| Optimization | AutoML studies, suggestions, trials |
| Champion lifecycle | selected champion, champion exports, demo predictions, feedback |
| Telemetry | LLM usage, embedding usage, source indexing, retrieval injection/cache/skipped state |

The important release-level point is that Model Express keeps the agent loop auditable. Recommendations, rejected candidates, selected candidates, scorecards, run outcomes, and export/demo results are all represented as durable backend state.

## Dataset and Metadata Flow

1. The operator creates a project in Mission Control.
2. The operator selects a local image dataset folder.
3. Mission Control validates the selected directory and streams a zip archive to S3-compatible storage.
4. The orchestrator creates a dataset row and queues a `profile_dataset` job.
5. A worker profiles the dataset locally or through Modal.
6. The worker discovers bounded metadata sidecar candidates and file inventory.
7. The worker posts normalized metadata imports and legacy profile JSON back to the orchestrator.
8. The backend stores agent-safe summaries separately from raw source previews.
9. The backend marks the dataset profiled and creates an initial experiment plan when appropriate.

Supported dataset evidence includes common image-folder layouts, single archive-wrapper folders, common image roots such as `images/`, `JPEGImages/`, `img/`, `imgs/`, and `data/`, generic CSV manifests, CUB sidecars, CUB part/keypoint files, Pascal VOC bounding boxes, and YOLO object-detection evidence through the legacy profiler/training path.

The metadata system is intentionally additive. It can preserve raw-ish backend/operator provenance while giving agents only compact, safe summaries.

## Planning and Validation Flow

Plans are represented as backend-owned experiment batches. Each planned experiment contains concrete training configuration: model, template, image size, epochs, batch size, learning rate, preprocessing, augmentation, optimizer, scheduler, class balancing, sampling, fine-tuning strategy, and optional AutoML metadata.

The backend validates:

- Supported model names.
- Task/model compatibility.
- Dataset compatibility for YOLO detector jobs.
- Epochs, batch size, learning rate, image size, optimizer, scheduler, dropout, label smoothing, gradient clipping, and related numeric bounds.
- Preprocessing and augmentation fields.
- Class balancing and sampling settings.
- Duplicate experiment signatures.
- Repeated low-value mechanisms.
- Automation mode, follow-up limits, and cost/budget settings.

The supported classification catalog currently includes compact live-friendly models and heavier quality challengers: MobileNet V3, EfficientNet, RegNet, ResNet, ConvNeXt Tiny, Swin-T, and ViT-B/16. For YOLO datasets, the catalog adds YOLO11 detector weights from nano through extra-large.

## Training Execution Flow

A training run starts when a user or an autonomous backend path executes a validated plan.

```text
POST /plans/:id/execute
        ↓
backend creates one train_experiment job per novel experiment
        ↓
workers poll /workers/:id/poll
        ↓
worker executes local or Modal-backed job
        ↓
worker reports epoch metrics
        ↓
worker upserts final run summary and evaluation
        ↓
worker completes or fails the job
        ↓
backend records events, diagnostics, agent outcomes, and potential follow-up decisions
```

Local execution is designed for fast development, contract testing, and demos. Modal execution is the real remote GPU path for supported classifier and detector training.

For classification jobs, the worker path supports transfer-learning style training with preprocessing, normalization, structured augmentation, class balancing, weighted/focal/effective-number loss, weighted sampling, scheduler and optimizer options, label smoothing, gradient clipping, and optional bbox crop/full-image ablations when validated annotations are available.

For YOLO detector jobs, the Modal path trains Ultralytics YOLO models from supported pretrained weights, preserves train/validation/test split evidence when available, and reports detection metrics such as mAP, precision, recall, losses, latency/FPS estimates, task type, and model kind.

## Agent Flow

Model Express currently focuses on two key agents:

### Training Monitor

The Training Monitor evaluates completed or failed training jobs. It can summarize overfitting, underfitting, plateau behavior, training dynamics, per-class problems, cost/runtime tradeoffs, and outcome lessons. Those lessons become memory for later planning.

### Experiment Planner

The Experiment Planner reviews compact project, dataset, run, champion, memory, trajectory, and validation cards. It can recommend:

- `ADD_EXPERIMENTS`
- `SELECT_CHAMPION`
- `STOP_PROJECT`
- `WAIT`

For `ADD_EXPERIMENTS`, the planner must return candidate hypotheses with a mechanism, intervention, evidence, expected effect, tradeoffs, risk/cost/novelty, proposed changes, and a complete executable experiment config.

The backend finalizer is authoritative. If no candidate survives deterministic checks, no experiment is scheduled.

## Memory and Retrieval

Agent memory is deliberately compact. The system indexes memory cards, strategy scorecards, dataset profile fingerprints, accepted visual-analysis cards, and preprocessing hypotheses rather than raw prompts, raw LLM outputs, full manifests, image URIs, or unbounded JSON payloads.

Retrieval can be enabled to supply relevant memory to the Training Monitor or Experiment Planner. Usage is tracked separately for source indexing and retrieval queries. The retrieval system supports query caching, budget caps, log-only rollout behavior, and lexical fallback when vector retrieval is unavailable or not worthwhile.

The goal is not “memory for memory’s sake.” The goal is to give future planner calls enough outcome context to avoid repeating low-value strategies and to explain why a proposed mechanism is worth trying.

## Champion, Export, and Demo Flow

A champion can be selected deterministically or through a validated planner decision. A champion record stores the selected job, plan, dataset, source decision, selection rationale, metrics, evaluation payload, deployment profile, and pending export notes.

Champion export and demo are backend-scheduled worker jobs:

1. The UI requests an export or demo prediction.
2. The backend validates that a champion exists and that the request is supported.
3. The backend creates an export/demo record and queues worker work when needed.
4. The worker produces an artifact or reports a clear unavailable/failure status.
5. The backend updates durable export/demo rows and emits execution events.

Supported export surfaces include ONNX-oriented metadata, TorchScript/checkpoint paths where available, and a portable inference bundle. The portable bundle is designed to be standalone: it contains the model artifact, runtime-neutral manifest, Python and Node ONNX Runtime examples, requirements, README, and expected outputs when self-test data is available.

Demo prediction rows are audit records. If artifacts or dependencies are missing, the system records that instead of fabricating predictions.

## API Surface

The orchestrator exposes a broad but conventional API. At a high level:

| Area | Example responsibilities |
| --- | --- |
| Health/settings/preflight | health checks, automation settings, cloud preflight validation |
| Projects | create/list/read projects, project detail surfaces |
| Datasets | upload registration, profile updates, metadata imports, metadata summaries/bundles, visual analyses, exemplars |
| Plans | create/list/read/execute/cancel plans |
| Jobs | create/list/read, metrics, summaries, evaluations, modal call records, complete/fail/retry paths |
| Workers | register, heartbeat, poll, worker requirements |
| Agents | decisions, invocations, memory, embeddings, scorecards, telemetry |
| Champions | champion selection, exports, demo images, demo predictions, feedback |
| Events | execution event history, SSE refresh streams, activity streams |

The API is intentionally not just a training API. It is a control-plane API for an auditable agentic workflow.

## Reliability State

Implemented current-scale hardening includes:

- Postgres-backed job assignment with row locks.
- Worker lease fields with owner, attempt, expiry, and heartbeat metadata.
- Expired non-terminal job recovery paths.
- Idempotent epoch metrics by job and epoch.
- Durable worker requirements for Mission Control-supervised worker startup.
- Durable execution events and SSE refresh hints.
- Backend validation for planner output.
- Backend-owned AutoML suggestions and trials.
- Additive normalized metadata imports.
- Agent-safe metadata summaries separated from raw source previews.
- Compact memory embeddings with telemetry, cache, fallback, and caps.
- Export/demo records that preserve pending, failed, and runtime-unavailable states.

Known limits are also explicit:

- The architecture is designed for desktop-first and small-to-moderate experiment batches, not internet-scale distributed training.
- Some duplicate prevention remains application-level rather than fully database-idempotency-key based.
- Some terminal job paths can still synchronously trigger agent work.
- Some automation locks are process-local.
- SSE streams are refresh hints, not a full event-sourcing system.
- Production artifact storage and export self-test coverage can be expanded further.

These are reasonable boundaries for a v1 release because they show where the system is already hardened and where future scaling work belongs.

## Why This Architecture Is Trustworthy

Model Express is trying to demonstrate agentic ML automation without hiding the hard parts. The codebase makes several choices that are good signals for a serious engineering project:

- The LLM boundary is explicit and enforced by backend validation.
- Every major lifecycle stage has durable state.
- The UI does not invent orchestration state.
- Dataset metadata is bounded and split into operator/debug provenance versus agent-safe summaries.
- Training outcomes, decisions, and follow-up memories are auditable.
- Remote GPU execution is a provider boundary, not an assumption baked into the entire system.
- Failure, rejection, pending, and runtime-unavailable states are first-class.

The result is not just a demo that trains models. It is a control plane for understanding how agentic computer-vision experiments were proposed, validated, executed, evaluated, and exported.
