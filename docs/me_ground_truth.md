# Model Express Ground Truth

This document is the durable system reference for Model Express. It describes how the system works now, including current safety boundaries and known limits.

## Core Boundary

Model Express is an agentic image-classification experiment platform with a strict control boundary:

```text
LLMs propose structured JSON -> backend validates -> backend stores/schedules -> workers execute
```

LLMs never directly create workers, mutate datasets, choose arbitrary filesystem paths, enqueue jobs, export models, or run inference. Every executable action must pass through deterministic Go backend validation.

## Main Components

- Mission Control (`apps/mission-control`): Electron/Vite/React operator UI for projects, datasets, plans, jobs, agents, workers, champions, export records, and demo surfaces.
- Orchestrator (`services/orchestrator`): Go control plane. It owns API validation, Postgres/in-memory stores, plan execution, job assignment, agent invocations, memory, decisions, scorecards, champions, export records, and execution events.
- Worker (`services/worker`): Python runtime. Workers register with the orchestrator, poll for jobs, profile datasets, run local or Modal training, report metrics/summaries/evaluations, and complete/fail jobs.
- Postgres: durable control-plane store. In-memory store mirrors behavior for tests/dev.
- LLM agents: Training Monitor and Experiment Planner. They produce JSON recommendations only.

## Data Model

Major tables and store concepts:

- `projects`: project name, goal, status.
- `datasets`: dataset metadata and canonical `profile` JSON.
- `dataset_profiles`: exists in migration but is not active. `datasets.profile` is the current source of truth.
- `experiment_plans`: planned experiment batches and source decision linkage.
- `experiment_jobs`: queued/assigned/running/terminal worker jobs.
- `epoch_metrics`: idempotent metrics keyed by `(job_id, epoch)`.
- `training_run_summaries`: run-level status, metrics, runtime, cost, provider, model.
- `training_run_evaluations`: objective profile, per-class metrics, confusion matrix, model profile, holistic scores.
- `project_champions`: selected champion job plus metrics, evaluation, deployment profile.
- `champion_exports`: additive export records for selected champions; repeated requests for the same project/champion/format update the existing record.
- `champion_demo_predictions`: demo prediction audit/history rows for selected champions, including image metadata, status, top-k payloads when available, latency/correctness, and runtime errors.
- `agent_invocations`: LLM input/output, validation status, parsed output, downstream outcome.
- `agent_memory_records`: distilled training/planning memory.
- `agent_decisions`: accepted project decisions such as `ADD_EXPERIMENTS` or `SELECT_CHAMPION`.
- `strategy_scorecards`: structured follow-up outcome memory.
- `worker_requirements`: durable requests for Mission Control to satisfy worker capacity.
- `execution_events`: audit events for queued jobs, worker requirements, agent outcomes, champion selection, and champion export requests.

## Dataset And Profile Flow

1. Mission Control creates a project and uploads an image-folder dataset to configured object storage.
2. The backend creates a dataset row and queues a `profile_dataset` job.
3. A worker downloads/extracts the dataset and runs `profile_image_folder`.
4. The worker posts profile JSON to `POST /datasets/:id/profile`.
5. The backend stores that JSON in `datasets.profile`, marks the dataset profiled, and creates the initial experiment plan if one does not exist.

Profile JSON includes class counts, image counts, image dimension stats, corrupt-file counts, split summaries, metadata/artifact detection, leakage warnings, dataset traits, normalization metadata, and optional `visual_exemplars`/`demo_images` arrays.

`dataset_profiles` rows are deferred. They should not become active until there is a complete write path, latest-profile read path, backfill plan, and tests proving one canonical source.

## Planning And Validation Flow

Plans are represented by `ExperimentPlan` and `PlannedExperiment`.

Supported execution-facing planning fields include:

- `resolution_strategy`
- `preprocessing.resize_strategy`
- `preprocessing.normalization`
- `preprocessing.crop_strategy`
- `preprocessing.bbox_mode`
- `preprocessing.use_dataset_normalization`
- `augmentation_policy`
- `augmentation`
- `class_balancing`
- `sampling_strategy`
- optimizer, scheduler, weight decay, fine-tune strategy, pretrained/freeze flags

The backend validates allowed model names, epochs, batch size, learning rate, image size, optimizer, scheduler, preprocessing values, augmentation keys, augmentation policy, class balancing, sampling strategy, early stopping, and fine-tune strategy.

Duplicate experiment signatures include preprocessing, augmentation policy, sampling strategy, and resolution strategy.

## Training Flow

1. A user or autonomous backend path executes a validated plan through `POST /plans/:id/execute`.
2. The backend creates one `train_experiment` job per novel experiment/provider/index.
3. Workers poll `POST /workers/:id/poll` and receive assigned jobs.
4. Local training simulates contract behavior for fast local demos.
5. Modal training performs torchvision transfer learning for supported model families.
6. Workers report epoch metrics, summary updates, and final evaluation.
7. Workers complete or fail the job through backend APIs.

Worker-side preprocessing currently supports deterministic resize/crop options, ImageNet/none normalization, bounded dataset-computed normalization, augmentation policies, weighted/focal loss, and weighted sampling.

Worker utility modules also include:

- split-file, Pascal VOC XML, and annotation JSON parsers
- class-balanced visual exemplar generation with PIL downscale/compression and byte/image caps
- champion export manifest/checkpoint helpers with guarded TorchScript/ONNX paths
- TorchScript demo inference helper that returns a ranked payload when a valid worker-owned artifact exists, or a deterministic pending/error payload when dependencies/artifacts are missing

Explicit split-file training, real bbox crop ablations, advanced augmentation object policies, and backend-scheduled export/inference worker jobs remain deferred.

## Agent Flow

Training Monitor evaluates individual completed/failed training jobs and writes training-evaluation memory.

Experiment Planner reviews completed plans and can recommend:

- `ADD_EXPERIMENTS`
- `SELECT_CHAMPION`
- `STOP_PROJECT`
- `WAIT`

Planner input includes project goal, dataset profile, dataset planning insights, optional visual exemplar evidence, objective context, deterministic diagnosis, supported model catalog, current champion, run deltas, no-improvement rounds, prior plans/jobs/evaluations, memory, rejected strategy memory, scorecards, and existing experiment signatures.

Visual exemplars are evidence only. Backend-curated/capped metadata may help the planner cite object scale, background dominance, blur, lighting, fine-grained classes, or bbox/crop plausibility. Planner context includes exemplar caps and audit details when available. Planner output is still JSON only, and backend validation remains the gate.

## Champion Flow

Champion selection can come from deterministic review or validated planner decisions. `SELECT_CHAMPION` persists a `project_champions` row with:

- selected job, plan, dataset, source decision
- selection reason
- metrics from the training run summary
- evaluation payload if available
- deployment profile and model-card-style pending export notes

Mission Control displays the selected champion, champion comparison table, objective fit, model profile, confusion preview, and deployment notes.

## Export And Demo Flow

Current backend export/demo APIs:

- `GET /projects/:id/champion/exports`
- `POST /projects/:id/champion/exports`
- `GET /projects/:id/champion/demo-images`
- `GET /projects/:id/champion/demo-predictions`
- `POST /projects/:id/champion/demo-predictions`
- `GET /datasets/:id/visual-exemplars`

`POST /projects/:id/champion/exports` validates that a champion exists, the champion job succeeded, and the requested format is supported. If no artifact URI exists yet, it records `PENDING_ARTIFACT` with validation errors rather than pretending export is ready. Repeated requests for the same project/champion/format are idempotent at the store/API behavior level.

Demo image listing reads capped metadata from `datasets.profile.demo_images` or `datasets.profile.visual_exemplars`. Demo prediction requests create durable `champion_demo_predictions` audit rows. Until worker-backed inference is wired, `POST /projects/:id/champion/demo-predictions` returns `202` with a `RUNTIME_UNAVAILABLE` prediction record rather than fake prediction data.

Mission Control can request an ONNX export record, show export status/errors, list demo images, browse next/random demo examples, and display prediction history, pending/runtime-unavailable status, true labels, top-k rows, latency, and correctness fields when present.

## Mission Control Flow

Mission Control polls project detail endpoints and renders:

- section navigation for Overview, Data, Agents, Runs, Operations, Export/Demo
- dataset intelligence and preprocessing recommendations
- automation settings and worker requirements
- experiment timeline and execution events
- agent decisions, rejections, candidate score components, and scorecards
- plans with typed preprocessing fields
- run summaries/evaluations and metric charts
- selected champion and export/demo panel
- champion demo prediction history and runtime-unavailable audit rows

Mission Control does not invent orchestration state. It uses backend APIs, renders partial data defensively, and surfaces rejection/failure states as part of operator trust.

## Reliability State

Implemented/current-scale hardening:

- Postgres row-lock assignment with `FOR UPDATE SKIP LOCKED`.
- Epoch metrics are idempotent by `(job_id, epoch)`.
- Positive epoch validation at API/store boundaries.
- Champion export requests are idempotent for the same project/champion/format at the store/API behavior level.
- Champion demo prediction attempts are durable audit rows with execution events.
- Worker requirements and execution events for local worker supervision.
- Additive indexes for common project/status/agent/memory/scorecard reads.
- Follow-up rounds remain bounded by automation settings and max follow-up rounds.

Known reliability gaps:

- No durable job lease/requeue loop yet.
- Some duplicate prevention is still application-level rather than DB idempotency-key based.
- Terminal job endpoints can still synchronously trigger agent work.
- `autoReviewMu` is process-local.
- `execution_events` are audit rows, not yet an SSE cursor stream.

## Deferred Work

- Fully promote `dataset_profiles` or keep it permanently documented as deferred.
- Wire generated/downscaled/compressed visual exemplars into backend profile/artifact upload paths.
- Persist visual exemplar generation/audit history beyond compact profile JSON/planner context.
- Wire parsed annotations, explicit split-file training, and bbox crop/full-image ablations into training.
- Advanced augmentation object policies.
- Backend-scheduled champion artifact export and worker-backed inference runtime.
- Job leases, stale job recovery, durable idempotency keys, async agent task queue, and SSE.

Do not add Kafka, Redis, NATS, WebSockets, or a workflow engine until Postgres hardening, leases, idempotency, and SSE are no longer sufficient.
