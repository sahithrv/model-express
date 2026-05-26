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
- Worker (`services/worker`): Python runtime. Workers register with the orchestrator, poll for jobs, profile datasets, run local or Modal training, produce controlled champion export/demo/exemplar job results, report metrics/summaries/evaluations, and complete/fail jobs.
- Postgres: durable control-plane store. In-memory store mirrors behavior for tests/dev.
- LLM agents: Training Monitor and Experiment Planner. They produce JSON recommendations only.

## Data Model

Major tables and store concepts:

- `projects`: project name, goal, status.
- `datasets`: dataset metadata and canonical `profile` JSON.
- `dataset_profiles`: exists in migration but is not active. `datasets.profile` is the current source of truth.
- `experiment_plans`: planned experiment batches and source decision linkage.
- `experiment_jobs`: queued/assigned/running/terminal worker jobs with attempt counts and lease fields for stale-job recovery.
- `epoch_metrics`: idempotent metrics keyed by `(job_id, epoch)`.
- `training_run_summaries`: run-level status, metrics, runtime, cost, provider, model.
- `training_run_evaluations`: objective profile, per-class precision/recall/F1 metrics, confusion matrix, model profile, holistic scores, and backend-added training diagnostics.
- `project_champions`: selected champion job plus metrics, evaluation, deployment profile.
- `champion_exports`: additive export records for selected champions; repeated requests for the same project/champion/format update the existing record.
- `champion_demo_predictions`: demo prediction audit/history rows for selected champions, including image metadata, status, top-k payloads when available, latency/correctness, and runtime errors.
- `agent_invocations`: LLM input/output, validation status, parsed output, downstream outcome.
- `agent_memory_records`: distilled training/planning memory.
- `agent_decisions`: accepted project decisions such as `ADD_EXPERIMENTS` or `SELECT_CHAMPION`.
- `strategy_scorecards`: structured follow-up outcome memory.
- `worker_requirements`: durable requests for Mission Control to satisfy worker capacity.
- `execution_events`: audit events for queued jobs, worker requirements, agent outcomes, champion selection, champion export requests, and demo prediction requests. The project SSE endpoint streams these durable rows as refresh hints.

## Dataset And Profile Flow

1. Mission Control creates a project and streams an image-folder dataset as a zip archive directly to configured object storage.
2. The backend creates a dataset row and queues a `profile_dataset` job.
3. A worker profiles the dataset. Modal-mode profile workers submit profiling to Modal, where the dataset archive is downloaded/extracted in temporary Modal storage. Local profile workers can still download/extract locally, but `.cache/datasets/:dataset_id` is cleaned by default after profiling unless `MODEL_EXPRESS_PERSIST_DATASET_CACHE=1` is set.
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

Duplicate experiment signatures include preprocessing, augmentation policy, sampling strategy, and resolution strategy. Backend follow-up validation also rejects or filters repeated mechanisms where the only changes are minor tuning knobs such as epochs, batch size, or learning rate.

When an LLM planner proposal passes JSON/schema parsing but fails backend validation, the backend records the rejected invocation outcome and performs one bounded correction retry with `planner_validation_feedback` in the prompt context. The retry can only succeed if the corrected JSON passes the same backend validation and scheduling gates.

## Training Flow

1. A user or autonomous backend path executes a validated plan through `POST /plans/:id/execute`.
2. The backend creates one `train_experiment` job per novel experiment/provider/index.
3. Workers poll `POST /workers/:id/poll` and receive assigned jobs with backend-owned attempt and lease metadata.
4. Local training simulates contract behavior for fast local demos.
5. Modal training performs torchvision transfer learning for supported model families.
6. Workers report epoch metrics, summary updates, and final evaluation.
7. Workers complete or fail the job through backend APIs.

Worker-side preprocessing currently supports deterministic resize/crop options, ImageNet/none normalization, bounded dataset-computed normalization, augmentation policies, weighted/focal loss, and weighted sampling.

Training evaluations for image classification include confusion matrices and per-class precision/recall/F1 when real validation/test labels are available. The backend enriches final evaluation payloads with deterministic `training_diagnostics` inside `holistic_scores`, including train/validation loss gap, divergence status, severity, and trend deltas derived from persisted epoch metrics and run summaries.

Worker utility modules also include:

- split-file, Pascal VOC XML, and annotation JSON parsers
- class-balanced visual exemplar generation with PIL downscale/compression and byte/image caps
- champion export manifest/checkpoint helpers with guarded TorchScript/ONNX paths
- TorchScript demo inference helper that returns a ranked payload when a valid worker-owned artifact exists, or a deterministic pending/error payload when dependencies/artifacts are missing
- champion job handlers for `export_champion`, `champion_demo_prediction`, and `generate_visual_exemplars`

Explicit split-file training, real bbox crop/full-image ablations, and advanced augmentation object policies remain deferred because they require a larger dataset/dataloader contract than the current safe helper slice.

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

If a validated `STOP_PROJECT` decision does not name a champion but successful runs exist, the backend selects the best successful run so far using the same objective-aware ranking helper used for planner context. This produces a usable champion without letting the LLM execute work or bypass validation.

Mission Control displays the selected champion, champion comparison table, objective fit, model profile, confusion preview, train/validation gap, seed variance when repeated seeded runs exist, and deployment notes.

## Export And Demo Flow

Current backend export/demo APIs:

- `GET /projects/:id/champion/exports`
- `POST /projects/:id/champion/exports`
- `GET /projects/:id/champion/demo-images`
- `GET /projects/:id/champion/demo-predictions`
- `POST /projects/:id/champion/demo-predictions`
- `GET /datasets/:id/visual-exemplars`
- `POST /datasets/:id/visual-exemplars`
- `POST /jobs/:id/champion-export-result`
- `POST /jobs/:id/champion-demo-prediction-result`

`POST /projects/:id/champion/exports` validates that a champion exists, the champion job succeeded, and the requested format is supported. If worker work is needed, the backend records the export and queues an `export_champion` job. Workers report results through `POST /jobs/:id/champion-export-result`; the backend validates the job template/config before updating `champion_exports`. If no artifact can be produced, the export stays `PENDING_ARTIFACT` or `FAILED` with validation errors rather than pretending it is ready. Repeated requests for the same project/champion/format are idempotent at the store/API behavior level.

Demo image listing reads capped metadata from `datasets.profile.demo_images` or `datasets.profile.visual_exemplars`. Demo prediction requests create durable `champion_demo_predictions` audit rows. When a `READY` champion export exists, the backend queues a `champion_demo_prediction` job and workers report results through `POST /jobs/:id/champion-demo-prediction-result`. Missing manifests, missing dependencies, or unavailable local artifacts are recorded as `RUNTIME_UNAVAILABLE`; the backend never fabricates predictions.

Workers can generate class-balanced exemplars with `generate_visual_exemplars` and persist the capped result through `POST /datasets/:id/visual-exemplars`. The backend enforces count/per-class/byte caps and merges accepted `visual_exemplars` and `demo_images` into canonical `datasets.profile` JSON.

Mission Control can request an ONNX export record, show export status/errors, list demo images, browse next/random demo examples, and display prediction history, pending/running/succeeded/failed/runtime-unavailable status, true labels, top-k rows, latency, and correctness fields when present.

## Mission Control Flow

Mission Control polls project detail endpoints and optionally consumes `GET /projects/:id/events/stream` as a server-sent-event refresh hint. Polling remains the durable fallback. It renders:

- section navigation for Overview, Data, Agents, Runs, Operations, Export/Demo
- dataset intelligence and preprocessing recommendations
- automation settings and worker requirements
- experiment timeline and execution events
- agent decisions, rejections, candidate score components, and scorecards
- plans with typed preprocessing fields
- run summaries/evaluations, metric charts, per-class metrics, confusion previews, and backend training diagnostics
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
- Job assignment sets `attempt`, `max_attempts`, `lease_owner_worker_id`, `lease_expires_at`, and `lease_last_heartbeat_at`.
- Polling and explicit store recovery requeue expired non-terminal jobs until `max_attempts`, then fail them.
- `GET /projects/:id/events/stream` streams durable `execution_events` over SSE for UI refresh without adding a broker.
- Worker requirements and execution events for local worker supervision.
- Additive indexes for common project/status/agent/memory/scorecard reads.
- Follow-up rounds remain bounded by automation settings and max follow-up rounds.

Known reliability gaps:

- Lease recovery currently runs when workers poll or when `RecoverExpiredJobLeases` is called; there is no standalone recovery ticker.
- Some duplicate prevention is still application-level rather than DB idempotency-key based.
- Terminal job endpoints can still synchronously trigger agent work.
- `autoReviewMu` is process-local.
- SSE streams recent durable events as refresh hints, not as a separate event-sourcing system.

## Deferred Work

- Fully promote `dataset_profiles` or keep it permanently documented as deferred.
- Persist visual exemplar generation/audit history beyond compact profile JSON/planner context.
- Wire parsed annotations, explicit split-file training, and bbox crop/full-image ablations into training.
- Advanced augmentation object policies.
- Production storage upload for generated export/exemplar artifacts beyond current worker-local `file://` URIs.
- Real model reconstruction/export from completed training runs when no worker-visible artifact exists.
- Durable idempotency keys, async agent task queue, and a standalone lease-recovery loop.

Do not add Kafka, Redis, NATS, WebSockets, or a workflow engine until Postgres hardening, leases, idempotency, and SSE are no longer sufficient.
