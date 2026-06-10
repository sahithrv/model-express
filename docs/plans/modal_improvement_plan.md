# Modal Improvement Plan

Inspection date: 2026-06-10

This document is a planning and analysis note for the current Model Express Modal integration. It focuses first on:

1. Why multiple Modal app cards and containers appear during runs.
2. How to stop YOLO and other image training jobs from failing or silently losing the intended batch size because the selected GPU/system memory is too small.

It also inventories the rest of the Modal surface in the project and lists follow-up improvements.

## Sources Inspected

Project files:

- `services/worker/worker/training/modal_app.py`
- `services/worker/worker/training/modal_provider.py`
- `services/worker/worker/training/modal_dispatcher.py`
- `services/worker/worker/training/providers.py`
- `services/worker/worker/jobs.py`
- `services/worker/worker/main.py`
- `services/worker/worker/datasets/profiler.py`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/execution/model.go`
- `services/orchestrator/internal/jobs/model.go`
- `services/orchestrator/internal/plans/model.go`
- `services/orchestrator/internal/settings/model.go`
- `apps/mission-control/src/App.tsx`
- `apps/mission-control/electron/main.cjs`
- `.env.example`
- `docs/plans/remote_gpu_cost_reduction_plan.md`

Modal docs checked:

- Apps, Functions, and entrypoints: https://modal.com/docs/guide/apps
- Scaling out: https://modal.com/docs/guide/scale
- Dynamic Function configuration: https://modal.com/docs/guide/dynamic-function-config
- GPU acceleration: https://modal.com/docs/guide/gpu
- CPU, memory, and disk: https://modal.com/docs/guide/resources
- Pricing: https://modal.com/pricing
- GPU metrics: https://modal.com/docs/guide/gpu-metrics
- Function dynamic options and spawn: https://modal.com/docs/reference/modal.Function
- FunctionCall cancellation: https://modal.com/docs/reference/modal.FunctionCall

No live Modal account state was inspected from the CLI/API. The dashboard behavior analysis below is based on the screenshot, the local code, and current Modal docs.

## Executive Summary

The code already defines one Modal app name, `model-express-training`, in `modal_app.py`. The reason the dashboard can still show multiple `model-express-training` app cards is that the worker uses `app.run()`, which creates an ephemeral Modal app session. Each local worker/dispatcher process can create its own ephemeral session, even with the same app name.

The code can also show multiple containers inside a single app because Modal scales each function independently and, by default, each container handles one input at a time. Seeing `train_yolo_detector` with `2 containers, 2 inputs` is normal when two YOLO jobs are active in the same app session.

The most urgent training-resource problem is that `gpu_type` in job config is not reliably controlling the actual Modal GPU. The Modal decorators use `gpu=DEFAULT_GPU`, and `DEFAULT_GPU` is evaluated when `modal_app.py` is imported. `run_modal_training()` mutates `os.environ["MODAL_GPU_TYPE"]` per job, but that does not change an already-registered Modal function in a long-running dispatcher process. In practice, the app is very likely running on the import-time default, usually `T4`, while job config and cost telemetry may say something else.

For memory failures, current retry behavior requeues the same job config. There is no OOM classifier, no GPU escalation ladder, no backend config patch on retry, and no structured requested-vs-effective batch-size telemetry. This can burn attempts and cost while repeating the same failing resource choice.

There is also a separate retry/idempotency risk: a Modal training input can report an error back to the backend, causing the backend to requeue the job, while the same container or another worker path may still continue far enough to train or finish the same model. That would explain cases where the same model appears to train multiple times even though one attempt eventually completed. This needs to be treated as a correctness and cost-control issue, not only a resource-sizing issue.

Recommended first fixes:

- Use Modal `Function.with_options(gpu=..., memory=..., scaledown_window=...)` at the call site so each job's selected GPU is the actual Modal resource.
- Stop hardcoding manual plan execution to `gpu_type: "T4"` in Mission Control.
- Default YOLO object-detection Modal jobs to at least `L4` unless the user explicitly chooses prototype/cheap mode.
- Add an OOM-aware retry path that escalates `T4 -> L4 -> A10 -> L40S -> A100-40GB` before reducing batch size.
- Add a single-writer/idempotency guard for training attempts so one completed attempt prevents stale retry attempts from continuing to publish artifacts, metrics, or completion events.
- Add a confirmed "Stop run" action that prevents new work, cancels in-flight Modal function calls and LLM calls, and promotes the best exportable model found so far into the normal champion/export area.
- Add resource telemetry: requested GPU, effective GPU policy, system memory request, requested/effective batch size, Modal app/session id, function call id, input id, and GPU memory used where available.
- Add a deployed-app or singleton-dispatcher mode if the goal is one durable dashboard app instead of multiple ephemeral app cards.

## Modal Concepts That Matter Here

Modal's own docs draw three important boundaries:

- An App groups one or more Functions and acts as a deployment namespace.
- Functions scale independently from other Functions in the same App.
- `app.run()` creates an ephemeral App that exists for the duration of the calling script/process.

That means "single app" and "single container" are different goals:

- Single app: one dashboard app namespace/session, useful for debugging and grouping.
- Single container: one running function container. Modal will not put unrelated active function inputs into one container by default, and GPU training should usually not use input concurrency anyway.

For training workloads, the realistic goal is:

- One durable Modal app or one active ephemeral session per project/run.
- Multiple function containers only when there are multiple active training inputs.
- No idle GPU containers after demand is satisfied.
- Clear telemetry showing which app/session/container/input ran each job.

## Current Modal Inventory

### Modal App Definition

`services/worker/worker/training/modal_app.py` defines:

- `APP_NAME = "model-express-training"`
- `DEFAULT_GPU = os.getenv("MODAL_GPU_TYPE", "T4")`
- `app = modal.App(APP_NAME)`
- dataset volume: `model-express-dataset-cache`
- torch cache volume: `model-express-torch-cache`
- one image based on `modal.Image.debian_slim(python_version="3.11")` with `torch`, `torchvision`, and `ultralytics`

This single Python module is the only place Modal functions are registered.

### Modal Functions

GPU functions:

- `train_image_classifier`
  - `gpu=DEFAULT_GPU`
  - `timeout=60 * 60`
  - torch cache volume mounted
  - optional `min_containers`, `buffer_containers`, and `scaledown_window`

- `train_yolo_detector`
  - `gpu=DEFAULT_GPU`
  - `timeout=60 * 60`
  - torch cache volume mounted
  - optional warm-container settings

- `train_modal_preview_batch`
  - `gpu=DEFAULT_GPU`
  - `timeout=60 * 60`
  - torch cache volume mounted
  - runs multiple compatible preview jobs sequentially inside one Modal function after one dataset materialization

CPU functions:

- `materialize_image_dataset`
  - no GPU
  - mounts the dataset cache volume at `/cache/model-express/datasets`
  - intended for prewarming/staging

- `profile_image_dataset`
  - no GPU
  - uses a temporary directory and does not write to the durable dataset volume

Important naming caveat: `modal_dispatcher.py` can claim templates such as `analyze_dataset_visuals`, `export_champion`, `champion_demo_prediction`, and `generate_visual_exemplars`, but those job handlers currently run in the local worker process unless they explicitly call a Modal function. "Claimed by a Modal dispatcher" does not always mean "executed inside a Modal container."

## Current Runtime Flow

### Manual Plan Execution

Mission Control's `executePlan()` currently:

1. Starts/ensures a project worker with `gpuType: "modal"`.
2. Sends `/plans/:id/execute` with `{ provider: "modal", gpu_type: "T4" }`.

The Electron worker launcher sees `gpuType: "modal"` and starts one local Python process in dispatcher mode. It sets:

- `GPU_TYPE=modal`
- `MODEL_EXPRESS_MODAL_DISPATCHER=true`
- `MODEL_EXPRESS_MODAL_DISPATCHER_SLOTS=<target worker count>`

The Python worker then enters `ModalDispatcher.run_forever()`, which opens one `modal_app_session()` around its whole loop.

### Automatic Execution

The backend creates worker requirements for auto-follow-up execution. Mission Control watches those requirements and calls `ensureProjectWorker()` for each active requirement. If the provider is Modal, it starts the same dispatcher mode described above.

The backend records worker requirements with:

- `provider`
- `gpu_type`
- `target_count`
- `max_concurrent_jobs`
- dataset cache/materialization policy

The dispatcher polls based on those requirements and uses logical slots to claim jobs. A slot is a backend worker record, not a Modal container.

### Training Provider Selection

`worker.training.providers.run_training_job()` reads `job.config.provider`:

- `local` -> local deterministic simulator
- `modal` -> `run_modal_training()`
- `persistent_gpu` / `persistent_disk` -> persistent GPU provider

`run_modal_training()` selects `train_yolo_detector` when the config looks like YOLO/object detection; otherwise it selects `train_image_classifier`.

## Why Multiple Modal App Cards And Containers Appear

### Multiple App Cards

Likely causes:

1. Each `app.run()` creates an ephemeral Modal app session.
2. `modal_provider.modal_app_session()` keeps one `app.run()` open while a dispatcher process is running.
3. Electron can start one dispatcher process per project/worker key.
4. Direct calls outside `modal_app_session()` also open their own `app.run()` around a single remote invocation.
5. If multiple projects or stale worker processes are active, the dashboard can show multiple ephemeral `model-express-training` cards even though the code uses one `APP_NAME`.

This matches the screenshot: multiple cards with the same app name and "Ephemeral" label are consistent with multiple local `app.run()` sessions.

### Multiple Containers In One App

This is normal Modal autoscaling:

- Each Modal Function has its own autoscaling pool.
- By default, a container handles one input at a time.
- Two active calls to `train_yolo_detector` can produce two containers under one app.

The screenshot line `2 containers, 2 inputs` for `train_yolo_detector` is consistent with two active YOLO function inputs in the same app session.

### Idle Apps Or Idle Containers

There are two different idle states:

- Idle app session: the local dispatcher is still connected through `app.run()`, but no function containers may be active. This is mostly a debugging/noise issue, not necessarily GPU cost.
- Idle function container: a warm container remains alive during `scaledown_window`, or because `min_containers` / `buffer_containers` is set. This can create cost risk.

Current defaults:

- `MODEL_EXPRESS_MODAL_TRAIN_MIN_CONTAINERS` unset -> no minimum warm pool.
- `MODEL_EXPRESS_MODAL_TRAIN_BUFFER_CONTAINERS` unset -> no buffer pool.
- `MODEL_EXPRESS_MODAL_TRAIN_SCALEDOWN_WINDOW_SECONDS` defaults to 10 minutes, or 120 seconds when cost-sensitive defaults are enabled.
- `MODEL_EXPRESS_MODAL_DISPATCHER_IDLE_EXIT_SECONDS` exists, but is unset by default, so a dispatcher can stay open indefinitely after demand is satisfied.

## Can We Use A Single App?

Yes, but the exact meaning matters.

The code already uses a single app definition, but not a single app session. For a cleaner dashboard and debugging model, there are two viable approaches.

### Option A: Deployed App Mode

Deploy `model-express-training` once and invoke deployed functions by name:

- Use `modal deploy` or a guarded `app.deploy()` workflow.
- In the worker, use `modal.Function.from_name("model-express-training", "train_yolo_detector")`.
- Apply per-job resources with `.with_options(gpu=..., memory=..., scaledown_window=...)`.

Pros:

- One persistent app in the Modal dashboard.
- No repeated ephemeral app cards.
- Better for production debugging and historical links.
- Cleaner separation between worker process lifetime and Modal app lifetime.

Cons:

- Requires a deployment step when Modal function code changes.
- Local development has to choose between ephemeral and deployed modes.

Recommendation: make this the production/default path once the training functions stabilize. Keep ephemeral mode as a local dev fallback.

### Option B: Singleton Dispatcher Ephemeral Mode

Keep `app.run()`, but enforce one dispatcher process per project/run and avoid direct per-job `app.run()` calls.

Pros:

- Smaller code change.
- Good enough for local development.

Cons:

- Still shows "Ephemeral" apps.
- Multiple projects still produce multiple app cards.
- Stale local processes can keep sessions visible.

Recommendation: use this as the near-term cleanup while building deployed-app mode.

## Can We Put All Containers In One Container?

Not for parallel GPU training. Modal intentionally scales function inputs across containers. Running multiple YOLO trainings concurrently inside one GPU container would compete for the same VRAM and make the OOM problem worse.

The useful consolidation is not "one GPU container for everything." It is:

- one app/session for observability;
- one container per active GPU training input;
- a remote batch runner for compatible preview trials that should run sequentially after one dataset extraction;
- strict concurrency caps so the scheduler does not create more expensive containers than intended.

## Current GPU And Memory Problem

### The Job GPU Does Not Reliably Control The Modal GPU

Current implementation:

- `DEFAULT_GPU = os.getenv("MODAL_GPU_TYPE", "T4")`
- Modal decorators use `gpu=DEFAULT_GPU`
- `run_modal_training()` sets `os.environ["MODAL_GPU_TYPE"] = config["gpu_type"]` before importing `modal_app.py`

This only works if `modal_app.py` has not already been imported. In the dispatcher path, `modal_app.py` is imported when the app session opens, before arbitrary later jobs are submitted. Once imported, the Modal function resource definition is already fixed.

Effect:

- Job config can say `gpu_type: "A10"`.
- The print/cost telemetry can say A10.
- The actual Modal function may still be registered with the import-time `DEFAULT_GPU`, usually T4.

This should be treated as a correctness bug, not just an optimization issue.

### Manual UI Execution Currently Hardcodes T4

Mission Control manual execution and follow-up execution send:

```json
{ "provider": "modal", "gpu_type": "T4" }
```

This bypasses automation settings' default GPU field. It also makes YOLO experiments start on T4 even when the dataset profile suggests object detection or high-resolution work.

### YOLO Plans Are More Memory Hungry

The deterministic planner creates YOLO jobs with:

- `yolo11n.pt`: batch 8, image size 640
- `yolo11s.pt`: batch 8, image size 640
- `yolo11m.pt`: batch 6, image size 640

These are reasonable training configs, but T4 is a weak default for larger YOLO datasets, larger images, many objects, or medium/larger YOLO models. When the run fails and the system retries the same job on the same resource, it can waste time without learning anything.

### Retry Today Repeats The Same Resource Choice

A Modal training exception is reported to `/jobs/:id/fail` with `retryable: true`. The backend calls `RetryJob()`, which requeues the same job until max attempts are exhausted. It does not patch the job config, GPU type, memory request, batch size, or model settings.

For OOM, this means:

```text
T4 OOM -> requeue same T4 config -> T4 OOM -> requeue same T4 config -> fail
```

That should become:

```text
T4 OOM -> requeue with L4/A10 resource config -> preserve intended batch -> continue
```

### Retry Can Duplicate Successful Training Work

The recent YOLO behavior suggests another failure mode:

```text
attempt 1 starts in Modal container A
attempt 1 hits an error path and reports retryable failure to backend
backend requeues the same logical job
attempt 2 starts in Modal container B, or in another Modal app session
attempt 1 still continues or restarts enough work inside container A
both attempts train or publish progress/artifacts for the same logical model
```

This is plausible because the current retry boundary is the backend job attempt, while the expensive work is inside a remote Modal function. If the local caller, remote function, callbacks, or failure handling disagree about whether an attempt is terminal, the backend can schedule a replacement before the original remote execution has fully stopped. The problem is worse when failures happen after partial training, after artifact writes, or around callback/reporting code rather than before the first batch.

This needs live validation with Modal input ids, backend attempt ids, and artifact ids, but the plan should assume duplicate execution is possible until proven otherwise.

Fix direction:

- Generate a stable `training_attempt_id` for each backend attempt and pass it into the Modal payload.
- Persist the active attempt id on the job before launching the Modal call.
- Require every worker progress, metric, artifact, completion, and failure update to include `job_id` and `training_attempt_id`.
- Backend accepts terminal updates only from the current active attempt.
- Backend rejects stale completion/failure/artifact updates from old attempts after a retry has been queued.
- Modal training code checks a lightweight cancellation/lease endpoint before expensive phases and before artifact publication.
- Retry scheduling records the old attempt as superseded and, where possible, cancels or ignores the old Modal input.

Acceptance:

- A job cannot be marked both failed/requeued and completed by the same attempt.
- A stale attempt cannot overwrite artifacts or metrics from the current attempt.
- If two Modal inputs exist for the same logical job, only one attempt can publish the final model.
- Run history clearly shows `attempt 1 superseded by attempt 2`, not two independent successful trainings.

### System Memory Is Not Requested Explicitly

The Modal training decorators do not set `memory=` or `cpu=`. Modal allows functions to request CPU, memory, and disk explicitly. For large datasets, DataLoader workers, metadata bundles, YOLO label parsing, and export can need more guaranteed system RAM than the default request.

The current code also does not report:

- actual GPU model from inside the container;
- GPU total/used/free memory;
- system memory limit/usage;
- effective DataLoader worker count for YOLO;
- requested vs effective batch size.

## Recommended GPU Selection And Escalation Design

### Principle

Preserve the intended batch size first. Escalate GPU/system memory before shrinking batch size. Only reduce batch size when:

- the cost policy explicitly prioritizes cheapest possible execution;
- the user has capped max GPU tier;
- or all allowed GPU tiers fail.

### GPU Ladder

Use a coarse ladder to avoid too many Modal function variants:

```text
T4 -> L4 -> A10 -> L40S -> A100-40GB -> A100-80GB
```

Default policy:

- Classification, small image sizes: T4 unless profile indicates high risk.
- YOLO/object detection: L4 default for balanced mode.
- YOLO with `yolo11m.pt` or larger, `image_size >= 960`, large datasets, or repeated OOM: A10 or higher.
- Quality mode can start one tier higher.
- Prototype/cheap mode can start lower but must still escalate on OOM if the user allows retries.

Modal pricing as of the checked pricing page:

- T4: $0.000164/sec
- L4: $0.000222/sec
- A10: $0.000306/sec
- L40S: $0.000542/sec
- A100-40GB: $0.000583/sec
- A100-80GB: $0.000694/sec

The L4 step is materially cheaper than failed T4 retries plus long reduced-batch training.

### Use `with_options`, Not Global Env Mutation

Modal documents `Function.with_options()` for call-site resource configuration. The worker should do something like:

```python
configured_function = training_function.with_options(
    gpu=resolved_gpu_type,
    memory=resolved_memory_mb,
    scaledown_window=resolved_scaledown_window,
)
result = configured_function.remote(payload)
```

This fixes the import-time GPU problem and makes the actual Modal resource match the job.

Because each distinct `with_options()` configuration has its own container pool, keep resource choices coarse. Do not create per-job memory values like 19317 MiB; bucket them into values such as 16 GiB, 32 GiB, 48 GiB, and 64 GiB.

### Dataset Profile Resource Hints

The dataset profile already has enough raw facts to compute a first-pass resource policy:

- `task_type`
- `total_images`
- `class_count`
- `image_dimension_stats`
- `width_max`, `height_max`
- `yolo_available`
- `yolo_summary`
- `object_count` / `bbox_count`
- split image counts
- dataset `size_bytes`
- checksum/cache key

Add a backend-owned resource estimator that uses profile facts plus experiment config:

Input:

- task type
- model family/size
- requested batch size
- image size
- total images
- max/median image dimensions
- object count / label count
- dataset archive size
- cost mode
- user GPU cap/default

Output:

```json
{
  "schema_version": "modal_resource_estimate_v1",
  "requested_gpu_type": "T4",
  "effective_gpu_type": "L4",
  "gpu_reason": "YOLO object detection at image_size=640 with batch_size=8 defaults to L4 in balanced mode.",
  "gpu_ladder": ["T4", "L4", "A10", "L40S", "A100-40GB"],
  "memory_mb": 32768,
  "ephemeral_disk_mb": 131072,
  "preserve_batch_size": true,
  "requested_batch_size": 8,
  "max_resource_attempts": 3
}
```

This can be stored in job config under `modal_resources` and copied into run summaries.

I would keep raw profile facts in the dataset profile and compute the actual GPU policy in the backend at execution time. That avoids making old profiles stale when Modal pricing, available GPUs, or project cost mode changes.

### OOM-Aware Retry

Add a structured failure class from the worker:

```json
{
  "retryable": true,
  "failure_class": "gpu_oom",
  "resource_retry": {
    "current_gpu_type": "T4",
    "next_gpu_type": "L4",
    "preserve_batch_size": true,
    "requested_batch_size": 8,
    "attempt": 1
  }
}
```

Backend changes:

- Extend fail-job request with optional `failure_class` and `config_patch`, or add a dedicated `RetryJobWithConfigPatch` store method.
- Validate patches server-side.
- Record an execution event such as `RESOURCE_ESCALATION_QUEUED`.
- Requeue the same job with updated `config.gpu_type`, `config.modal_resources`, and `config.resource_attempt`.
- Stop requeueing the same resource after an OOM.

Worker changes:

- Classify OOM messages from PyTorch, CUDA, Ultralytics, exit 137, and container OOM cases.
- Include requested/effective resource telemetry in failure payload.
- If the backend cannot patch retry config yet, fail fast with a clear message instead of silently retrying the same doomed config.

### Attempt Idempotency And Duplicate-Work Guard

Retries need a separate correctness guard from OOM escalation. Even if retry resource selection is fixed, the system still needs to prevent old attempts from publishing results after the backend has moved on.

Add these fields to every Modal training payload and backend event:

```json
{
  "job_id": "...",
  "attempt": 2,
  "training_attempt_id": "job-uuid:attempt-2",
  "modal_function_call_id": "...",
  "modal_input_id": "..."
}
```

Backend rules:

- `training_attempt_id` is assigned server-side when a job attempt is queued.
- the job row stores the current active attempt id;
- progress, metrics, artifacts, completion, and failure updates are conditional on matching the active attempt id;
- stale attempts are recorded as ignored/superseded, not treated as successful;
- after a terminal success, retryable failures from older attempts are ignored and logged.

Worker rules:

- include attempt identity in every backend callback;
- stop expensive follow-up work when the backend says the attempt is stale;
- before writing final artifacts, ask the backend whether the attempt still owns the job;
- avoid broad exception handling that reports retryable failure after a successful artifact/summary write.

This gives the system a consistent answer to "which Modal input owns this model?" even if Modal, the local dispatcher, or backend retry timing briefly overlap.

### User-Initiated Stop Run

Mission Control should expose a destructive-but-safe "Stop run" action whenever a project or plan has active jobs, active Modal inputs, or active LLM agent calls. The UX should require confirmation because it intentionally terminates paid work.

Expected behavior after confirmation:

1. Freeze scheduling immediately: no new jobs, retries, follow-up plans, dataset profiling, visual analysis, export helper jobs, or LLM agents should start for that run.
2. Mark the run as `CANCELLING_BY_USER` and emit a clear execution event.
3. Cancel queued jobs for the run without starting them.
4. Cancel active Modal function calls and terminate their containers where possible.
5. Shrink/cancel the run's worker requirements so local dispatchers stop polling and the `app.run()` session can close.
6. Cancel in-flight orchestrator and worker LLM calls where possible.
7. Ignore stale callbacks from any work that races with cancellation.
8. Select the best exportable model already found so far and surface it in the existing champion/export area.

Current gaps:

- There is no obvious project/plan-level cancel endpoint in the router.
- Job statuses are `QUEUED`, `ASSIGNED`, `RUNNING`, `SUCCEEDED`, and `FAILED`; there is no first-class `CANCELLED` or `CANCELLING` state.
- Worker requirements can be marked `CANCELLED`, but that does not by itself cancel assigned jobs, Modal calls, or LLM requests.
- Current Modal training uses blocking `.remote()` calls, so the caller does not persist a `modal.FunctionCall` handle before waiting.
- Orchestrator LLM calls use `context.Context`, but run-level cancellation is not threaded through agent execution.
- Worker visual LLM calls have timeouts but no run-cancellation lease check around the request.

Modal cancellation design:

- For long-running Modal work, invoke with `.spawn()` rather than blocking `.remote()`.
- Persist `modal_function_call_object_id` as soon as the function call is spawned.
- Keep existing in-container `modal_function_call_id` and `modal_input_id` telemetry for dashboard correlation.
- On stop, call `modal.FunctionCall.from_id(function_call_object_id).cancel(terminate_containers=True)`.
- Record cancellation result per input: `cancel_requested`, `cancelled`, `not_found`, `already_terminal`, or `cancel_failed`.
- Set the related worker requirement target count to zero or `CANCELLED` so the local dispatcher stops polling and exits its `app.run()` session.
- If Electron still has a project Modal worker process after all known inputs are cancelled, stop that process from Mission Control's worker supervisor.
- Still keep backend attempt-id guards, because cancellation can race with callbacks.

LLM cancellation design:

- Create a run-scoped cancellation context in the orchestrator.
- Pass that context into experiment review, training monitor, experiment planner, memory retrieval, embeddings, and any other LLM path.
- When the user stops the run, cancel the context so in-flight HTTP requests are aborted locally.
- For provider APIs that support server-side cancellation for background/stored responses, persist provider response ids and attempt a provider-side cancel as a best-effort cleanup.
- For worker-side visual LLM calls, add a cancellation lease check before request start, before repair requests, after each retry backoff, and before reporting results.
- Stale LLM responses after cancellation should be recorded as ignored, not allowed to schedule follow-up work.

Best-model-so-far behavior:

- Do not destroy completed training artifacts when stopping a run.
- Select the best successful training summary/evaluation for the run or project using the same champion-selection scoring path already used by stop/budget logic.
- Persist that model as the project champion with selection source `user_cancel_best_available`.
- Call the existing champion export path so the model appears in the normal Export tab.
- If no completed run exists but periodic checkpoints exist, prefer the best validated checkpoint that has enough metadata to export safely.
- If no exportable checkpoint exists, show a clear "run stopped; no exportable model yet" state instead of pretending cancellation produced a model.

Important tradeoff:

- Immediate hard-kill minimizes spend but cannot package an in-memory model that has not been checkpointed yet.
- To make "best model so far" reliable, training should upload/sync best checkpoints at epoch boundaries or YOLO validation milestones before cancellation happens.
- Stop should use already persisted checkpoints/artifacts, not wait indefinitely for a dying container to produce new ones.

Acceptance:

- Clicking Stop, confirming, and waiting for acknowledgement prevents any new jobs or follow-up LLM agent calls for that run.
- Active Modal calls are cancelled by `FunctionCall.cancel(terminate_containers=True)` when their call ids are known.
- User cancellation never goes through the retryable failure path.
- Any late job, metric, artifact, completion, failure, or LLM callback after cancellation is rejected or recorded as stale.
- The Export tab shows the best completed/exportable model found before cancellation, or an explicit no-exportable-model state.

### Backend Cancellation API Contract

Implemented additive endpoints:

- `POST /plans/:plan_id/cancel-active-execution`
  - Cancels open jobs for the plan, cancels active worker requirements, records remote-work cancellation intent, and optionally promotes the best completed/exportable model.
  - Request: `{ "reason": "user_requested", "promote_best_available": true, "terminate_remote_work": true }`.
- `POST /projects/:project_id/cancel-active-executions`
  - Fan-out wrapper over active plan cancellations for the project.
  - TODO: decide whether the UI should expose this as an emergency stop, or keep the visible action plan/run-scoped.
- `POST /jobs/:job_id/modal-call`
  - Worker-only additive callback that persists Modal `FunctionCall` identifiers and resource telemetry so a later stop request can target known remote calls.

Proposed future endpoint:

- `POST /executions/:execution_id/cancel`
  - Preferred long-term target once plan executions/runs have a durable first-class identity.
  - TODO: add only after project-vs-plan-vs-run cancellation semantics are finalized.

Target identity:

- Current primary target: `plan_id`.
- Current project-level target: `project_id`, fanning out to active plan IDs.
- Current `execution_id` response field is compatibility-shaped as `plan:<plan_id>` or `project:<project_id>`, not a durable execution table ID.
- Future primary target: `execution_id`, once a concrete run/execution model exists.

New statuses/events:

- Response statuses: `CANCELLING_BY_USER`, `CANCELLED_BY_USER`.
- Worker requirement status: `CANCELLED`.
- Job status: still `FAILED` for compatibility, with additive config fields `cancel_requested: true`, `failure_class: "cancelled"`, `cancel_reason`, and Modal cancellation metadata.
- Execution events: `EXECUTION_CANCELLATION_REQUESTED`, `EXECUTION_CANCELLED`, `REMOTE_WORK_CANCEL_REQUESTED`, `REMOTE_WORK_CANCEL_FAILED`, `JOB_STALE_CALLBACK_IGNORED`.

Response payload:

```json
{
  "execution_id": "plan:plan_1",
  "target": { "project_id": "project_1", "plan_id": "plan_1" },
  "status": "CANCELLED_BY_USER",
  "message": "Cancelled plan plan_1: 3 queued job(s), 2 active job(s), 1 worker requirement(s).",
  "queued_jobs_cancelled": 3,
  "active_jobs_marked_cancelling": 2,
  "already_terminal_jobs": 1,
  "modal_calls": [
    {
      "job_id": "job_1",
      "training_attempt_id": "job_1:attempt-2",
      "modal_function_call_object_id": "fc-...",
      "modal_function_call_id": "fc-...",
      "modal_input_id": "in-...",
      "cancel_status": "cancel_requested"
    }
  ],
  "worker_requirements": [
    { "id": "worker_requirement_1", "status": "CANCELLED" }
  ],
  "best_available_model": {
    "job_id": "job_0",
    "exportable": true,
    "champion_selection_source": "user_cancel_best_available"
  },
  "compatibility": {
    "job_cancelled_status_enabled": false,
    "late_callbacks_ignored_by_attempt_id": true
  },
  "jobs": []
}
```

Fields the UI would need:

- `execution_id`, `project_id`, `plan_id`, current cancellation status, and `message`.
- Counts for queued, active, cancelled, failed-to-cancel, and already-terminal work.
- Per-Modal-call cancel status plus `modal_function_call_object_id`, `modal_function_call_id`, and `modal_input_id` when known.
- Worker requirement status and target counts after cancellation.
- Best available model/export status, or a clear no-exportable-model reason.
- Compatibility flags for whether `CANCELLED` job status is enabled and whether stale callbacks are guarded.
- TODO: add a durable `execution_id` to active run state before the UI treats plan cancellation as a named run history object.

Compatibility notes:

- Add endpoints and fields without changing existing plan, project, job, summary, or evaluation response shapes.
- Keep current `FAILED`/`SUCCEEDED` consumers working until `CANCELLED` is explicitly feature-flagged through `MODEL_EXPRESS_CANCELLED_JOB_STATUS=1`.
- User cancellation must not use retryable failure handling.
- Late Modal, worker, or LLM callbacks must be accepted as 2xx when possible but ignored if their `training_attempt_id` is stale or the job is already terminal.
- Project-level cancellation currently means all active plan work discoverable from open jobs or active worker requirements.
- TODO: add LLM provider-side cancellation ids where supported; current safeguards prevent late results from scheduling job completions/retries, but cannot always abort an already in-flight provider HTTP request.

### Batch-Size Telemetry

Add to every training summary/evaluation:

- `requested_batch_size`
- `effective_batch_size`
- `batch_size_policy`: `preserved`, `auto_reduced`, `user_reduced`, `failed_before_batch`
- `resource_attempt`
- `requested_gpu_type`
- `effective_gpu_type`
- `modal_function_gpu_config`

For classifier jobs, `train_loader.batch_size` gives the effective loader batch size. For YOLO, inspect the Ultralytics trainer result/callback state after training and include the value it actually used.

## Current Modal Batch Runner

The existing `remote_gpu_cost_reduction_plan.md` is now partly stale. The current code does include:

- dispatcher grouping for compatible preview jobs;
- `MODEL_EXPRESS_MODAL_BATCH_RUNNER`;
- `MODEL_EXPRESS_YOLO_BATCH_PREVIEW`;
- `train_modal_preview_batch`;
- one dataset materialization reused by several sequential preview jobs inside a single Modal function.

Current limitations:

- The batch runner is disabled unless `MODEL_EXPRESS_MODAL_BATCH_RUNNER=1`.
- YOLO preview batching is separately disabled unless `MODEL_EXPRESS_YOLO_BATCH_PREVIEW=1`.
- Only preview-tier jobs with matching batch keys are grouped.
- `MODEL_EXPRESS_MODAL_PREVIEW_TIER_METADATA` is disabled by default, so jobs may not get `training_tier: "preview"` unless cost policy adds it.
- If remote batching fails or is unsupported, it falls back to single-job Modal submissions.

Recommendation:

- Validate the batch runner with small classification and YOLO preview plans.
- Turn it on by default for prototype/balanced preview runs once telemetry proves correct.
- Keep full/champion validation as single-job runs unless there is a clear reason to batch them.

## Dataset Materialization And Cache Findings

There are two cache paths:

- `materialize_image_dataset` writes to Modal dataset volume at `/cache/model-express/datasets`.
- Training functions default to `/tmp/model-express/training-datasets`.

Training functions only mount `training_volume_mounts`, which currently come from the torch cache volume, not the dataset cache volume. Therefore, dispatcher prewarm can materialize a durable dataset volume that training does not read unless env/config changes both the training cache root and the volume mount strategy.

Recommendation:

- Treat remote batch runner as the primary near-term fix for repeated download/extract.
- For prewarm, either:
  - mount the dataset volume into training functions and set training cache root to the same volume path, or
  - remove/disable prewarm when it cannot be reused by training.
- Add telemetry checks: prewarm cache root, training cache root, cache key, cache hit/miss, and reuse status.

## Container And Cost Controls

Current controls:

- Dispatcher slots default to 2 and cap at 8.
- Worker requirements can carry `max_concurrent_jobs`.
- Modal training functions have optional `min_containers`, `buffer_containers`, `scaledown_window`.
- Cost mode can cap concurrency and skip some jobs.

Gaps:

- Modal function decorators do not set `max_containers`.
- Multiple dispatcher processes can each submit work.
- `MODEL_EXPRESS_MODAL_DISPATCHER_IDLE_EXIT_SECONDS` exists but is unset by default.
- The UI does not expose a clear "stop Modal dispatcher for this project" action.
- App/session identity is not persisted in backend run summaries.
- There is no project/plan-level "stop this run and kill paid work" action.

Recommendations:

- Set a conservative default idle exit, for example 120-300 seconds after zero demand.
- Add a Mission Control stop action for project workers/dispatchers.
- Add a separate Mission Control "Stop run" action that cancels active work, not just idle dispatchers.
- Include dispatcher PID, worker ID, Modal app id/dashboard URL, function call id, and input id in execution events.
- Consider `max_containers` via `with_options()` or base decorators for training functions when project-level concurrency must be hard-capped.
- Show "app session is open" separately from "GPU containers are running" in UI/debug logs.

## Other Modal Improvement Areas

### Separate CPU And GPU Images

`profile_image_dataset` and `materialize_image_dataset` use the same heavy image as GPU training. A lean CPU image for profile/materialization could reduce cold starts and image build time.

Suggested split:

- CPU data image: `boto3`, `pillow`, `pyyaml`, dataset helpers.
- GPU training image: `torch`, `torchvision`, `ultralytics`, export dependencies.

Both can still live in one Modal app.

### Pin Training Dependencies

The Modal image currently installs unpinned `torch`, `torchvision`, and `ultralytics`. Pin versions to reduce training drift, reproducibility issues, and surprise memory changes.

### Add Modal Secrets

S3 credentials are passed in the function payload. Move stable credentials to Modal Secrets where possible. Keep payload overrides for local tunnels/endpoints if needed.

### Improve YOLO Runtime Knobs

Add explicit YOLO resource knobs:

- `workers`
- `cache=False` unless intentionally caching in RAM/disk
- `device=0`
- `amp=True` when supported
- explicit output cleanup

Record all of these in `preprocessing_summary.training_hyperparameters`.

### Persist GPU Metrics

Modal exposes GPU memory used, utilization, power, and temperature in the dashboard. Inside the container, add a lightweight `torch.cuda`/`nvidia-smi` snapshot at:

- start of training;
- after model load;
- first batch;
- after OOM catch when possible;
- final summary.

Do not over-sample; one snapshot per phase is enough.

### Keep Pricing Fresh

`_modal_gpu_price_per_second()` currently matches the checked Modal pricing page for the listed GPUs, but hard-coded prices will drift. Add a doc comment with the pricing URL and a test/table review reminder, or move cost estimates into config.

## Proposed Implementation Sequence

### Phase 0: Observability Before Behavior Changes

Add:

- `modal_resources` block to job config and training summaries.
- `requested_gpu_type`, `effective_gpu_type`, `memory_mb`, `ephemeral_disk_mb`.
- requested/effective batch size.
- dispatcher PID, worker ID, Modal app id/dashboard URL if accessible.
- OOM failure class fields in worker failure payloads.
- stage telemetry default-on for Modal training, or at least default-on for failures.

Acceptance:

- A failed YOLO job shows exactly which GPU was requested, which GPU Modal ran, which batch size was requested, and whether it failed before/after first batch.

### Phase 1: Make Job GPU Control The Actual Modal GPU

Change worker submission to use `Function.with_options()` for training functions. Stop relying on per-job `MODAL_GPU_TYPE` environment mutation.

Acceptance:

- A job with `gpu_type: "L4"` runs a Modal function variant configured with L4.
- A later job with `gpu_type: "A10"` in the same dispatcher process runs A10, not the prior GPU.
- Run summary and Modal dashboard agree.

### Phase 2: Remove T4 Hardcoding From Mission Control

Use automation settings or backend defaults for manual execution:

- `provider = automationSettings.default_training_provider`
- `gpu_type = automationSettings.default_gpu_type`
- if blank and dataset is YOLO/object detection, backend resource estimator selects L4/A10.

Acceptance:

- Changing the GPU field in Automation Settings actually changes future manual Modal jobs.
- YOLO plans no longer start on T4 unless explicitly selected.

### Phase 3: Backend Resource Estimator

Add a deterministic resource estimator for Modal jobs.

Acceptance:

- YOLO/object-detection datasets get a resource recommendation before training starts.
- The estimate is visible in job config, execution events, and Mission Control.
- Cost mode can cap the maximum GPU tier.

### Phase 4: OOM-Aware GPU Escalation

Add retry-with-config-patch support and worker OOM classification.

Acceptance:

- T4 OOM requeues the same job with L4 or A10, preserving batch size.
- A second OOM escalates again until the allowed max tier.
- The system never repeats the same GPU/batch combination after a confirmed OOM.
- Exhausted escalation fails with an actionable message.

### Phase 5: Retry Idempotency And Duplicate Training Guard

Add attempt ownership to backend job attempts and every Modal callback.

Acceptance:

- duplicate Modal inputs for the same job cannot both publish final artifacts;
- stale retries cannot mark a completed job failed;
- stale completions cannot overwrite the active attempt;
- run history makes superseded attempts explicit.

### Phase 6: User Stop Run And Resource Cancellation

Add a confirmed run-stop flow across Mission Control, orchestrator, worker, Modal, and LLM clients.

Acceptance:

- Stop marks the run cancelling, prevents new jobs/retries/follow-up agents, and cancels queued jobs.
- Active Modal calls spawned for the run are cancelled with container termination when possible.
- In-flight LLM calls receive a cancelled context or cancellation lease and cannot schedule new work after returning.
- Cancellation is terminal and does not enter retryable failure handling.
- The best exportable model already produced is selected as champion and shown in the normal Export tab.

### Phase 7: App Consolidation

Pick one of:

- Deployed app mode for production.
- Singleton dispatcher ephemeral mode for local development.

Acceptance:

- Normal training for one project appears under one app/session in the Modal dashboard.
- The UI exposes which app/session/function input belongs to each job.
- Idle dispatchers exit or can be stopped.

### Phase 8: Batch Runner And Cache Validation

Validate and enable remote preview batching.

Acceptance:

- A compatible preview plan with 2-3 jobs performs one dataset materialization in one Modal container.
- Each job still reports independent metrics, summary, evaluation, completion/failure.
- YOLO preview batching is enabled only after a smoke test proves correct split/subset handling.

## P0/P1 Task List

P0:

- Replace actual training invocation with `with_options(gpu=resolved_gpu_type)`.
- Add `memory=` support through coarse buckets.
- Remove Mission Control's hardcoded T4 request.
- Persist requested/effective GPU and batch-size telemetry.
- Add OOM classification and explicit "same resource retry blocked" guard.
- Add training attempt ids and reject stale retry/completion/artifact callbacks.
- Add run cancellation states and guarantee user cancellation does not retry.
- Switch long-running Modal calls to `.spawn()` and persist cancellable FunctionCall object ids.
- Add Mission Control Stop Run UI and best-available champion/export selection.

P1:

- Backend Modal resource estimator.
- OOM retry config patching.
- Default YOLO balanced-mode GPU to L4.
- Add provider-side cancellation for stored/background LLM responses where supported.
- Add dispatcher idle exit default.
- Add Modal app/session/function links to Mission Control.

P2:

- Deployed app mode.
- Split CPU and GPU Modal images.
- Enable Modal preview batching by default for preview tiers.
- Fix or remove dataset-volume prewarm if it is not reused by training.
- Modal Secrets for stable credentials.

## Open Questions For Live Validation

- Do current live runs use one Electron worker process or multiple processes for the same project?
- Does the Modal dashboard show old ephemeral apps because local dispatchers are still running, or because direct per-job `app.run()` calls are being made?
- What exact failure messages are produced by the recent YOLO crashes: CUDA OOM, system memory OOM, container exit 137, disk, or timeout?
- In the duplicate-training cases, do two backend attempts share one Modal input id, or are there multiple Modal input ids for the same logical job?
- Can a Modal attempt report retryable failure after writing model artifacts or completion telemetry?
- Which user-visible unit should Stop Run target first: project, plan execution, or a new explicit run/execution id?
- Are best checkpoints currently uploaded often enough that a hard cancellation can still provide a useful model?
- Does Ultralytics ever auto-adjust batch size in current configs, or are smaller batches coming from user/planner config after failures?
- Are current dataset prewarm telemetry paths equal to training telemetry paths in real runs?
- What cost mode should be the default for YOLO: balanced L4, or prototype T4 with escalation?

## Recommended Immediate Operating Settings

Until code changes land:

- Set Automation Settings GPU to `L4` or `A10` before YOLO runs.
- Avoid running multiple projects' Modal workers at once.
- Set `MODEL_EXPRESS_MODAL_DISPATCHER_IDLE_EXIT_SECONDS=180` for local debugging.
- Keep `MODEL_EXPRESS_MODAL_TRAIN_MIN_CONTAINERS` and `MODEL_EXPRESS_MODAL_TRAIN_BUFFER_CONTAINERS` unset unless intentionally prewarming.
- Do not rely on `MODAL_GPU_TYPE` changing per job in a long-running dispatcher; restart the dispatcher after changing it.

These settings reduce the worst confusion but do not fully solve the import-time GPU bug. The durable fix is `with_options()`.
