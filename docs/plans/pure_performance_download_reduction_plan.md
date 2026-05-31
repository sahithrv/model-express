# Pure Performance And Download Reduction Plan

## Purpose

This plan focuses on pure performance improvements for Model Express, with dataset download reduction as the primary goal.

The current system has a good control-plane boundary:

```text
Mission Control uploads dataset metadata
Go orchestrator owns plans, jobs, leases, retries, validation, and state
Python workers execute assigned jobs
Modal runs remote training
```

The main performance problem is that the dataset artifact is uploaded once, then repeatedly downloaded and extracted by separate jobs. The highest-value work is to make dataset materialization a shared, content-addressed operation instead of a per-job side effect.

## Current Download Model

Today, object storage serves roughly:

```text
bytes_served ~= dataset_archive_size * materializing_job_count
```

For a single dataset, `materializing_job_count` can include:

- local profile job
- Modal profile job
- visual analysis job
- visual exemplar job
- every Modal training experiment
- champion demo/export paths that need dataset files

The costly paths are:

- Electron upload streams one zip to S3 from `apps/mission-control/electron/main.cjs`.
- Local worker jobs download and extract via `services/worker/worker/jobs.py`.
- Modal training downloads and extracts per run via `services/worker/worker/training/modal_app.py`.
- Modal profiling downloads and extracts per profile job via `services/worker/worker/training/modal_app.py`.
- Job-scoped cache cleanup prevents reuse across default local jobs in `services/worker/worker/datasets/cache.py`.

For a plan with 8 Modal experiments and a 10 GB dataset, the current system can ask object storage to serve about 80 GB just for training startup, before profiling, visual analysis, or retries.

## Download Reduction Contract

The target behavior should be:

```text
bytes_served ~= dataset_archive_size * cache_miss_count
```

For the common case:

```text
cache_miss_count = 1 per dataset checksum per execution environment
```

That means:

- Upload still happens once.
- The first Modal job or pre-stage task downloads the archive once.
- The archive is extracted once into a shared Modal cache or volume.
- Later Modal jobs use the materialized path and do not download the archive again.
- Local jobs reuse a content-addressed cache when safe.
- Concurrent jobs coordinate through a lock so they do not all miss the cache at the same time.

The cache key should be the dataset checksum, not only dataset ID:

```text
dataset_cache_key = checksum_sha256 || storage_uri fingerprint
```

Using checksum lets repeated project uploads, retries, and follow-up plans reuse the same materialized dataset when the bytes are identical.

## Priority 1: Modal Dataset Cache Or Materialization By Checksum

### Goal

Ensure Modal training jobs do not download and extract the same dataset archive per experiment.

### Proposed Shape

Add a Modal dataset materialization helper:

```text
ensure_dataset_materialized(dataset_id, storage_uri, checksum_sha256)
```

This helper should:

1. Compute a safe cache key from checksum.
2. Check for a completed extracted dataset marker.
3. Acquire a per-checksum materialization lock.
4. Re-check the completed marker after acquiring the lock.
5. Download the archive only if the cache is absent.
6. Extract into a staging directory.
7. Atomically publish the extracted directory and completion marker.
8. Return the extracted dataset path to training/profile code.

### Modal Storage Options

Preferred:

- Use a Modal Volume keyed by dataset checksum.
- Store:
  - `archive.zip` if useful for validation/debugging
  - `extracted/`
  - `manifest.json`
  - `.complete`

Fallback:

- Use a per-container temp cache only for a single job. This does not solve repeated downloads across jobs, so it should be treated as a fallback, not the target.

### Why This Reduces Downloads

Current:

```text
8 Modal experiments -> 8 S3 downloads -> 8 archive extractions
```

Target:

```text
8 Modal experiments -> 1 S3 download on first cache miss -> 1 extraction -> 8 cache reads
```

If several jobs start together, the lock prevents a thundering herd where every job decides the cache is absent and downloads the same archive.

### Metrics To Add

Track:

- `dataset_materialization_cache_hit`
- `dataset_materialization_cache_miss`
- `dataset_materialization_bytes_downloaded`
- `dataset_materialization_extract_seconds`
- `dataset_materialization_wait_seconds`
- `dataset_checksum`
- `storage_uri_fingerprint`

Success metric:

```text
bytes_downloaded_from_s3_per_plan <= dataset_archive_size + small_overhead
```

for a fully warm Modal environment.

## Priority 2: Single Async Modal Dispatcher

### Goal

Stop representing Modal parallelism as many local blocking Python worker processes.

The dispatcher should not replace the Go orchestrator. It should be a provider-specific execution adapter.

### Responsibility Split

The Go orchestrator still owns:

- canonical job queue
- validation
- leases and retries
- provider/project concurrency limits
- run state
- metrics, summaries, evaluations
- agent decisions and automation policy

The Modal dispatcher owns only:

- claiming runnable Modal jobs
- submitting Modal calls
- forwarding metrics and final results
- renewing leases while remote calls are active
- enforcing the capacity granted by the orchestrator

### Why This Reduces Downloads

The dispatcher does not reduce downloads by itself. It reduces downloads when paired with materialization because it can enforce this rule:

```text
No batch of Modal jobs starts until the dataset checksum is materialized or one materialization task is in progress.
```

Instead of 8 local workers simultaneously calling Modal and causing 8 remote jobs to race into `download_s3_uri`, the dispatcher can:

1. Claim a batch of Modal jobs for the same dataset checksum.
2. Call `ensure_dataset_materialized` once.
3. Submit the training jobs with `materialized_dataset_path`.
4. Limit concurrent cold-cache starts.

This turns the dispatcher into a traffic shaper, not a second orchestrator.

### Staged First Slice

The first safe slice is an execution-adapter-only Modal materialization gate:

- Worker Modal training/profile functions still execute jobs claimed through the existing orchestrator worker poll/complete/fail APIs.
- Modal jobs call a checksum-keyed `ensure_dataset_materialized(...)` helper before loading image data.
- The helper uses a shared Modal dataset cache volume when available, a completion marker, and an atomic per-cache-key lock so concurrent same-dataset starts converge on one cold download/extract.
- Cache telemetry is attached to Modal summaries/evaluations for backend/operator visibility without making the worker own scheduling state.

Next slice: add an explicit Modal dispatcher/batched runner process that registers as a worker, claims only orchestrator-granted Modal capacity, groups claimed jobs by dataset checksum, pre-warms one materialization per checksum, and then submits remote Modal calls while renewing leases. That process should remain an adapter and must not create plans/jobs or bypass orchestrator validation.

### Concurrency Policy

Add policy at the orchestrator level:

```text
max_modal_jobs_per_project
max_modal_jobs_per_dataset_checksum
max_cold_dataset_materializations
```

Recommended defaults:

- 1 cold materialization per checksum
- 2 to 4 concurrent Modal trainings per project
- lower concurrency while the dataset cache is cold

## Priority 3: Compact Frontend Data Model

### Goal

Reduce UI and backend churn caused by broad polling and large JSON payloads.

This does not directly reduce dataset downloads. It reduces the lag that appears when workers and LLM hooks produce many status changes.

### Proposed Shape

Add a compact project overview endpoint:

```text
GET /projects/:id/overview
```

It should return:

- project
- latest dataset summary
- latest plan summary
- job counts by status
- active worker counts by provider
- latest run summaries
- latest decision summary
- champion summary
- latest execution events

Move heavy data behind detail endpoints:

- full training evaluation
- full confusion matrix
- full agent invocation trace
- full decision payload
- all historical jobs

### Why This Helps

Mission Control currently refreshes many endpoints every few seconds. When Modal jobs report metrics or LLM hooks store large payloads, the UI repeatedly pulls and renders more data than the active view needs.

## Priority 4: Postgres-Backed LLM And Automation Task Queue

### Goal

Move post-training monitor/planner work out of ad hoc goroutines and into durable bounded tasks.

This does not directly reduce dataset archive downloads. It prevents LLM calls, large audit writes, and follow-up scheduling from piling up unpredictably when many jobs finish together.

### Proposed Shape

Create durable task rows for:

- training monitor evaluation
- experiment planner review
- planner validation retry
- follow-up plan creation
- automatic execution request
- visual analysis trigger evaluation

Workers claim tasks with `FOR UPDATE SKIP LOCKED`.

### Download-Related Guardrail

Automation tasks that create jobs should check:

```text
dataset_materialization_status
```

before scheduling large batches. If the dataset is not materialized for Modal, they can enqueue materialization first or mark the plan as waiting for dataset cache warmup.

## Priority 5: Event-Specific SSE With Cursors

### Goal

Replace broad refresh-on-event with resource-specific updates.

This does not reduce dataset downloads. It reduces dashboard request load and renderer work.

### Proposed Shape

Events should include:

- event ID cursor
- resource type
- resource ID
- changed fields or summary payload
- project ID
- plan ID
- job ID when relevant

Mission Control should update only the affected slice:

```text
JOB_METRIC_REPORTED -> refresh selected job metrics only if selected
JOB_COMPLETED -> refresh job summary and run summary
AGENT_INVOCATION_RECORDED -> refresh agent summary only if Agents tab is active
WORKER_REQUIREMENT_UPDATED -> refresh worker requirement slice
```

## Priority 6: Move Upload Packaging Off Electron Main

### Goal

Avoid desktop UI lag while walking, zipping, CRCing, hashing, and uploading large datasets.

This does not reduce remote downloads served after upload. It improves upload responsiveness and makes upload cancellation/progress possible.

### Proposed Shape

Move dataset packaging/upload to:

- a worker thread, or
- a child process, or
- a dedicated upload service process

Add:

- progress events
- cancellation
- resumable multipart upload
- upload manifest cache

## Download Reduction Implementation Sequence

Do the download-reduction work before UI and LLM polish:

1. Add instrumentation around every dataset download path.
2. Add Modal materialization helper with checksum cache key.
3. Add cache locking and completion markers.
4. Change Modal training/profile code to use materialized paths.
5. Add dispatcher or batched Modal runner that warms materialization once per checksum.
6. Add orchestrator-visible dataset materialization status.
7. Add concurrency limits for cold-cache starts.
8. Add local content-addressed cache for local profile/visual/exemplar jobs.

## Acceptance Criteria

For a plan with one dataset and N Modal experiments:

- Cold cache:
  - object storage serves the dataset archive once per checksum per Modal cache environment
  - all N jobs reuse the same extracted dataset
  - concurrent starts do not produce duplicate downloads
- Warm cache:
  - object storage serves zero dataset archive bytes for training startup
  - jobs begin from the materialized path
- Retries:
  - retrying a failed training job reuses the same materialized dataset
- Follow-up plans:
  - follow-up jobs for the same dataset checksum reuse the same materialized dataset
- Observability:
  - Mission Control or backend logs can show cache hit/miss, bytes downloaded, extraction seconds, and waiting time

## Open Questions

1. Should the Modal materialized cache be considered durable enough for normal use, or should the orchestrator treat it as best-effort and always tolerate cache misses?
2. Should cache eviction be manual first, then automated later by size/age?
3. Should the Go orchestrator own dataset materialization task state, or should the Modal dispatcher expose it through worker heartbeats and job summaries?
4. Should local dataset cache persistence become the default with TTL/size cleanup, or remain opt-in?
5. Should future dataset upload store raw objects or WebDataset shards instead of a single zip archive?

## Summary

Only priorities 1 and 2 directly reduce the number of dataset downloads that object storage must serve.

The key performance shift is:

```text
per-job dataset download
```

to:

```text
per-checksum dataset materialization
```

The remaining priorities reduce the secondary lag from broad UI refresh, large JSON payloads, LLM automation bursts, and Electron main-thread upload work.
