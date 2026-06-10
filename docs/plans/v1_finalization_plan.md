# Model Express V1 Finalization Plan

Date: June 10, 2026
Target release window: June 15, 2026

This plan focuses on getting the current implementation reliable enough for a v1 release without broad rewrites. The goal is to prevent run-stopping failures, local machine lag, runaway disk usage, and slow UI behavior while accepting that small product bugs may remain.

## Scope

Primary goal:

- Make local operation predictable when Postgres, MinIO, MLflow, the Go orchestrator, Mission Control, and workers are running on the same machine.
- Keep default behavior conservative. Expensive or autonomous modes should require explicit opt-in.
- Add caps, pagination, cleanup, and lightweight observability before attempting architecture changes.

Non-goals for the next five days:

- Replacing the current worker/orchestrator architecture.
- Building a complete production scheduler.
- Converting all local training to real PyTorch training.
- Perfecting every dashboard view.

## Local Run Path Map

Current local path, based on code inspection:

1. User creates a project and chooses a dataset folder in Mission Control.
2. Electron uploads the dataset to MinIO as a generated ZIP archive:
   - `apps/mission-control/electron/main.cjs:770`
   - `apps/mission-control/electron/zip-stream.cjs:12`
3. Mission Control creates the dataset record in the orchestrator.
4. Mission Control starts a Python profiling worker for the project:
   - `apps/mission-control/src/App.tsx:1863`
   - `apps/mission-control/electron/main.cjs:437`
5. The Python worker registers itself and polls every 5 seconds:
   - `services/worker/worker/main.py:12`
   - `services/worker/worker/main.py:41`
6. The worker claims a `profile_dataset` job from the orchestrator:
   - `services/orchestrator/internal/api/handlers.go:9667`
   - `services/orchestrator/internal/store/postgres.go:914`
7. The worker downloads the ZIP from MinIO, extracts it, profiles images, imports metadata, and posts the profile:
   - `services/worker/worker/jobs.py:54`
   - `services/worker/worker/jobs.py:162`
   - `services/worker/worker/datasets/profiler.py:65`
   - `services/worker/worker/datasets/metadata_discovery.py:72`
8. The backend creates an initial experiment plan after profiling:
   - `services/orchestrator/internal/api/handlers.go:10207`
   - `services/orchestrator/internal/agents/planner.go:41`
9. If auto-execution is enabled, the backend executes the plan and queues jobs:
   - `services/orchestrator/internal/api/handlers.go:10265`
   - `services/orchestrator/internal/api/handlers.go:10277`
10. Mission Control supervises worker requirements every 3 seconds and starts workers or a Modal dispatcher:
    - `apps/mission-control/src/App.tsx:1595`
    - `apps/mission-control/src/App.tsx:1817`
11. Workers claim `train_experiment`, `analyze_dataset_visuals`, `export_champion`, and demo jobs.
12. Training reports metrics, summaries, and evaluations back to the orchestrator:
    - Local training path is currently a simulator in `services/worker/worker/training/local.py:10`.
    - Modal training is remote-dispatched in `services/worker/worker/training/modal_provider.py:71`.
13. Backend stores metrics/evaluations, may run monitor/planner/reviewer decisions, selects a champion, and may queue export/demo work.
14. Mission Control refreshes project state every 10 seconds, keeps an activity SSE stream open, and refreshes on activity events:
    - `apps/mission-control/src/App.tsx:78`
    - `apps/mission-control/src/App.tsx:1402`
    - `apps/mission-control/src/App.tsx:1752`
    - `apps/mission-control/src/App.tsx:1760`

## Process And Runtime Boundaries

Local processes that can be active at once:

- Docker Postgres with pgvector from `infra/compose.yaml`.
- Docker MinIO from `infra/compose.yaml`.
- Docker MLflow from `infra/compose.yaml`.
- Go orchestrator process from `services/orchestrator/cmd/orchestrator/main.go`.
- Vite dev server and/or Electron Mission Control.
- Electron main process.
- Python worker child processes launched by Electron.
- Modal dispatcher thread pool inside a Python worker when Modal mode is enabled.
- Remote Modal containers for real training, dataset profile, materialization, or batching.
- LLM and embedding API calls from the orchestrator when enabled.
- Local filesystem work for ZIP creation, ZIP extraction, dataset scanning, logs, exports, and caches.

Unknown or partially unclear:

- There is no single documented "local safe profile" that pins every process limit across Docker, frontend, orchestrator, and worker.
- It is not obvious from the UI whether the user is running a dev profile or a v1-safe profile.
- Docker resource limits are not declared in `infra/compose.yaml`, so host Docker Desktop settings currently dominate.

## Performance Risk Ranking

### Critical

1. No v1-safe local resource profile is enforced.

Evidence:

- `.env.local` currently enables autonomous behavior, auto execution, Modal as the default training provider, a six-slot Modal dispatcher, memory embeddings, dataset shards, and balanced cost mode. This may be useful for development, but it is too aggressive as a default local release profile.
- Electron can spawn up to 8 worker slots through `normalizeWorkerCount` in `apps/mission-control/electron/main.cjs:630`.
- Planner defaults recommend 2 workers for balanced and up to 3 for best quality in `services/orchestrator/internal/agents/planner.go:256`.
- Auto-execute builds a default execution request without `max_concurrent_jobs` in `services/orchestrator/internal/api/handlers.go:14252`.

Impact:

- CPU, disk, DB, network, and process count can spike because planning, worker supervision, polling, and remote dispatch are all active.

Safe fix:

- Add a v1 local-safe mode and make it the default:
  - `MODEL_EXPRESS_AUTO_EXECUTE_PLANS=false`
  - `MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS=false`
  - `MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS=1`
  - `MODEL_EXPRESS_MAX_AUTO_WORKERS=1`
  - `MODEL_EXPRESS_LOCAL_MAX_WORKERS=1`
  - `MODEL_EXPRESS_LOCAL_MAX_CONCURRENT_JOBS=1`
  - `MODEL_EXPRESS_MODAL_DISPATCHER_SLOTS=1` unless user explicitly selects remote parallelism.

2. Dataset profiling can monopolize disk and CPU on large datasets.

Evidence:

- `profile_image_folder` recursively scans image class dirs and opens every image in `services/worker/worker/datasets/profiler.py:65`.
- Artifact detection separately walks the dataset tree in `services/worker/worker/datasets/profiler.py:359`.
- YOLO dimension stats can reopen all image files in `services/worker/worker/datasets/profiler.py:333`.
- Metadata discovery has caps, but still walks the tree up to large limits in `services/worker/worker/datasets/metadata_discovery.py:72`.
- Visual sample generation is bounded but can still inspect up to 2,500 image records in `services/worker/worker/datasets/visual_sampling.py:50`.

Impact:

- CPU, disk, and memory pressure during import/profile. This is likely the most visible source of local lag.

Safe fix:

- Add a profile cache keyed by dataset checksum plus profile schema version.
- Add caps to the initial profile scan:
  - max image headers opened by default: 25,000
  - max artifact files inspected by default: 25,000
  - max corrupt image paths retained: 25
  - max profile JSON size target: under 1 MB
- Return warnings when caps are hit instead of scanning indefinitely.

### High

1. Backend and frontend state endpoints are mostly unpaginated.

Evidence:

- Project jobs are unpaginated in `services/orchestrator/internal/api/handlers.go:1109` and `services/orchestrator/internal/store/postgres.go:1107`.
- Job metrics are unpaginated in `services/orchestrator/internal/api/handlers.go:1160`.
- Training summaries and evaluations are unpaginated in `services/orchestrator/internal/api/handlers.go:1368`.
- Mission Control stores returned arrays in state and only slices for display in many places.

Impact:

- DB and UI work grow with every run. Long sessions can become slow even when only a few rows are visible.

Safe fix:

- Add `limit` and `offset` to jobs, summaries, evaluations, and metrics.
- Default UI fetches:
  - jobs: latest 100
  - summaries: latest 100
  - evaluations: latest 50
  - metrics: latest 200 per selected job, ordered by epoch

2. Activity SSE does repeated broad store reads.

Evidence:

- Activity stream defaults to every 2 seconds in `services/orchestrator/internal/api/activity.go:24`.
- Every stream tick loads execution events, all project jobs, agent invocations, and agent decisions in `services/orchestrator/internal/api/activity.go:122`.
- The frontend also has a 10 second live refresh and event-triggered refreshes in `apps/mission-control/src/App.tsx:1752` and `apps/mission-control/src/App.tsx:1776`.

Impact:

- Sustained DB and render churn while the UI is open.

Safe fix:

- Use one primary live mechanism:
  - active run: SSE interval 3,000-5,000 ms, live refresh 10,000 ms
  - idle project: SSE interval 10,000 ms or disconnect, live refresh 30,000 ms
- Coalesce event-triggered refreshes to at most once every 3 seconds.

3. Postgres job polling filters too late.

Evidence:

- `PollJob` selects queued project jobs and filters provider/template in Go after scanning rows in `services/orchestrator/internal/store/postgres.go:993`.

Impact:

- With many queued jobs and multiple providers, workers can repeatedly scan jobs they cannot run.

Safe fix:

- Push template filtering into SQL immediately.
- Add a generated/provider expression or precomputed provider column for `config->>'provider'`.
- Limit candidate rows per poll, then retry with backoff when none match.

4. Worker process/log behavior can accumulate.

Evidence:

- Direct worker logs "There are no jobs." every 5 seconds in `services/worker/worker/main.py:49`.
- Electron appends worker stdout/stderr to JSONL without rotation in `apps/mission-control/electron/main.cjs:675`.

Impact:

- Low CPU but steady disk churn and noisy logs during idle sessions.

Safe fix:

- Make worker poll interval configurable and jittered.
- Log idle state at most once per minute.
- Rotate `artifacts/logs/*.jsonl` by size, for example 10 MB per file and 5 retained files.

5. Docker service resources are unconstrained.

Evidence:

- `infra/compose.yaml` declares Postgres, MLflow, and MinIO but no CPU or memory caps.

Impact:

- Docker Desktop can consume host resources unpredictably, especially during uploads, profile scans, and DB writes.

Safe fix:

- Add `infra/compose.local-safe.yaml` with conservative `cpus` and `mem_limit`.
- Keep full-speed compose as an opt-in profile.

### Medium

1. Dataset upload does full file inventory and JavaScript CRC work.

Evidence:

- Electron collects and sorts all files before upload in `apps/mission-control/electron/zip-stream.cjs:12`.
- ZIP streaming computes CRC in JavaScript while sending to MinIO in `apps/mission-control/electron/zip-stream.cjs:96`.

Impact:

- Large file counts can lag the Electron process and host filesystem.

Safe fix:

- Add upload preflight summary before upload:
  - file count
  - total size
  - warning above 25,000 files or 5 GB
- Add an import mode that can use an existing local ZIP or pre-uploaded S3 URI.

2. Persistent caches have no size or age policy.

Evidence:

- Persistent dataset cache is enabled through env and skipped by cleanup in `services/worker/worker/datasets/cache.py:272`.
- Export and exemplar caches live under `.cache` paths in `services/worker/worker/champion_jobs.py`.
- Docker volumes for MinIO, Postgres, and MLflow are unbounded.

Impact:

- Disk growth over repeated runs.

Safe fix:

- Add a `scripts/cleanup_local_artifacts.ps1` and UI "Clear local caches" action.
- Default cache budget: 5 GB.
- Delete job scratch caches after success/failure unless persistent cache is explicitly enabled.

3. Python demo inference loads the model per request in worker jobs.

Evidence:

- `run_demo_inference_from_manifest` loads TorchScript/framework-native artifacts inside each call in `services/worker/worker/exporting/inference.py:78`.

Impact:

- Demo prediction jobs can feel slow and CPU-heavy for repeated local tests.

Safe fix:

- Cache one model per manifest path inside the worker process with an LRU size of 1-2.
- Keep frontend ONNX local runtime for interactive demos where possible.

4. Modal DataLoader defaults are fine for remote containers but too high for local fallback.

Evidence:

- Modal defaults are 2 ImageFolder workers and 4 metadata-aware workers in `services/worker/worker/training/modal_app.py:74`.
- DataLoader workers can be raised to 16 via env in `services/worker/worker/training/modal_app.py:3191`.

Impact:

- Mostly remote, but risky if reused for local PyTorch execution later.

Safe fix:

- Add local-specific thread and DataLoader caps now:
  - `OMP_NUM_THREADS=4`
  - `MKL_NUM_THREADS=4`
  - `TOKENIZERS_PARALLELISM=false`
  - local DataLoader workers: 0 or 1

### Low

1. Local training simulator is intentionally cheap.

Evidence:

- Local training in `services/worker/worker/training/local.py:10` emits synthetic metrics and sleeps per epoch.

Impact:

- This is not the likely source of local lag.

Safe fix:

- Keep it as v1 fallback.
- Label it clearly as local simulator mode in UI and docs.

2. Some release hygiene is not performance-specific but should be fixed before v1.

Evidence:

- `.env.local` contains live-looking secret values. Do not ship or commit this file.

Safe fix:

- Rotate any exposed local keys.
- Keep `.env.local` ignored.
- Add `.env.v1.local.example` with safe defaults and no secrets.

## Suspected Bottlenecks

### Dataset Upload And Profiling

File/function:

- `apps/mission-control/electron/zip-stream.cjs:12`
- `apps/mission-control/electron/main.cjs:770`
- `services/worker/worker/jobs.py:54`
- `services/worker/worker/datasets/profiler.py:65`

Resource impacted:

- CPU, Disk, RAM, UI responsiveness, MinIO IO.

Why it may lag or crash:

- Dataset upload scans all files, streams a ZIP, hashes the ZIP, uploads to MinIO, then the worker downloads, extracts, recursively scans, and opens image headers. This is multiple full passes over the same data.

How to verify:

- Measure:
  - upload seconds
  - ZIP size
  - file count
  - profile seconds
  - profile file count
  - worker process RSS before/after
  - MinIO volume growth

Safe fix:

- Add profile cache by checksum.
- Add caps and warnings to profile scans.
- Show upload preflight and block very large datasets unless the user confirms.

### Local Worker Count And Auto Execution

File/function:

- `apps/mission-control/electron/main.cjs:437`
- `apps/mission-control/electron/main.cjs:630`
- `services/orchestrator/internal/agents/planner.go:256`
- `services/orchestrator/internal/api/handlers.go:14252`

Resource impacted:

- CPU, process count, DB, local logs.

Why it may lag or crash:

- The planner and UI can request more than one worker. Current local env can auto-execute plans and run Modal dispatcher slots. The app needs a conservative default regardless of how ambitious the plan is.

How to verify:

- During a run, capture:
  - Python process count
  - active worker count
  - queued/running job count
  - dispatcher slot count
  - DB poll rate

Safe fix:

- Local safe mode hard-caps local process count and concurrent jobs to 1.
- UI can still show recommended workers, but launch should default to 1 unless the user opts into parallel mode.

### Mission Control Refresh And Activity Stream

File/function:

- `apps/mission-control/src/App.tsx:1402`
- `apps/mission-control/src/App.tsx:1752`
- `apps/mission-control/src/App.tsx:1760`
- `services/orchestrator/internal/api/activity.go:53`

Resource impacted:

- UI, DB, CPU.

Why it may lag or crash:

- Live refresh fetches many resources every 10 seconds. SSE ticks every 2 seconds and can also trigger refreshes. Slow data is cached for some endpoints, but broad state arrays are still stored in the frontend.

How to verify:

- Browser devtools:
  - requests per minute idle
  - requests per minute active
  - average payload size
  - React commit time on refresh
- Backend:
  - endpoint count per minute
  - average duration per list endpoint

Safe fix:

- Increase idle refresh interval to 30 seconds.
- Coalesce event-triggered refreshes.
- Add payload limits for jobs/summaries/evaluations/metrics.

### Backend List Payload Growth

File/function:

- `services/orchestrator/internal/api/handlers.go:1109`
- `services/orchestrator/internal/api/handlers.go:1160`
- `services/orchestrator/internal/api/handlers.go:1368`
- `services/orchestrator/internal/store/postgres.go:1107`
- `services/orchestrator/internal/store/postgres.go:1348`
- `services/orchestrator/internal/store/postgres.go:1487`
- `services/orchestrator/internal/store/postgres.go:1583`

Resource impacted:

- DB, RAM, UI.

Why it may lag or crash:

- These endpoints return all rows for a project or job. As runs accumulate, payloads become slow even if the UI only displays a subset.

How to verify:

- Seed a project with 500 jobs and 500 evaluations.
- Compare endpoint response times and payload size before/after pagination.

Safe fix:

- Add pagination defaults and preserve unlimited access only behind explicit query params for developer views.

### Job Polling Query Shape

File/function:

- `services/orchestrator/internal/store/postgres.go:993`

Resource impacted:

- DB and worker responsiveness.

Why it may lag or crash:

- A worker can scan queued jobs it cannot execute because provider/template matching happens in Go.

How to verify:

- Queue mixed local, Modal, and persistent GPU jobs.
- Run several workers.
- Measure poll query duration and rows inspected per poll.

Safe fix:

- Filter templates in SQL.
- Add provider filter support to SQL.
- Add short backoff when no matching job exists.

### Artifact, Cache, And Log Growth

File/function:

- `services/worker/worker/datasets/cache.py:272`
- `apps/mission-control/electron/main.cjs:675`
- `infra/compose.yaml`

Resource impacted:

- Disk, IO, app startup time.

Why it may lag or crash:

- Persistent dataset caches, Docker volumes, logs, exports, and exemplars can grow without a default retention policy.

How to verify:

- Measure:
  - `artifacts/`
  - `services/worker/.cache/`
  - Docker volumes for Postgres, MinIO, MLflow
  - log file size after 2 hours idle plus one run

Safe fix:

- Add cache budget and cleanup script.
- Rotate logs.
- Document Docker volume cleanup for v1.

## Immediate Low-Risk Fixes

Order these by impact for the five-day release window.

1. Add a v1 local-safe profile.

Implementation:

- Add `.env.v1.local.example`.
- Add code-level caps so the safe profile is not only documentation.
- Default local workers and local max concurrent jobs to 1.
- Disable auto-execute and auto-followups in v1 examples.
- Make parallel/Modal/autonomous mode explicit.

Acceptance:

- Starting the app with the v1 env never creates more than one local Python worker per project unless the user opts in.
- Auto-created plans do not auto-execute by default.

2. Add pagination and payload limits.

Implementation:

- Add `limit` and `offset` to:
  - `/projects/:id/jobs`
  - `/projects/:id/training-run-summaries`
  - `/projects/:id/training-run-evaluations`
  - `/jobs/:id/metrics`
- Update Mission Control to request bounded lists.

Acceptance:

- A project with 500 historical jobs still loads the main page quickly.
- Developer/raw views can still request larger payloads deliberately.

3. Cap and cache dataset profiling.

Implementation:

- Cache successful profile payload by dataset checksum and profile version.
- Add env caps:
  - `MODEL_EXPRESS_PROFILE_MAX_IMAGES=25000`
  - `MODEL_EXPRESS_PROFILE_MAX_FILES=25000`
  - `MODEL_EXPRESS_PROFILE_MAX_METADATA_FILES=20000`
- Add warnings to the profile when caps are hit.

Acceptance:

- Large datasets produce bounded profile work and a clear warning instead of locking up the machine.

4. Reduce duplicate live refresh pressure.

Implementation:

- Active run: keep 10 second live refresh.
- Idle project: 2 minute live refresh.
- Activity SSE: default to 5 seconds active and 10 seconds idle.
- Event-triggered refresh: throttle to 3 seconds minimum.

Acceptance:

- Idle Mission Control generates minimal network/DB activity.
- Active runs still feel live enough for v1.

5. Add local cache/log cleanup.

Implementation:

- Add `scripts/cleanup_local_artifacts.ps1`.
- Add log rotation for Mission Control and worker JSONL files.
- Add docs for Docker volume cleanup.

Acceptance:

- A user can reclaim local disk without manually hunting paths.

6. Push polling filters into SQL.

Implementation:

- Filter `template` in SQL.
- Filter provider through `config->>'provider'` where possible.
- Keep Go filter as defensive validation.

Acceptance:

- Polling mixed provider queues does not scan unrelated queued jobs every time.

7. Add lightweight performance instrumentation.

Implementation:

- Log duration and size for:
  - dataset upload
  - dataset download
  - ZIP extraction
  - profile scan
  - metadata import
  - job poll no-match count
  - frontend refresh payload size in dev/debug mode

Acceptance:

- A slow run has enough timing data to identify whether lag came from upload, profile, backend, worker, or UI.

## Required Safety Settings

Use these as v1 defaults.

| Setting | Recommended v1 default | Reason |
| --- | ---: | --- |
| Max local workers | 1 | Prevent local process spikes. |
| Max concurrent local training jobs | 1 | Protect CPU, DB, disk, and user responsiveness. |
| Default local classification batch size | 8 | Conservative if real local training is later enabled. |
| Default local detection batch size | 2-4 | Safer for memory-heavy image workloads. |
| Local DataLoader workers | 0 or 1 | Avoid CPU oversubscription. |
| Modal dispatcher slots in v1 local profile | 1 | Keep control-plane work predictable. |
| Modal dispatcher slots in explicit fast mode | 2 | Allow faster remote mode without overwhelming the local app. |
| Worker poll interval | 5 seconds active, 10-15 seconds idle | Reduce DB/log churn. |
| Activity stream interval | 5 seconds active, 10 seconds idle | Avoid duplicate refresh pressure. |
| Main UI live refresh | 10 seconds active, 30 seconds idle | Good enough for v1 while reducing DB work. |
| Max log lines retained in UI | 1,000-2,000 | Avoid large frontend state. |
| Max metrics returned by default | 200 per selected job | More than enough for current epoch ranges. |
| Max jobs returned by default | 100 latest | Keeps main page bounded. |
| Max evaluations returned by default | 50 latest | Evaluations can contain large JSON. |
| Max local artifact/cache size | 5 GB soft cap | Prevent disk growth from repeated runs. |
| Dataset profile image cap | 25,000 image headers | Bound import/profile cost. |
| Dataset metadata inventory cap | 20,000-50,000 files | Existing cap is 50,000; lower default for v1 safe mode. |
| Auto execute plans | false | User action should start spend/work in v1. |
| Auto follow-up rounds | 1 by default | Prevent runaway agent loops. |

## Five-Day Execution Plan

### Day 1 - June 10, 2026: Local Safety Defaults

Tasks:

- Add `.env.v1.local.example` with safe settings and no secrets.
- Enforce local worker and local concurrency caps in code.
- Make Modal/autonomous/parallel run mode explicit in UI.
- Add a clear "safe local" vs "fast remote" runtime mode in settings.

Exit criteria:

- With v1 env, a project import plus plan creation starts at most one local worker.
- No plan auto-executes unless the user enables it.

### Day 2 - June 11, 2026: Dataset Profile Guardrails

Tasks:

- Add profile cache by dataset checksum and profile schema version.
- Add profile scan caps and warnings.
- Add timing logs for upload/download/extract/profile/metadata import.
- Add import preflight warning for large datasets.

Exit criteria:

- A large dataset cannot silently monopolize the machine.
- Profile timing appears in diagnostics.

### Day 3 - June 12, 2026: Bounded Backend And UI Refresh

Tasks:

- Add pagination to jobs, summaries, evaluations, and metrics.
- Update Mission Control fetches to bounded defaults.
- Adjust live refresh and SSE intervals.
- Throttle event-triggered refreshes.

Exit criteria:

- Main project view remains responsive with hundreds of historical rows.
- Idle app has low request rate.

### Day 4 - June 13, 2026: Cleanup, Polling, And Docker Safety

Tasks:

- Add local artifact/log cleanup script.
- Add log rotation.
- Push job poll filtering into SQL.
- Add optional `infra/compose.local-safe.yaml` with resource caps.

Exit criteria:

- A user can reclaim disk safely.
- Mixed provider queues do not create heavy poll scans.
- Docker local-safe mode has documented resource expectations.

### Day 5 - June 14, 2026: Verification And Release Cut

Tasks:

- Run a full local smoke:
  - create project
  - upload small dataset
  - profile
  - create plan
  - execute one local/simulated run
  - observe metrics
  - select/export champion path where available
  - stop/cancel path
- Run a large-dataset profile stress test using caps.
- Run a history-load test with hundreds of jobs/evaluations.
- Record process count, CPU, memory, disk growth, and request rate.

Exit criteria:

- No run-stopping errors in smoke.
- No runaway local worker count.
- No unbounded UI payloads.
- Disk growth is explainable and cleanable.

June 15, 2026 is reserved for packaging, README updates, release notes, and any emergency fixes found on Day 5.

## Implemented In This Pass

Completed fixes:

- Critical #2: dataset profiling is now bounded by configurable image/file/corrupt-path caps, emits profile warnings, logs download/extract/profile/metadata timings, and reuses a checksum plus profile-version cache when available.
- High risks: dashboard jobs, metrics, summaries, evaluations, and activity reads now have bounded backend defaults; Mission Control fetches use bounded limits, lower idle refresh pressure, and throttled SSE-triggered refreshes; Postgres job polling applies provider/template filters in SQL with defensive Go filtering.
- High risks: worker/orchestrator/Electron JSONL logs now rotate, idle worker logging is throttled/configurable, and `infra/compose.local-safe.yaml` documents resource-capped Postgres/MinIO/MLflow defaults.
- Medium risks: dataset folder upload has preflight file/size warnings and caps, local cache/log/export cleanup has a dry-run script, demo inference reuses loaded model runtimes, and Modal/local training defaults reduce DataLoader/thread pressure.

Verification run:

- `go test ./internal/api ./internal/store ./internal/diagnostics` from `services/orchestrator` with workspace-local `GOCACHE`.
- `python -m pytest tests/test_dataset_cache.py tests/test_profiler.py tests/test_diagnostics.py tests/test_demo_inference.py tests/test_training_modal_helpers.py tests/test_worker_main.py` from `services/worker`.
- `node apps/mission-control/electron/zip-stream.test.cjs`.
- `npm run build` from `apps/mission-control`.

Remaining risk:

- Critical #1 local-safe profile was intentionally not implemented in this pass.
- Full local smoke, large-dataset stress, and history-load tests are still required before v1 release confidence.
- Docker resource caps are available through the local-safe override but are not enforced by the base compose file.

## Verification Checklist

Local run reliability:

- Project creation succeeds.
- Dataset upload to MinIO succeeds.
- Profile job completes or fails with actionable error.
- Planner creates a valid plan.
- Manual plan execution queues jobs.
- Worker claims and completes the job.
- Metrics and summary appear in UI.
- Cancel stops queued/running work and stops workers.
- Export/demo paths do not block the core run if optional artifacts are unavailable.

Local performance:

- Idle app stays below 1 local worker process.
- Idle UI request rate is low.
- Active run does not exceed configured worker cap.
- Dataset profile logs elapsed seconds and file/image counts.
- Main project endpoint payloads stay bounded.
- `artifacts/`, `.cache/`, and Docker volumes have documented cleanup.

Regression tests:

- Go API/store tests around pagination and poll filtering.
- Worker tests around profile cap warnings and cache hits.
- Frontend build.
- Manual smoke with the v1 env.

## Final Recommendation

High-end machine:

- Safe with limits.
- The current repo can run on a capable machine if worker count, auto-execution, dataset size, and UI payload growth are controlled. Without those limits, the current local env can still create visible lag.

Mid-range machine:

- Not safe yet.
- It needs v1 safe defaults, profile caps/cache, bounded backend lists, and reduced refresh pressure before it is reasonable to call local operation dependable.

Low-end machine:

- Not safe yet.
- Local Docker plus Electron plus orchestrator plus workers will be too sensitive without explicit low-power mode, one-worker execution, capped profiling, and cleanup.

Overall:

- The weakest parts are not the local training simulator. They are local orchestration defaults, dataset import/profile work, unbounded history payloads, duplicate live refresh paths, and missing cache/log/Docker limits.
- The fastest path to a credible v1 is to ship conservative defaults and explicit opt-in performance modes, then verify the full project-to-run-to-champion path under those limits.
