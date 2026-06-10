# Remote GPU Cost Reduction Plan

## Purpose

Reduce remote GPU training cost for Model Express, especially Modal-backed YOLO and image-classification runs. The main target is wasted billed GPU/runtime from repeated dataset download, extraction, materialization, oversized trial schedules, weak preview/full-train policy, late stopping, and warm/idle container behavior.

This is a planning document only. It intentionally does not prescribe implementation code changes in this PR.

## Repo State Inspected

This plan is based on the current local repository state, including:

- `docs/plans/pure_performance_download_reduction_plan.md`
- `docs/plans/smarter_agentic_orchestration_plan.md`
- `docs/plans/yolo_plan.md`
- `services/worker/worker/training/modal_app.py`
- `services/worker/worker/training/modal_provider.py`
- `services/worker/worker/training/modal_dispatcher.py`
- `services/worker/worker/datasets/cache.py`
- `services/worker/worker/jobs.py`
- `services/worker/worker/main.py`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/runs/model.go`
- `services/orchestrator/internal/workers/model.go`
- `services/orchestrator/internal/settings/model.go`
- `services/orchestrator/internal/execution/model.go`
- `apps/mission-control/electron/main.cjs`
- `apps/mission-control/src/App.tsx`
- Modal, dataset cache, dispatcher, worker-requirement, training-helper, early-stopping, planner, and replay tests under `services/worker/tests` and `services/orchestrator/internal`.

The worktree already had modified implementation files during inspection. This plan treats those as existing local changes and does not rely on reverting them.

## 1. Current-State Analysis

### How Modal Training Gets Datasets Today

The current path is:

1. Mission Control uploads a zipped dataset artifact through Electron.
2. The orchestrator creates training jobs with `dataset_id`, provider, GPU type, and a `dataset_materialization` config block when the provider is Modal.
3. `run_modal_training` in `modal_provider.py` fetches the dataset record from the orchestrator and sends the dataset storage URI, checksum fields, S3 endpoint, and job config to a Modal function.
4. `train_image_classifier` and `train_yolo_detector` in `modal_app.py` call `ensure_dataset_materialized(...)`.
5. For training functions, `ensure_dataset_materialized(...)` uses `_modal_training_dataset_cache_root()`, which defaults to `/tmp/model-express/training-datasets`.

That means training currently materializes into a local ephemeral path inside the Modal training container by default. The helper is checksum-keyed when a valid checksum is available, and it falls back to a storage URI fingerprint when no checksum is available.

### Where Repeated Download/Extract/Materialization Happens

The highest-risk repeated work is still per remote training invocation:

- Each Modal training job can download and extract the dataset into `/tmp/model-express/training-datasets`.
- A cache hit is possible only when the same Modal container and the same local cache root are reused.
- Separate cold containers will miss the local cache.
- There is no current Modal batch function that runs several experiments after one ephemeral extraction.
- Modal profiling uses a `TemporaryDirectory`, so profiling intentionally downloads/extracts ephemerally and does not reuse the durable Modal dataset volume.
- Local profile and visual analysis jobs use `job_dataset_cache_root(job_id)` and clean it up unless `MODEL_EXPRESS_PERSIST_DATASET_CACHE` is enabled.
- Local job-scoped cache keys are dataset-ID-based, not checksum-based, while the newer materialization helper is checksum/storage-fingerprint based.

The previous `pure_performance_download_reduction_plan.md` correctly identified the major problem: the dataset archive can be served once per materializing job rather than once per dataset checksum. The current repo has implemented pieces of that idea, but not the full end-to-end training behavior.

### Current Modal Volumes, Local Temp Cache, Torch Cache, Dispatcher Prewarm, And Checksum Cache

Observed current state:

- Dataset checksum cache: present in `worker/datasets/cache.py`. It uses a checksum or storage URI fingerprint key, a `.complete` marker, a manifest, and a directory lock. Tests cover reuse and concurrent single-download behavior.
- Local temp cache: present. Training defaults to `/tmp/model-express/training-datasets`; local worker jobs default to job-scoped scratch unless persistence is explicitly enabled.
- Modal dataset Volume: present as `model-express-dataset-cache`, mounted at `/cache/model-express/datasets` only on the `materialize_image_dataset` function.
- Modal torch cache Volume: present as `model-express-torch-cache`, mounted in training functions. Classifier training explicitly reloads it and optionally commits it when `MODEL_EXPRESS_MODAL_SYNC_TORCH_CACHE_COMMIT` is true. YOLO training does not show the same explicit reload/commit calls, although the volume is mounted.
- Dispatcher prewarm: present in `modal_dispatcher.py`, but disabled by default. It becomes active only when job config says `dataset_materialization.enabled` or `prewarm_enabled`, or when `MODEL_EXPRESS_MODAL_DATASET_PREWARM_ENABLED` is true.
- Orchestrator dataset materialization policy: present. Modal jobs get `dataset_materialization` fields such as `dataset_cache_key`, `cold_cache_policy`, `max_concurrent_jobs`, and `max_cold_materializations`.

Important gap: as inspected, the prewarm function writes to the durable dataset Volume at `/cache/model-express/datasets`, while the training functions default to `/tmp/model-express/training-datasets` and their `volumes=` only includes `training_volume_mounts`, which currently comes from the torch cache volume mounts. Unless runtime configuration changes both the training cache root and the Modal training volume mounts, dispatcher prewarm can warm a cache that training does not actually read. This should be verified in a live Modal run by comparing:

- prewarm telemetry cache key and path
- training telemetry cache key and path
- training cache hit/miss after prewarm
- Modal function volume mount list for training functions

### Existing Container Warmth And Possible Idle Cost

The Modal training functions have:

- `timeout=60 * 60`
- optional `min_containers` from `MODEL_EXPRESS_MODAL_TRAIN_MIN_CONTAINERS`
- optional `buffer_containers` from `MODEL_EXPRESS_MODAL_TRAIN_BUFFER_CONTAINERS`
- `scaledown_window` defaulting to 10 minutes through `MODEL_EXPRESS_MODAL_TRAIN_SCALEDOWN_WINDOW_SECONDS`

With `min_containers` and `buffer_containers` unset or zero, the default scaledown window can still keep a warm training container around after a job. That can be helpful for local `/tmp` cache reuse, but it can also create idle cost risk if Modal bills or reserves resources during the warm window. If `min_containers` or `buffer_containers` are nonzero, idle cost risk becomes explicit.

Mission Control starts one Modal dispatcher process when GPU type is `modal`. The dispatcher registers logical slots and polls indefinitely while the Electron app is running. It also keeps a Modal app session open around `run_forever()`. This appears intended to avoid repeated app hydration while submitting remote calls. It should be audited separately from GPU-container cost: an open app/session is not necessarily an active GPU container, but the current UI/backend telemetry does not distinguish these states.

Worker processes are killed on Electron `before-quit`, but there is no obvious cost-sensitive idle shutdown for a Modal dispatcher after all jobs complete.

### Current Cost And Runtime Telemetry

Current telemetry exists but is not stage-complete:

- `TrainingRunSummary` stores `runtime_seconds`, `estimated_cost_usd`, `gpu_type`, `provider`, `modal_function_call_id`, and `modal_input_id`.
- Modal classifier training updates run summaries every epoch and at success.
- Modal YOLO training posts one final summary after Ultralytics training and validation.
- `_modal_identifiers()` attempts to record `modal.current_function_call_id()` and `modal.current_input_id()`.
- Modal estimated cost uses hard-coded GPU prices in `_modal_gpu_price_per_second`.
- Dataset materialization telemetry is carried in the payloads sent by workers:
  - `dataset_materialization_cache_hit`
  - `dataset_materialization_cache_miss`
  - `dataset_materialization_bytes_downloaded`
  - `dataset_materialization_extract_seconds`
  - `dataset_materialization_wait_seconds`
  - `dataset_materialization_status`
  - `dataset_checksum`
  - `storage_uri_fingerprint`
  - `dataset_materialization_cache_key`
- Materialization telemetry is included inside classifier and YOLO summary/evaluation payloads, but `TrainingRunSummary` does not have first-class stage timing fields.
- `_modal_training_phase(...)` logs useful phase markers to stdout, but those are not yet persisted as canonical per-job stage events.

Uncertain assumption: `estimated_cost_usd` is useful for relative reporting, but it should not be treated as authoritative billing unless the hard-coded prices are reconciled against current Modal billing exports.

## 2. Dataset Access And Materialization Strategy

### Do Not Treat Durable Modal Volumes As The Default Answer

Durable Modal Volumes are useful for content-addressed archive/materialization state and for avoiding repeated remote downloads. They are not automatically the best runtime substrate for many-image training.

For YOLO and ImageFolder-style datasets, training touches many small image and label files repeatedly. A durable network-backed or mounted storage layer can be slower than local ephemeral disk because:

- directory traversal and metadata operations dominate startup for many small files
- DataLoader workers issue many small opens rather than a few large sequential reads
- YOLO has paired image and label lookups, plus `data.yaml` path resolution
- repeated random file access can be latency-sensitive
- volume commit/reload semantics add coordination overhead

The best pattern for training is usually:

1. Transfer as few large objects as possible.
2. Extract/materialize once onto local ephemeral disk.
3. Run multiple related trials against that same local materialization.
4. Report each trial independently.

Durable storage should be used as an optional staging/cache layer, not assumed to be the final training filesystem.

### Strategy A: Batch Multiple Experiments Inside One Modal Function

Add a Modal batch runner for jobs that share:

- project ID
- dataset ID
- dataset checksum/cache key
- task type
- compatible training environment
- compatible preview/full tier

The batch function should:

1. Receive a list of trial configs and job IDs from the dispatcher/orchestrator.
2. Download and extract the dataset once into local ephemeral storage.
3. Build any preview subset or split manifest once.
4. Run several trial configs sequentially inside the same Modal function.
5. Report metrics, summaries, evaluations, failures, and completion per job ID.
6. Stop the batch when the mode budget is exhausted.

The default batch size should be small:

- Prototype/cheap: 2 to 3 preview trials
- Balanced: 3 to 5 preview trials
- Quality: 4 to 8 preview trials when the user has enough budget

This is the highest-value first behavior change because it directly turns:

```text
N trials -> N downloads/extractions
```

into:

```text
N trials in one batch -> 1 download/extraction
```

It also avoids relying on warm container reuse or durable volume training performance.

### Strategy B: Checksum-Keyed Local Ephemeral Materialization Per Batch

The existing `ensure_dataset_materialized(...)` helper should be reused, but the cache root should be a batch-local ephemeral path:

```text
/tmp/model-express/batches/<batch_id>/<dataset_cache_key>
```

This gives these properties:

- one download/extract per batch
- no cross-batch storage correctness dependency
- no accidental use of stale extracted files after dataset replacement
- no expensive durable volume reads during the training loop
- simple cleanup at the end of the Modal function

If the same warm container receives another batch for the same checksum, it can optionally reuse `/tmp/model-express/training-datasets`, but that should be treated as opportunistic. Cost reduction should come from the batch contract, not from hoping the same container stays warm.

### Strategy C: Convert Uploaded Zips Into Larger Shard Artifacts

The current upload artifact is a zip. Zip extraction can be expensive and creates many tiny files. A later phase should add a dataset artifact conversion job:

- input: uploaded zip and dataset checksum
- output: task-aware larger shard artifacts under the same checksum
- metadata: class counts, split manifests, file counts, bytes, compression type, schema version

Possible artifacts:

- Classification: WebDataset-style tar shards, class-balanced preview shards, and full shards.
- YOLO: split-level image shards and label shards, or larger compressed tar groups that are extracted once to local disk before Ultralytics training.

For YOLO, do not force Ultralytics to read directly from shards in the first slice. Keep Ultralytics normal by extracting shards into a standard local YOLO directory with `images/`, `labels/`, and `data.yaml`.

### Strategy D: Preview And Full Dataset Tiers

Add deterministic dataset tiers:

- `preview`: smaller, stratified, stable subset for cheap signal.
- `full`: normal train/val/test dataset.
- `champion_validation`: final heldout/test pass on the candidate champion.

The preview subset must be content-addressed by:

```text
dataset_checksum + task_type + tier + fraction + seed + split_policy + image_size_family
```

For classification, preview selection should be class-stratified and should preserve official split rows when metadata is available.

For YOLO, preview selection should preserve:

- train/val/test split semantics
- image to label pairing
- class/object distribution as much as possible
- labels with boxes for selected images
- `data.yaml` pointing at the preview-local train/val/test paths

### Strategy E: Provider-Specific Persistent Disk Workers As Optional Non-Modal Backend

For large datasets with many small files, a persistent disk GPU worker can be more cost-effective than repeated serverless Modal startup. This should be an optional backend, not a replacement for Modal.

Possible backend shape:

- a cloud VM or managed GPU worker with persistent NVMe/EBS disk
- checksum-keyed dataset materialization rooted on persistent disk
- worker leases from the existing Go orchestrator
- same job reporting endpoints as Modal/local
- cost mode caps enforced by the orchestrator

This is most useful when:

- the same dataset is trained for many follow-up rounds
- the dataset is too large for repeated serverless materialization
- many tiny files dominate startup
- quality mode needs larger YOLO models or high image sizes

### YOLO Versus Image Classification Tradeoffs

YOLO:

- Needs image/label pairing, `data.yaml`, and official split preservation.
- Is more sensitive to incorrect path rewriting.
- Benefits strongly from local ephemeral extraction because Ultralytics expects a normal filesystem layout.
- Preview subsets must preserve detection labels, not just sample images.
- Direct cloud-mounted training over many small image and label files is likely poor.
- First batching slice should target YOLO because repeated YOLO materialization is expensive and the trial count can grow quickly.

Image classification:

- Current worker supports `ImageFolder` and metadata-bundle-aware loading.
- Preview subsets are easier to create with stratified image lists.
- Shard loaders can be introduced later, but first should keep the existing ImageFolder path and run several configs after one extraction.
- Classification already has deterministic early stopping; the bigger win is batching and preview/full scheduling.

### Recommended First Implementation Slice

Do not start by making the durable Modal dataset Volume the primary training filesystem.

First behavior slice:

1. Add a Modal preview batch runner behind a rollout flag, initially for YOLO and then classification.
2. Batch same-dataset, same-tier Modal trials into one function.
3. Download/extract once to local ephemeral disk.
4. Generate one preview `data.yaml` or classification split manifest.
5. Run 2 to 3 trials and report each result separately.
6. Persist per-stage telemetry for materialization, training, evaluation, export, and idle/wait time.

This directly reduces repeated materialization without betting on durable volume read performance.

## 3. YOLO-Specific Cost Reduction

### YOLO Dataset Structure To Preserve

YOLO detector training must preserve:

- `data.yaml`, `data.yml`, `dataset.yaml`, or `dataset.yml`
- `images/` and `labels/` directories
- normalized YOLO label rows
- train/val/test split paths
- class `names` and `nc`

The current `train_yolo_detector` path searches for a YOLO config file and requires it to look like a YOLO data config. This is the right validation boundary. Cost reductions should keep that contract intact.

### Why Direct Cloud-Mounted YOLO Training Is Risky

Direct S3/cloud-bucket-mounted YOLO training over many tiny files is likely bad because:

- every image and label pair can cause separate object metadata and read operations
- Ultralytics will traverse paths and load labels/images through filesystem APIs
- object stores are optimized for large object throughput, not small random file access
- latency variance can starve GPU batches
- split-relative paths in `data.yaml` can become fragile under mounts

Use cloud/object storage for transfer and durable artifacts. Use local ephemeral disk for the actual Ultralytics training loop.

### YOLO-Friendly Batch Approach

The batch function should:

1. Materialize the dataset once locally.
2. Resolve or generate a local batch `data.yaml` once.
3. For preview tier, create deterministic subset folders or symlink/hardlink-safe local paths for selected images/labels.
4. Run each YOLO trial config with normal `YOLO(model).train(data=..., ...)`.
5. Save trial outputs under separate run roots.
6. Validate each trial separately.
7. Post per-job summaries and evaluations separately.

Trial configs can differ by:

- model size, such as `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`
- image size
- epochs/batch/learning rate
- augmentation settings once supported
- preprocessing variants that still preserve labels and coordinates

### Fair Preprocessing Comparisons

For preprocessing variants, fairness rules must be explicit:

- same preview subset
- same seed
- same train/val/test split
- same model weights
- same epoch/step budget
- same image size unless image size is the tested variable
- one variable family changed at a time

For YOLO, examples:

- compare letterbox strategy variants only if boxes are transformed consistently
- compare image size 512 vs 640 with the same selected images and labels
- compare augmentation intensity only with the same base subset
- never compare a preprocessing variant on a different subset and treat it as a model improvement

### Preview Runs Are Not Hard Knockout Rounds

Preview runs should reject candidates that are broken or clearly dominated. They should not eliminate every ambiguous hypothesis.

Reject in preview when:

- training fails or labels/config are invalid
- loss explodes or metrics are nonsensical
- recall stays near zero on a representative validation subset
- a paired control dominates the candidate by a large margin under the same subset
- materialization overhead already consumed the mode budget

Promote or rescue when:

- metric differences are small and variance is likely
- a preprocessing hypothesis improves recall but lowers precision
- a larger model is slower but may be useful for quality mode
- YOLO11n underperforms but YOLO11s/YOLO11m could still validate a capacity hypothesis
- preview subset is too small or class coverage is weak

This avoids the specific failure mode where one weak early YOLO11n result blocks useful preprocessing or larger-model hypotheses.

## 4. Preview Runs, Promotion, And Early Stop

### Multi-Fidelity Policy

Use this training ladder:

```text
preview -> promote -> full train -> champion validation
```

Preview:

- small deterministic subset
- short epoch budget
- cheap image size
- paired controls for preprocessing hypotheses
- aggressive sanity stops

Promote:

- select the smallest set of hypotheses with meaningful signal
- keep one baseline/control alive when the result is ambiguous
- allow rescue for hypotheses that address known dataset evidence

Full train:

- full dataset
- normal image size
- fewer candidates
- deterministic early stopping
- richer evaluation and export only when justified

Champion validation:

- final heldout/test pass
- export readiness if the user wants deployment
- no new search unless explicit reopen or budget remains

### Promotion Rules

Promote when any of these are true:

- candidate beats paired control by a useful delta on target metric
- candidate improves a critical secondary metric such as recall or per-class F1
- candidate has lower cost/runtime with similar quality
- candidate is a required control for judging another promoted candidate
- candidate is ambiguous but aligned with strong dataset evidence

Rescue when:

- preview subset has poor class/object coverage
- YOLO recall improves but mAP is noisy
- larger model has expected benefit and cost mode allows one capacity challenge
- preprocessing mechanism is evidence-backed but first result is inconclusive

Reject when:

- candidate is broken
- candidate duplicates an already tested signature
- candidate is dominated by its paired control under the same subset
- candidate violates task compatibility, such as classifier on YOLO evidence
- candidate cannot fit budget even as a preview

Escalate model size when:

- small model underfits and recall/mAP are low despite sane losses
- loss curves still improve at the end of preview
- full-data evidence suggests capacity, not preprocessing, is the bottleneck
- project mode is `balanced` or `quality`

Do not escalate model size when:

- data labels are broken
- materialization dominates runtime
- preview subset has too little class/object coverage
- the project is in prototype/cheap mode and budget is nearly exhausted

### Existing Stopper Behavior To Reuse

The classifier path has deterministic early stopping:

- disabled when patience is zero
- waits for patience after the best epoch
- stops non-egregious runs only after roughly half of the epoch budget
- allows earlier stop for egregiously low target metric after a warmup

That behavior should remain the base classifier stopper.

### Additional Stop Signals To Add

Add stage-aware and task-aware stop signals:

- no learning signal after warmup
- exploding or NaN loss
- validation loss diverges from training loss beyond a bounded threshold
- YOLO recall near zero after enough labeled objects were seen
- YOLO box/cls/DFL losses fail to decrease
- candidate dominated by paired control
- materialization overhead exceeds a cost-mode threshold
- projected cost exceeds budget cap
- full-data run is unlikely to beat current champion by useful delta

For YOLO, prefer monitoring Ultralytics epoch CSV as it is produced rather than waiting until final validation. If that is too invasive for the first slice, post-process epoch CSV immediately after training and add a second slice for live early abort.

### Avoid Weak Early Baseline Lockout

Planner and promotion policy must preserve hypothesis diversity:

- A weak YOLO11n baseline can reject "YOLO11n as champion" but should not reject "preprocessing fixes labels/scale" or "YOLO11s has enough capacity."
- A preview failure caused by invalid `data.yaml` should block the dataset configuration, not the model family.
- A low-score baseline should trigger diagnosis before more full training, but it should not automatically close all follow-up routes.

## 5. Container Lifecycle And Idle Cost Audit

### Current Warm-Start Surface

Relevant current knobs:

- `MODEL_EXPRESS_MODAL_TRAIN_MIN_CONTAINERS`
- `MODEL_EXPRESS_MODAL_TRAIN_BUFFER_CONTAINERS`
- `MODEL_EXPRESS_MODAL_TRAIN_SCALEDOWN_WINDOW_SECONDS`, default 600 seconds
- `MODEL_EXPRESS_MODAL_DISPATCHER_SLOTS`, default 2, max 8
- `MODEL_EXPRESS_MODAL_DATASET_PREWARM_ENABLED`, default false unless job config enables it
- Mission Control starts Modal workers/dispatchers when executing plans, scheduling follow-ups, or when worker requirements are active
- The dispatcher keeps polling indefinitely while its process is alive

### Places Containers Or Processes May Remain Open Without Active Work

Audit these paths:

- Modal training function warm pool after a completed job because of scaledown window.
- Nonzero Modal `min_containers` or `buffer_containers`.
- Dispatcher process with no queued Modal jobs.
- Dispatcher app session open while no active remote calls are running.
- Mission Control ensuring workers repeatedly for active requirements.
- Automatic profile worker started with Modal as default training provider when project creation only needs a profile.
- Prewarm job that creates durable materialization but no following training job consumes it.
- Failed/retryable Modal calls where the dispatcher remains alive and heartbeating but no useful work runs.

### Telemetry To Add

Persist per job and per project:

- `queue_wait_seconds`: job created to assigned/running.
- `dispatcher_wait_seconds`: claimed by dispatcher to remote call submitted.
- `container_start_seconds`: remote function invoked to first phase log, if measurable.
- `dataset_materialization_seconds`: download + extract + lock wait.
- `dataset_download_seconds` and `dataset_extract_seconds` separately.
- `active_training_seconds`: first epoch/train start to train done.
- `evaluation_seconds`: validation/test/export readiness.
- `export_seconds`: artifact production/upload.
- `idle_wait_seconds`: remote function time not assigned to materialization/training/evaluation/export.
- `warm_container_policy`: min containers, buffer containers, scaledown window.
- `dispatcher_idle_seconds`: dispatcher process alive with zero active jobs and no queued jobs.
- `orphaned_container_suspected`: remote function/container still active after job terminal state.

Events should distinguish:

- active training time
- dataset materialization time
- idle/wait time
- warm container time
- queue wait time
- orphaned/unused container time

### Safe Defaults For Cost-Sensitive Mode

For prototype/cheap mode:

- Modal `min_containers=0`
- Modal `buffer_containers=0`
- scaledown window 60 to 120 seconds
- dispatcher slots 1
- prewarm disabled unless a batch will consume it immediately
- no project-start Modal warmup
- auto-stop dispatcher after no active/queued Modal jobs for 2 to 5 minutes
- no export for preview runs
- hard budget cap stops new scheduling

For balanced mode:

- Modal `min_containers=0`
- Modal `buffer_containers=0`
- scaledown window 120 to 300 seconds
- dispatcher slots 2
- batch prewarm only for same-dataset batches
- stop dispatcher after 5 to 10 idle minutes

For quality mode:

- allow longer scaledown, but require visible estimated idle cost
- still default `min_containers=0` unless user opts into fast warm starts
- cap dispatcher slots and concurrent GPU functions by budget

## 6. Cost Modes

Add user-facing cost modes that drive scheduling, preview/full policy, early stop, export, warm container settings, and budget behavior.

| Setting | Prototype/Cheap | Balanced | Quality |
| --- | --- | --- | --- |
| Max concurrent Modal jobs | 1 active GPU batch or 1 single job | 2 active GPU jobs or batches | 3 to 4 if budget allows |
| Max preview trials | 2 to 3 | 4 to 6 | 6 to 8 |
| Max full trials | 1 | 2 | 3 to 4 |
| Max follow-up rounds | 1 | 2 | 3, or user configured cap |
| Preview dataset fraction | 10% to 25%, min class/object coverage | 25% to 50% | 50% or adaptive |
| Full dataset fraction | 100% for promoted candidate only | 100% | 100% |
| Classification image size | 160 to 224 preview, 224 full | 224 to 256 | 224 to 384 when justified |
| YOLO image size | 416 to 512 preview, 640 full only for promoted | 512 preview, 640 full | 640 baseline, 768/960/1280 only for small-object evidence |
| Early stop | aggressive sanity and plateau stops | moderate | conservative, but still stop broken/dominated runs |
| Export policy | champion only, no preview exports | champion plus one final candidate if useful | top candidates or champion validation exports |
| Warm container policy | min/buffer 0, scaledown 60 to 120 seconds | min/buffer 0, scaledown 120 to 300 seconds | min/buffer 0 by default, longer scaledown only with visible idle cost |
| Budget cap behavior | hard stop new jobs, cancel queued previews | finish active job, block new full trains | soft warning, require explicit continue past cap |

Budget cap behavior must be deterministic and backend-enforced. The LLM planner can explain tradeoffs, but it should not be the only budget guard.

## 7. Implementation Phases

### Phase 1: Stage Cost Telemetry And Current-Behavior Audit

Files likely touched:

- `services/worker/worker/training/modal_app.py`
- `services/worker/worker/training/modal_provider.py`
- `services/orchestrator/internal/runs/model.go`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/store/*`
- `apps/mission-control/src/App.tsx`

Behavior change:

- Persist materialization, training, evaluation, export, queue wait, and idle/wait stage timings.
- Surface per-project and per-run cost by stage.
- Preserve existing summary fields.

Migration/config needs:

- Add optional JSON stage telemetry or a new run-stage table.
- No behavior flags required for read-only telemetry, but UI display can be behind a feature flag.

Tests:

- Worker posts stage telemetry for classifier and YOLO.
- Orchestrator stores and returns stage telemetry.
- UI renders missing stage fields gracefully.

Rollout flags:

- `MODEL_EXPRESS_REMOTE_GPU_STAGE_TELEMETRY=1`

Risks:

- Too much telemetry can bloat summaries.
- Modal phase logs may not map perfectly to billed time.

Acceptance criteria:

- A completed Modal job shows dataset materialization seconds, active training seconds, evaluation/export seconds, queue wait, GPU type, estimated cost, and Modal IDs.
- The system can calculate materialization share of total runtime per run.

### Phase 2: Fix Modal Cache Truthfulness And Safe Lifecycle Defaults

Files likely touched:

- `services/worker/worker/training/modal_app.py`
- `services/worker/worker/training/modal_dispatcher.py`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/execution/model.go`
- `apps/mission-control/electron/main.cjs`
- `apps/mission-control/src/App.tsx`

Behavior change:

- Make prewarm status truthful: a prewarm should only mark a dataset warm for a training path that will read the same cache.
- Add telemetry when prewarm and training roots differ.
- Add cost-sensitive defaults for min/buffer containers and scaledown window.
- Stop or pause idle dispatcher processes after a bounded idle window.

Migration/config needs:

- Optional worker requirement fields for warm status updates, or a separate materialization status table.
- Config defaults for cost mode lifecycle policies.

Tests:

- Prewarm does not claim success for training cache reuse unless cache roots and mounts match.
- Dispatcher exits or idles safely after an idle threshold.
- Cost-sensitive mode emits min/buffer/scaledown defaults.

Rollout flags:

- `MODEL_EXPRESS_MODAL_COST_SENSITIVE_DEFAULTS=1`
- `MODEL_EXPRESS_MODAL_DISPATCHER_IDLE_EXIT_SECONDS`

Risks:

- Too aggressive dispatcher exit could delay follow-up jobs.
- Changing scaledown defaults may reduce warm-cache hit rates.

Acceptance criteria:

- Warm idle container/process time is visible and bounded.
- Prewarm telemetry predicts actual training cache hits.

### Phase 3: Modal Preview Batch Runner

Files likely touched:

- `services/worker/worker/training/modal_app.py`
- `services/worker/worker/training/modal_provider.py`
- `services/worker/worker/training/modal_dispatcher.py`
- `services/worker/worker/orchestrator_client.py`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/jobs/model.go`

Behavior change:

- Group same-dataset preview trials into one Modal function.
- Materialize once to local ephemeral disk.
- Run several trial configs.
- Report each job independently.

Migration/config needs:

- Add batch ID and batch membership fields to job config or a new batch table.
- Add `training_tier=preview|full|champion_validation`.

Tests:

- Two same-checksum jobs in a batch call download once.
- Each trial reports separate metrics and terminal status.
- A failed trial does not hide other trial results.
- Batch respects budget cap.

Rollout flags:

- `MODEL_EXPRESS_MODAL_BATCH_RUNNER=1`
- `MODEL_EXPRESS_MODAL_BATCH_MAX_TRIALS`

Risks:

- One Modal function now owns multiple jobs; failure isolation must be careful.
- Long batches can hit function timeout.

Acceptance criteria:

- For N preview jobs in one batch, object storage serves the dataset archive once.
- Each job still appears as an independent run in the orchestrator/UI.

### Phase 4: Deterministic Preview/Full Dataset Tiers

Files likely touched:

- `services/worker/worker/datasets/cache.py`
- `services/worker/worker/datasets/profiler.py`
- `services/worker/worker/training/modal_app.py`
- `services/orchestrator/internal/datasets/*`
- `services/orchestrator/internal/plans/model.go`
- `services/orchestrator/internal/api/handlers.go`

Behavior change:

- Create deterministic preview subsets for classification and YOLO.
- Preserve official splits where available.
- Store subset manifests keyed by checksum, tier, fraction, seed, and policy.

Migration/config needs:

- Add subset manifest metadata storage, or store manifests as artifacts with references in run evaluation.

Tests:

- Same checksum/tier/seed produces same subset.
- Classification preview is class-stratified.
- YOLO preview preserves image/label pairs and valid `data.yaml`.
- Full tier remains unchanged.

Rollout flags:

- `MODEL_EXPRESS_PREVIEW_DATASET_TIERS=1`

Risks:

- Preview subsets can be unrepresentative on small or imbalanced datasets.
- YOLO subset generation can break relative paths if not carefully tested.

Acceptance criteria:

- Preview runs are reproducible.
- YOLO preview `data.yaml` validates before training.
- Promotion decisions record the subset manifest used.

### Phase 5: YOLO Batch Training And Fair Hypothesis Controls

Files likely touched:

- `services/worker/worker/training/modal_app.py`
- `services/orchestrator/internal/agents/*`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/agents/evals/*`

Behavior change:

- Batch YOLO trial configs after one local extraction.
- Enforce fair paired comparisons for preprocessing and image-size variants.
- Keep Ultralytics training normal.

Migration/config needs:

- Add trial grouping metadata:
  - `control_job_id`
  - `paired_hypothesis_id`
  - `subset_manifest_id`
  - `changed_variables`

Tests:

- YOLO paired controls share subset and seed.
- Detection jobs cannot use classifier-only preprocessing.
- Candidate ranking keeps useful YOLO challengers alive after a weak baseline.

Rollout flags:

- `MODEL_EXPRESS_YOLO_BATCH_PREVIEW=1`

Risks:

- YOLO train calls are expensive; timeout and per-trial cleanup must be bounded.
- Fairness constraints can reject useful free-form planner proposals unless validation messages are clear.

Acceptance criteria:

- YOLO preprocessing variants can be compared fairly.
- One weak YOLO11n run does not block a supported YOLO11s or preprocessing rescue candidate.

### Phase 6: Promotion Policy And Early Stop Signals

Files likely touched:

- `services/worker/worker/training/modal_app.py`
- `services/orchestrator/internal/agents/training_monitor_llm.go`
- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/candidate_ranking.go`
- `services/orchestrator/internal/api/handlers.go`

Behavior change:

- Add deterministic promote/rescue/reject decisions.
- Add materialization-aware and budget-aware stops.
- Extend YOLO monitoring beyond final metrics.

Migration/config needs:

- Store promotion decisions and stop reasons per candidate.

Tests:

- Broken candidates reject.
- Dominated paired candidates reject.
- Ambiguous evidence-backed candidates rescue.
- Budget exceeded blocks new scheduling.
- Classifier existing early stopper behavior remains intact.

Rollout flags:

- `MODEL_EXPRESS_MULTI_FIDELITY_POLICY=1`

Risks:

- Over-aggressive preview stopping can reduce model quality.
- Too much rescue can increase cost.

Acceptance criteria:

- Preview runs reduce full-train count without eliminating promising preprocessing hypotheses.
- Stop reasons are visible in the UI and backend payloads.

### Phase 7: Sharded Dataset Artifacts

Files likely touched:

- `apps/mission-control/electron/main.cjs`
- `apps/mission-control/electron/zip-stream.cjs`
- `services/worker/worker/datasets/cache.py`
- `services/worker/worker/datasets/storage.py`
- `services/orchestrator/internal/datasets/*`

Behavior change:

- Convert uploaded zips into checksum-keyed shard artifacts.
- Use shards for faster transfer and local extraction.
- Keep current zip path as fallback.

Migration/config needs:

- Artifact manifest table or dataset profile artifact references.
- Artifact schema version.

Tests:

- Shard manifest reproduces expected file counts.
- YOLO labels/images remain paired after shard extraction.
- Classification class counts remain stable.
- Fallback zip path still works.

Rollout flags:

- `MODEL_EXPRESS_DATASET_SHARDS=1`

Risks:

- More artifact formats increase maintenance.
- Bad sharding can make small datasets slower.

Acceptance criteria:

- For large datasets, transfer/extract time falls measurably versus zip baseline.
- Training still sees normal local filesystem paths.

### Phase 8: Cost Modes And Budget Caps

Files likely touched:

- `services/orchestrator/internal/settings/model.go`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/store/*`
- `apps/mission-control/src/App.tsx`

Behavior change:

- Add `prototype/cheap`, `balanced`, and `quality` modes.
- Enforce mode-specific limits in the backend.
- Surface budget cap behavior in Mission Control.

Migration/config needs:

- Add mode and budget fields to automation settings.
- Optional project-level budget cap.

Tests:

- Each mode produces expected concurrency/trial/export policy.
- Budget cap blocks queued full trains.
- UI settings round-trip.

Rollout flags:

- `MODEL_EXPRESS_COST_MODES=1`

Risks:

- Users may expect quality mode to be unconstrained.
- Existing automation settings need careful defaults.

Acceptance criteria:

- Cost mode changes scheduling behavior without requiring prompt changes.
- Costs are visible per project and per training stage.

### Phase 9: Optional Persistent-Disk GPU Backend

Files likely touched:

- `services/worker/worker/training/providers.py`
- `services/orchestrator/internal/workers/model.go`
- `services/orchestrator/internal/api/handlers.go`
- deployment/docs files

Behavior change:

- Add an optional provider for persistent disk workers.
- Reuse checksum-keyed dataset materialization across many jobs.
- Keep the orchestrator job contract unchanged.

Migration/config needs:

- Provider capability config.
- Worker disk cache root, TTL, and size cap.

Tests:

- Persistent provider claims only compatible jobs.
- Dataset cache persists across worker restarts.
- Cache eviction does not delete active datasets.

Rollout flags:

- `MODEL_EXPRESS_PERSISTENT_GPU_PROVIDER=1`

Risks:

- Operational complexity.
- Persistent workers can cost more if left running.

Acceptance criteria:

- Large repeated YOLO/classification projects can avoid repeated serverless materialization.
- Provider costs are visible and bounded.

## 8. Acceptance Criteria

Concrete measurable outcomes:

- One dataset download/extract per Modal batch instead of per experiment.
- For a batch with one dataset and N preview trials, `dataset_materialization_cache_miss` is true for at most one materialization path in that batch.
- GPU/runtime spent in dataset materialization is reduced by at least 60% for 4+ same-dataset preview trials compared with the current per-job baseline.
- Warm idle container/process time is visible and bounded by mode defaults.
- Dispatcher idle time is visible, and cost-sensitive mode exits or pauses idle dispatchers.
- Preview runs reduce full-train count by at least 40% on multi-trial plans while preserving explicit rescue paths.
- YOLO runs compare preprocessing variants with the same subset, seed, split, model, and budget unless the changed variable is explicitly recorded.
- A weak YOLO11n preview cannot globally block preprocessing or larger-model hypotheses without paired evidence.
- Modal prewarm telemetry matches real training cache hits, or prewarm is marked as staging-only.
- Costs are visible per project, per job, per GPU type, and per training stage.
- Budget caps are enforced by backend scheduling, not only by LLM prompt guidance.

## Open Assumptions To Verify

- Whether Modal bills during `scaledown_window` the way this plan treats as idle cost risk. Verify against Modal billing exports and function-call timestamps.
- Whether the current durable dataset Volume prewarm is ever consumed by training under any environment configuration. Verify by comparing prewarm and training cache roots, mounts, and hit/miss telemetry.
- Whether `estimated_cost_usd` hard-coded rates match current Modal pricing. Verify against a billing export or official pricing before using as user-facing final cost.
- Whether local `/tmp` survives across enough Modal warm invocations to matter. Treat it as opportunistic until telemetry proves it.
- Whether YOLO epoch CSV can be streamed or polled early enough for live early abort. If not, first implement post-train detection diagnostics and then add active abort later.
