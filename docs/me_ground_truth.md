# Model Express Ground Truth

This document is the durable system reference for Model Express. It describes how the system works now, including current safety boundaries and known limits.

## Core Boundary

Model Express is an agentic image-classification experiment platform with a strict control boundary:

```text
LLMs propose structured JSON -> backend validates -> backend stores/schedules -> workers execute
```

LLMs never directly create workers, mutate datasets, choose arbitrary filesystem paths, enqueue jobs, export models, or run inference. Every executable action must pass through deterministic Go backend validation.

For Experiment Planner `ADD_EXPERIMENTS` decisions, the LLM proposes mechanism-level `candidate_hypotheses`. The backend finalizer is authoritative: direct LLM `proposed_experiments` are draft-only, and the backend ranks candidates before populating final `proposed_experiments` and `proposal_mechanisms`.

## Main Components

- Mission Control (`apps/mission-control`): Electron/Vite/React operator UI for projects, datasets, plans, jobs, agents, workers, champions, export records, and demo surfaces.
- Orchestrator (`services/orchestrator`): Go control plane. It owns API validation, Postgres/in-memory stores, plan execution, job assignment, agent invocations, memory, decisions, scorecards, champions, export records, and execution events.
- Worker (`services/worker`): Python runtime. Workers register with the orchestrator, poll for jobs, profile datasets, run local or Modal training, produce controlled champion export/demo/exemplar job results, report metrics/summaries/evaluations, and complete/fail jobs.
- Postgres: durable control-plane store. In-memory store mirrors behavior for tests/dev.
- LLM agents: Training Monitor and Experiment Planner. They produce JSON recommendations only.

## Data Model

Major tables and store concepts:

- `projects`: project name, goal, status.
- `datasets`: dataset metadata and canonical legacy `profile` JSON.
- `dataset_profiles`: exists in migration but is not active. `datasets.profile` is the current source of truth.
- `dataset_metadata_imports`: backend-owned normalized metadata import attempts/versions with active-import selection, parser provenance, full summary, agent-safe summary, warnings, and errors.
- `dataset_metadata_sources`: bounded provenance for discovered/imported sidecars, including parser status and bounded raw previews for backend/operator debugging only.
- `dataset_classes`: normalized class vocabulary for an import.
- `dataset_manifest_records`: canonical sample records with relative path, label, split, dimensions/checksum fields, and source provenance.
- `dataset_annotations`: optional normalized per-sample annotations such as bounding boxes.
- `dataset_splits`: explicit or derived split declarations/counts.
- `experiment_plans`: planned experiment batches and source decision linkage.
- `experiment_jobs`: queued/assigned/running/terminal worker jobs with attempt counts and lease fields for stale-job recovery.
- `epoch_metrics`: idempotent metrics keyed by `(job_id, epoch)`.
- `training_run_summaries`: run-level status, metrics, runtime, cost, provider, model.
- `training_run_evaluations`: objective profile, per-class precision/recall/F1 metrics, confusion matrix, model profile, holistic scores, and backend-added training diagnostics.
- `project_champions`: selected champion job plus metrics, evaluation, deployment profile.
- `champion_exports`: additive export records for selected champions; repeated requests for the same project/champion/format update the existing record.
- `champion_demo_predictions`: demo prediction audit/history rows for selected champions, including image metadata, status, top-k payloads when available, latency/correctness, and runtime errors.
- `agent_invocations`: LLM input/output, validation status, parsed output, downstream outcome, and invocation runtime metadata. When the provider returns usage, `input_context.invocation_runtime.llm_usage` stores exact input, output, total, cached-input, and reasoning-token counts plus request model, API style, and tool round count; byte-based token estimates remain the fallback.
- `agent_memory_records`: distilled training/planning memory.
- `agent_memory_embeddings`: compact vector-memory rows with source-card text hashes so unchanged source cards are not re-embedded for the same model/dimensions/content.
- `memory_embedding_usage_events`: embedding spend telemetry split by `source_index` versus `retrieval_query`, including provider-call count, input bytes, source/query hash, retrieved count, injected/log-only/cache/skipped flags, and provider usage when available.
- `memory_retrieval_query_cache`: short-lived cache for repeated retrieval-query embeddings by project/dataset scope, purpose, model, dimensions, and normalized query hash.
- `agent_decisions`: accepted project decisions such as `ADD_EXPERIMENTS`, `SELECT_CHAMPION`, or `REOPEN_EXPERIMENTATION`.
- `strategy_scorecards`: structured follow-up outcome memory, including first-class mechanism/intervention/evidence metadata.
- `automl_studies`: backend-owned hyperparameter optimization studies linked to project, plan, dataset, experiment index, LLM-owned strategy snapshot, sampler, seed, and validated search space.
- `automl_suggestions`: concrete hyperparameter suggestions with per-value provenance, validation status/errors, and optional linked job ID.
- `automl_trials`: compact observed trial results linked to study/suggestion/job with target metric, score, metrics summary, and error text.
- `worker_requirements`: durable requests for Mission Control to satisfy worker capacity.
- `execution_events`: audit events for queued jobs, worker requirements, agent outcomes, champion selection, champion export requests, and demo prediction requests. The project SSE endpoint streams these durable rows as refresh hints.

## Dataset And Profile Flow

1. Mission Control creates a project and streams an image-folder dataset as a zip archive directly to configured object storage.
2. The backend creates a dataset row and queues a `profile_dataset` job.
3. A worker profiles the dataset. Modal-mode profile workers submit profiling to Modal, where the dataset archive is downloaded/extracted in temporary Modal storage. Local profile workers can still download/extract locally, but `.cache/datasets/:dataset_id` is cleaned by default after profiling unless `MODEL_EXPRESS_PERSIST_DATASET_CACHE=1` is set. The legacy image-folder profiler resolves common dataset layouts before counting classes, including single archive-wrapper folders and common image roots such as `images/`, `JPEGImages/`, `img(s)/`, and `data/`.
4. The worker discovers bounded metadata sidecar candidates and dataset file inventory, posts them opportunistically to `POST /datasets/:id/metadata/imports`, then posts legacy profile JSON to `POST /datasets/:id/profile`. If the metadata endpoint is unavailable, profiling continues through the legacy profile path.
5. The backend stores normalized metadata imports additively, makes the latest successful/partial import active only after persistence succeeds, stores agent-safe summaries separately from raw source previews, then stores profile JSON in `datasets.profile`, marks the dataset profiled, and creates the initial experiment plan if one does not exist.

Profile JSON includes class counts, image counts, image dimension stats, corrupt-file counts, split summaries, metadata/artifact detection, leakage warnings, dataset traits, layout summary, normalization metadata, and optional `visual_exemplars`/`demo_images` arrays. CUB-style `bounding_boxes.txt` is treated as bbox evidence in the legacy profile/visual-trait summary, even though normalized metadata remains the richer source for executable bbox crops.

The legacy worker profiler now accepts YOLO object-detection datasets. It detects `data.yaml`, `data.yml`, `dataset.yaml`, or `dataset.yml`, plus `labels/**/*.txt` files containing normalized YOLO rows shaped as `class_id x_center y_center width height`. YOLO evidence sets `task_type: object_detection` and emits `yolo_available`, `object_detection_available`, `bbox_count`, `bbox_per_class`, and `yolo_summary` at the profile/metadata-summary surfaces. `yolo_summary` includes bounded class names/ids, label-file counts, bbox counts, split hints, per-split image/label/bbox counts, and per-class box distribution. The profiler supports YOLO `names` as an inline list, inline dict, keyed YAML block, or list YAML block.

Accepted YOLO dataset structures for full current use:

```text
dataset/
  data.yaml
  images/train/*.jpg
  images/val/*.jpg
  images/test/*.jpg        # optional
  labels/train/*.txt
  labels/val/*.txt
  labels/test/*.txt        # optional
```

```text
dataset/
  data.yaml
  train/images/*.jpg
  train/labels/*.txt
  valid/images/*.jpg       # val is also accepted
  valid/labels/*.txt
  test/images/*.jpg        # optional
  test/labels/*.txt        # optional
```

`data.yaml` should declare `train`, `val` or `valid`, optional `test`, `nc`, and `names`. `train.txt`, `val.txt`, and `test.txt` split files referenced by `data.yaml` are accepted as split hints. Label files use one object per line with 0-based class ids and normalized coordinates in `[0, 1]`; empty label files mean the image has no annotated objects. YOLO segmentation, pose/keypoint, rotated-box, tracking, and video datasets are not part of the current accepted detector contract.

Normalized dataset metadata imports currently support image/split-folder inventory, single archive-wrapper folders, common image roots such as `images/`, `JPEGImages/`, `img(s)/`, and `data/`, generic CSV manifests, CUB sidecars, CUB part-location keypoints, and Pascal VOC XML bounding-box evidence. Manifest and sidecar paths are canonicalized against the dataset inventory when possible so metadata that omits the wrapper or image-root prefix can still resolve to the stored dataset path. Worker discovery sends up to 50,000 inventory files by default so common 10k-image datasets can be canonicalized without truncating the image inventory. Unsupported metadata-like files, including attribute/part/keypoint candidates outside supported parsers, are recorded as warnings rather than blocking normal registration in non-strict mode. YOLO is currently accepted through the legacy profiler/training path, not as a normalized metadata import parser. `datasets.profile` remains the legacy profile surface; normalized imports live beside it and do not activate `dataset_profiles`.

Current dataset metadata APIs:

- `POST /datasets/:id/metadata/imports`
- `GET /datasets/:id/metadata/imports`
- `GET /datasets/:id/metadata/imports/:import_id`
- `GET /datasets/:id/metadata/summary` defaults to agent-safe output.
- `GET /datasets/:id/metadata/bundle?purpose=training&include=bbox&limit=&offset=`

`GET /datasets/:id/metadata/summary` returns an empty 200 response when the dataset exists but has no active metadata import yet. `dataset_profiles` rows are still deferred. They should not become active until there is a complete write path, latest-profile read path, backfill plan, and tests proving one canonical source.

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
- `augmentation_policy_config`
- `augmentation`
- `class_balancing`
- `class_balancing_config`
- `sampling_strategy`
- optimizer, scheduler, weight decay, dropout, optimizer momentum, scheduler step parameters, label smoothing, gradient clipping, fine-tune strategy, pretrained/freeze flags
- first-class mechanism metadata: `mechanism`, `intervention`, `evidence_used`, `expected_effect`

The backend validates allowed model names, epochs, batch size, learning rate, image size, optimizer, scheduler, optimizer-specific knobs, scheduler-specific knobs, dropout, label smoothing, gradient clipping, preprocessing values, augmentation keys, augmentation policy, class balancing config, sampling strategy, early stopping, and fine-tune strategy.

Planning is task-aware for YOLO evidence. When a dataset profile or agent-safe summary contains YOLO-specific evidence, the supported model catalog includes COCO-pretrained YOLO11 detector candidates: `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, and `yolo11x.pt`. These entries are marked `task_type: object_detection`, `model_kind: ultralytics_yolo_detector`, `pretrained_weights`, default image size `640`, and `training_enabled: true`. The deterministic initial planner creates YOLO detector plans with target metric `mAP50_95` instead of classifier plans. Execution validates dataset/model compatibility: YOLO detector models require YOLO evidence, classifier models are rejected for YOLO object-detection datasets, and detection templates cannot be paired with classifier models. Detector jobs still use the existing `train_experiment` template but carry explicit `task_type`, `model_kind`, `pretrained_weights`, class names, and bounded YOLO summary metadata in job config.

AutoML is an optional backend hyperparameter suggestion layer, disabled by default through `automation_settings.automl_enabled`. When enabled, the backend can synthesize a default validated AutoML search space for a proposed experiment that omits `PlannedExperiment.automl`, so the LLM does not have to remember the AutoML JSON shape for every run. LLM-provided AutoML metadata is still honored when present, and an explicit disabled AutoML block remains disabled. The LLM remains responsible for strategy: model/template, preprocessing, image size/resolution intent, augmentation policy type, class-balancing strategy, sampling strategy, pretrained/freeze/fine-tune strategy, exploration/exploitation intent, and bounded search constraints.

AutoML may tune only backend/worker-supported hyperparameters: `learning_rate`, `weight_decay`, `batch_size`, `epochs`, `early_stopping_patience`, `optimizer`, `scheduler`, `dropout`, `optimizer_momentum` only with SGD, `scheduler_step_size` and `scheduler_gamma` only with the step scheduler, `label_smoothing`, `gradient_clip_norm`, selected structured augmentation config values after the LLM chose the policy type, `class_balancing_config.effective_number_beta` only with effective-number class balancing, and `class_balancing_config.focal_loss_gamma` only with focal loss. It cannot tune strategy-owned fields such as `model`, `template`, `preprocessing`, `resolution_strategy`, `image_size`, `augmentation_policy`, `augmentation_policy_config.policy_type`, `class_balancing`, `sampling_strategy`, `pretrained`, `freeze_backbone`, or `fine_tune_strategy`. Samplers are `seeded_random`, `grid`, and lightweight `adaptive_bayesian`, which exploits compact persisted trial history without adding a heavy Bayesian dependency.

Duplicate experiment signatures include preprocessing, augmentation policy/config, class-balancing config, sampling strategy, and resolution strategy. Backend follow-up validation also rejects or filters repeated mechanisms where the only changes are minor tuning knobs such as epochs, batch size, or learning rate.

LLM-originated follow-up proposals must name a mechanism and evidence-backed intervention. Label-quality mechanisms can only schedule report-only `label_quality_audit` jobs; they cannot create training jobs or mutate labels. MixUp/CutMix require bounded mixed-sample augmentation config. Effective-number class balancing requires bounded `class_balancing_config.effective_number_beta`.

The Experiment Planner now has a project trajectory governor. Planner input includes a `project_trajectory_card` with completed training runs, completed planner rounds, first successful score, current champion score, absolute champion gain, gain per completed run, recent best delta, minimum useful delta, no-improvement rounds, decision pressure, mechanism outcomes, blocked mechanisms, and warnings. The backend computes this card from prior plans, jobs, summaries, scorecards, rejected options, and champion context.

Mechanism outcomes aggregate attempts, plan counts, best score, best delta versus prior champion, recent best delta, cost, runtime, status, and exhaustion reason. Repeated low-yield `architecture_challenge` attempts are marked exhausted, so plateau alone no longer justifies another backbone/model-family sweep. Exhausted or blocked mechanisms are rejected by backend candidate ranking. Plateau exits include champion confirmation, non-architecture pivots such as class imbalance or preprocessing/resolution work, label/hard-example audits, `SELECT_CHAMPION`, `STOP_PROJECT`, and `WAIT`.

Autonomous mode is more conservative than propose mode. Unless overridden by environment variables, autonomous planner guards default on, the minimum meaningful target-metric delta defaults to `0.010`, and automatic follow-up rounds default to `3`. Propose/manual mode keeps the more flexible legacy defaults unless explicitly configured.

When an LLM planner proposal passes JSON/schema parsing but fails backend validation, the backend records the rejected invocation outcome and performs one bounded correction retry with `planner_validation_feedback` in the prompt context. The retry can only succeed if the corrected JSON passes the same backend validation and scheduling gates.

The default LLM model is `gpt-5.4-mini` unless explicitly overridden by environment or automation settings. The planner supports `MODEL_EXPRESS_PLANNER_STATIC_PROMPT_VERSION=compact_v1` and `MODEL_EXPRESS_PLANNER_CONTEXT_VERSION=v2`; V1 prompt/context behavior remains selectable for rollback. Prompt compression preserves the existing `ExperimentPlanningRecommendation` contract, including schema-valid `ADD_EXPERIMENTS` candidates with complete executable experiment configs.

## Training Flow

1. A user or autonomous backend path executes a validated plan through `POST /plans/:id/execute`.
2. The backend creates one `train_experiment` job per novel experiment/provider/index.
3. Workers poll `POST /workers/:id/poll` and receive assigned jobs with backend-owned attempt and lease metadata.
4. Local training simulates contract behavior for fast local demos, including a detection-shaped YOLO simulator for object-detection jobs.
5. Modal training performs torchvision transfer learning for supported classifier model families and Ultralytics YOLO training for supported detector jobs.
6. Workers report epoch metrics, summary updates, and final evaluation.
7. Workers complete or fail the job through backend APIs.

For AutoML-enabled experiments, the orchestrator samples concrete hyperparameters before job creation, persists study/suggestion provenance, and places only compact `automl_summary`, `automl_study_id`, and `automl_suggestion_id` metadata in the job config. Backend default AutoML samples concrete knobs such as learning rate, weight decay, batch size, epochs, early stopping, dropout, label smoothing, gradient clipping, optimizer/scheduler conditional numeric knobs, and numeric policy/loss parameters only after the LLM has selected the owning strategy. Workers still receive concrete training config and do not run search, schedule follow-up jobs, mutate datasets, or choose strategies.

Worker-side classifier training currently supports deterministic resize/crop options, ImageNet/none normalization, bounded dataset-computed normalization, structured image augmentation, training-only MixUp/CutMix, weighted/focal/effective-number loss, weighted sampling, dropout heads, SGD momentum, step scheduler size/gamma, label smoothing, focal-loss gamma, gradient clipping, and bbox crop/full-image ablations when backend-validated annotations are available. Modal classifier training fetches the active normalized metadata bundle when present and can use backend-owned labels, official splits, and bbox annotations; train/test-only metadata derives a deterministic validation subset from train while keeping test held out. If the bundle is absent or unavailable, legacy ImageFolder/random split behavior remains the fallback, with single archive wrappers and common image roots such as `images/` handled before falling back to the archive root.

YOLO detector training is separate from classifier training. Local detector jobs emit deterministic mAP/precision/recall/box-loss/cls-loss/DFL-loss metrics so control-plane E2E smoke tests can run without a GPU. Modal detector jobs dispatch to `train_yolo_detector`, load the dataset's `data.yaml`/`dataset.yaml`, train with Ultralytics from the requested YOLO11 pretrained weights, preserve official train/val/test splits, validate on `val`, and run final `test` evaluation when the config declares a test split. Detector summaries map `best_macro_f1` to `mAP50_95` and `best_accuracy` to `mAP50` for backward-compatible champion ranking, while the evaluation payload also carries explicit detection metrics, detection losses, latency/FPS estimates, `task_type: object_detection`, and `model_kind: ultralytics_yolo_detector`.

Training evaluations for image classification include confusion matrices and per-class precision/recall/F1 when real validation/test labels are available. The backend enriches final evaluation payloads with deterministic `training_diagnostics` inside `holistic_scores`, including train/validation loss gap, divergence status, severity, and trend deltas derived from persisted epoch metrics and run summaries.

Worker utility modules also include:

- bounded metadata discovery handoff for sidecars/inventory, including common labels/manifests, CUB files, attributes, parts, landmarks, and keypoints, without semantic parsing in the worker
- split-file, Pascal VOC XML, and annotation JSON helper parsers
- class-balanced visual exemplar and visual-analysis sample generation with PIL downscale/compression and byte/image caps; both unwrap single archive folders and common image roots before treating immediate subfolders as classes
- report-only label-quality audit jobs that persist capped profile audit metadata without label mutation
- champion export manifest/checkpoint helpers with guarded TorchScript/ONNX paths and detection-aware export metadata contracts
- TorchScript demo inference helper that returns a ranked payload when a valid worker-owned artifact exists, or a deterministic pending/error payload when dependencies/artifacts are missing
- champion job handlers for `export_champion`, `champion_demo_prediction`, and `generate_visual_exemplars`

Arbitrary split-file parsing in workers remains deferred, but backend-normalized official splits from supported imports can now drive Modal training. Model routing and prompt caching are still intentionally deferred. Vector retrieval is implemented as compact decision memory behind rollout flags; cross-project retrieval remains disabled by default.

## Agent Flow

Training Monitor evaluates individual completed/failed training jobs and writes training-evaluation memory.

Experiment Planner reviews completed plans and can recommend:

- `ADD_EXPERIMENTS`
- `SELECT_CHAMPION`
- `STOP_PROJECT`
- `WAIT`

Planner input is compacted into decision-ready cards rather than raw table dumps. It includes project and dataset cards, optional agent-safe normalized metadata summaries, optional visual evidence, objective context, deterministic diagnosis, project trajectory, training dynamics, per-class errors, deployment, mechanism coverage, label quality, a compact task-separated model catalog, current champion, run deltas, distilled memory lessons, rejected options, scorecards, retrieved memory cards when enabled, validation feedback, and existing experiment signatures.

For `ADD_EXPERIMENTS`, the Planner must return `candidate_hypotheses` with mechanism, intervention, evidence, expected effect, expected metric impact, tradeoffs, risk/cost/novelty, proposed changes, and a complete executable experiment config. `FinalizePlannerRecommendation` then applies deterministic backend ranking and governor checks. If no candidate survives, the planner output is rejected rather than scheduled.

Planner decisions persist the `project_trajectory_card` alongside candidate rankings, so operator-facing audit views can explain why the backend selected, rejected, blocked, or stopped a recommendation.

Live/real-time objective handling treats latency as a budget and tiebreaker rather than a primary search driver. Latency below roughly 25ms is considered acceptable for live use, so the Planner and Training Monitor should prioritize macro-F1, per-class recall, and meaningful quality gains unless latency exceeds the budget or quality is otherwise close.

Experiment Planner context includes `prompt_budget.section_estimates` so recent invocations can rank static instructions, output contract text, model catalog, retrieved memory, completed experiment ledger, and other prompt sections by approximate token weight. Training Monitor input is also compacted into run-evaluation cards. The backend still stores full run summaries, evaluations, epoch metrics, plans, and job configs, but the LLM receives capped cards and prompt-budget telemetry.

Vector retrieval indexes compact memory cards only: strategy scorecards, distilled planning/training memories, dataset profile fingerprints, accepted visual-analysis cards, and preprocessing hypotheses. New eligible writes trigger indexing when embeddings are configured, and existing project memories can be indexed through `POST /projects/:id/memory-embeddings/backfill`. Source indexing records usage telemetry, pre-checks unchanged source-card hashes, obeys daily call caps, and uses a conservative backfill limit. It does not vectorize raw prompts, raw LLM outputs, full invocation contexts, full epoch arrays, full manifests, image URIs, or unbounded JSON payloads.

Retrieval-query embeddings are accountable separately from source indexing. Log-only memory probes use lexical fallback unless `MODEL_EXPRESS_MEMORY_LOG_ONLY_EMBEDDINGS=true`; repeated queries use the retrieval-query cache; too-small indexes and exhausted budgets downgrade to lexical retrieval. Retrieval usage records whether cards were returned and whether they were actually injected into Planner or Training Monitor context. Retrieval can inform planner/monitor context and deterministic ranking, but backend validation remains the execution gate.

When AutoML trials exist, the Planner and Training Monitor receive compact `optimizer_feedback_summary` cards: trial counts, best score/job, best hyperparameters, train/validation gap, trend, failed-trial count/patterns, and bounded narrowing advice. Raw trial dumps are not included in default LLM context.

Visual exemplars are evidence only. Backend-curated/capped metadata may help the planner cite object scale, background dominance, blur, lighting, fine-grained classes, or bbox/crop plausibility. Planner context includes exemplar caps and audit details when available. Planner output is still JSON only, and backend validation remains the gate.

Normalized dataset metadata summaries are also evidence only in visual-analysis and planner context. Queued visual-analysis jobs include the active agent-safe metadata summary when present, and result validation merges the active summary into the profile evidence before gating bbox-dependent hypotheses. Initial planning also merges active metadata counts/capabilities so a corrected normalized import can prevent stale legacy profile fields from blocking plan creation. Planner tools expose `dataset_metadata_summary` as a compact safe summary and exclude source rows, relative paths, storage URIs, raw previews, sidecar contents, and manifest record dumps. Backend gates can use normalized bbox evidence from the safe summary in addition to legacy profile traits, and safe capability flags now include bbox, attribute, and keypoint annotation availability.

## Champion Flow

Champion selection can come from deterministic review or validated planner decisions. `SELECT_CHAMPION` persists a `project_champions` row with:

- selected job, plan, dataset, source decision
- selection reason
- metrics from the training run summary
- evaluation payload if available
- deployment profile and model-card-style pending export notes

If a validated `STOP_PROJECT` decision does not name a champion but successful runs exist, the backend selects the best successful run so far using the same objective-aware ranking helper used for planner context. This produces a usable champion without letting the LLM execute work or bypass validation.

Mission Control displays the selected champion, champion comparison table, objective fit, model profile, confusion preview, train/validation gap, seed variance when repeated seeded runs exist, and deployment notes.

Champion selection records the current best deployment candidate, but it is no longer terminal for the monitored autonomous improvement loop by default. A selected champion can remain the comparison anchor while the Planner continues proposing backend-validated follow-up experiments and accumulating richer strategy memory. Legacy terminal champion/near-ceiling/no-improvement guards can be re-enabled with `MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS=true`. The reopen endpoint still records `REOPEN_EXPERIMENTATION` and `EXPERIMENTATION_REOPENED` when an operator wants an explicit audit marker.

## Export And Demo Flow

Current backend export/demo APIs:

- `GET /projects/:id/champion/exports`
- `POST /projects/:id/champion/exports`
- `GET /projects/:id/champion/demo-images`
- `GET /projects/:id/champion/demo-predictions`
- `POST /projects/:id/champion/demo-predictions`
- `GET /datasets/:id/visual-exemplars`
- `POST /datasets/:id/visual-exemplars`
- `GET /datasets/:id/metadata/summary`
- `GET /datasets/:id/metadata/imports`
- `POST /jobs/:id/champion-export-result`
- `POST /jobs/:id/champion-demo-prediction-result`

`POST /projects/:id/champion/exports` validates that a champion exists, the champion job succeeded, and the requested format is supported. If worker work is needed, the backend records the export and queues an `export_champion` job. Workers report results through `POST /jobs/:id/champion-export-result`; the backend validates the job template/config before updating `champion_exports`. If no artifact can be produced, the export stays `PENDING_ARTIFACT` or `FAILED` with validation errors rather than pretending it is ready. Repeated requests for the same project/champion/format are idempotent at the store/API behavior level.

Champion export metadata now exposes a model-agnostic live inference contract. Metadata includes `model_kind`, `task_type`, `runtime`/`default_runtime`, input shape, class labels, confidence-threshold defaults, latency budget/profile fields when known, export self-test placeholders, preprocessing contract, and postprocessing contract. Classification postprocessing is logits plus softmax. Detection/YOLO postprocessing describes boxes, scores, classes, confidence threshold, IoU threshold, NMS, and max detections. YOLO Modal training attempts an ONNX export from the trained Ultralytics checkpoint and falls back to a checkpoint artifact with validation errors if ONNX export is unavailable.

Demo image listing reads capped metadata from `datasets.profile.demo_images` or `datasets.profile.visual_exemplars`. Demo prediction requests create durable `champion_demo_predictions` audit rows. When a `READY` champion export exists, the backend queues a `champion_demo_prediction` job and workers report results through `POST /jobs/:id/champion-demo-prediction-result`. Demo requests can carry detector confidence, IoU, and max-detection settings. Missing manifests, missing dependencies, or unavailable local artifacts are recorded as `RUNTIME_UNAVAILABLE`; the backend never fabricates predictions. Detection results are persisted in the existing `image_metadata` JSON as bounded boxes/classes/scores so no database migration is required.

Workers can generate class-balanced exemplars with `generate_visual_exemplars` and persist the capped result through `POST /datasets/:id/visual-exemplars`. The backend enforces count/per-class/byte caps and merges accepted `visual_exemplars` and `demo_images` into canonical `datasets.profile` JSON.

Mission Control can request an ONNX export record, show export status/errors, list demo images, browse next/random demo examples, run timed slideshow inference, and display prediction history, pending/running/succeeded/failed/runtime-unavailable status, true labels, top-k rows, latency, and correctness fields when present. Local ONNX inference supports classifier logits/softmax and YOLO-style detector outputs. For detector champions, Mission Control decodes common Ultralytics ONNX output shapes, applies confidence filtering and class-aware NMS, draws box overlays, exposes confidence/IoU/speed controls, and shows per-frame detection count, FPS estimate, and postprocess latency. Worker-backed demo inference also supports ONNX detector artifacts and returns the same detection metadata shape when the browser cannot run local inference.

## Mission Control Flow

Mission Control polls project detail endpoints and optionally consumes `GET /projects/:id/events/stream` as a server-sent-event refresh hint. Polling remains the durable fallback. It renders:

- section navigation for Overview, Data, Agents, Runs, Operations, Export/Demo
- dataset intelligence, normalized metadata import status, and preprocessing recommendations
- automation settings and worker requirements
- experiment timeline and execution events
- agent decisions, rejections, candidate score components, and scorecards
- read-only decision-quality visibility for project trajectory pressure, blocked mechanisms, completed runs, gain per run, recent best delta, selected/rejected candidate counts, and top rejection reasons
- plans with typed preprocessing fields
- run summaries/evaluations, metric charts, per-class metrics, confusion previews, and backend training diagnostics
- selected champion and export/demo panel
- champion demo prediction history and runtime-unavailable audit rows
- LLM/embedding telemetry showing today/7-day/lifetime token and cost estimates, exact versus approximate usage, model split, token-heavy invocations, prompt-section offenders, source-index embedding usage, retrieval-query hit/injection/cache/skipped usefulness, and embedding cost estimates

Mission Control does not invent orchestration state. It uses backend APIs, renders partial data defensively, and surfaces rejection/failure states as part of operator trust. The Electron orchestrator request bridge returns HTTP error envelopes to the renderer instead of throwing inside the IPC handler, so expected optional-endpoint or empty-state failures can be handled without noisy main-process handler errors.

Mission Control exposes AutoML HPO settings, including enabled/disabled state and sampler. Experiment plan cards show compact AutoML status, sampler, search parameters, concrete suggestions, and provenance per displayed value. Job summaries surface that a job came from an AutoML suggestion without exposing raw trial history.

Mission Control also reads dataset metadata summary/import endpoints defensively. When available, Dataset Intelligence shows import status, source kinds/formats, class/sample counts, official split availability, split/annotation counts, bbox coverage, unsupported-source warnings, and errors. Missing metadata endpoints fall back to the legacy profile display.

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
- Experiment Planner `ADD_EXPERIMENTS` cannot bypass backend candidate ranking.
- Dry-run planner validation uses the same backend finalizer as scheduling, so repair prompts receive finalizer rejection reasons.
- Deterministic replay evals under `services/orchestrator/internal/agents/evals` include plateau, image-classification, and YOLO smoke fixtures. Replay compares current V1, compact static prompt, V2 compact context, and distilled-memory-first variants for parse/finalizer success, duplicate-signature avoidance, mechanism diversity, task/model alignment, and prompt-size ordering.
- AutoML is settings-gated, backend-validated, persisted with provenance, and linked to normal plan/job execution rather than owning scheduling.
- Normalized dataset metadata imports are additive, active-import replacement is transactional in Postgres, and agent-safe summaries are separated from raw source previews.
- Vector retrieval uses a separate `agent_memory_embeddings` table, pgvector in Postgres, in-memory lexical/vector fallback for tests, prompt-budget caps, log-only rollout mode by default, source-index/retrieval-query usage telemetry, query caching, unchanged-card hash guards, daily caps, and deterministic replay eval telemetry.

Known reliability gaps:

- Lease recovery currently runs when workers poll or when `RecoverExpiredJobLeases` is called; there is no standalone recovery ticker.
- Some duplicate prevention is still application-level rather than DB idempotency-key based.
- Terminal job endpoints can still synchronously trigger agent work.
- `autoReviewMu` is process-local.
- SSE streams recent durable events as refresh hints, not as a separate event-sourcing system.

## Deferred Work

- Fully promote `dataset_profiles` or keep it permanently documented as deferred.
- Persist visual exemplar generation/audit history beyond compact profile JSON/planner context.
- Expand normalized metadata parsers beyond the MVP CSV/CUB/VOC/image-folder set, including COCO, OpenImages, CVAT, Label Studio, JSONL/YAML, video metadata, and a normalized YOLO import path separate from the current legacy YOLO profiler/training path.
- Add full export self-test/parity runs over heldout images, including framework-vs-export drift, p50/p95 latency, FPS, failure rate, low-confidence rate, and detector runtime parity checks.
- Add richer worker/local-training support for normalized metadata bundles beyond the Modal path.
- Advanced augmentation object policies.
- Production storage upload for generated export/exemplar artifacts beyond current worker-local `file://` URIs.
- Real model reconstruction/export from completed training runs when no worker-visible artifact exists.
- Heavier Bayesian/TPE/GP AutoML dependencies and richer multi-trial acquisition policies beyond the current lightweight adaptive sampler.
- Durable idempotency keys, async agent task queue, and a standalone lease-recovery loop.
- Multi-agent planner debate, tree search/MCTS over plans, planner fine-tuning, and provider prompt caching beyond recorded cached-token telemetry.

Do not add Kafka, Redis, NATS, WebSockets, or a workflow engine until Postgres hardening, leases, idempotency, and SSE are no longer sufficient.
