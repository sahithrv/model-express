# Python Worker / Training Agent Context

## Mission And Scope

The Python Worker / Training Agent owns the execution-plane context for Model Express image-classification work. Its job is to keep worker behavior aligned with the deterministic Go orchestrator contract: workers register, poll, run controlled jobs, report metrics/summaries/evaluations, and complete or fail jobs through orchestrator APIs.

Primary scope:

- `services/worker`: Python worker runtime, dataset download/cache/profile helpers, local training simulator, Modal training provider, training templates, and future export/inference helpers.
- Worker-facing contracts in `services/orchestrator/internal/api`, `jobs`, `plans`, `runs`, `datasets`, and `workers`.
- Mission Control expectations for worker status, dataset profile JSON, training summaries, evaluations, champion comparison, and future champion demo surfaces.

Non-goals for this agent unless explicitly requested: Go store migrations, frontend redesign, LLM planner policy, cloud provisioning, or replacing the orchestrator as the source of truth.

## Current Worker Flow

`worker.main.main` reads `ORCHESTRATOR_URL` and registers a project-scoped worker with `POST /workers/register` using `PROJECT_ID`, `WORKER_NAME`, and `GPU_TYPE`. It then loops forever on `POST /workers/:id/poll`; no job means sleep for 5 seconds. Any raised job exception is reported with `POST /jobs/:id/fail`.

`worker.jobs.run_job` dispatches by job template:

- `profile_dataset`: fetch dataset metadata from `GET /datasets/:id`, download the `s3://` archive through MinIO/S3 settings, extract to `.cache/datasets/:dataset_id`, run `profile_image_folder`, post profile JSON to `POST /datasets/:id/profile`, then complete the job.
- `train_experiment`: call `run_training_job`, which selects `config.provider` with default `local`.
- Unknown templates currently run the fake metric loop in `OrchestratorClient.run_fake_job`.

Training providers:

- Local: `worker.training.local.run_local_training` is a deterministic simulator. It reads model/training/preprocessing config, emits epoch metrics, upserts running/final training summaries, posts a synthetic evaluation payload, and completes the job. This path locks backend/frontend contracts without doing real PyTorch training.
- Modal: `worker.training.modal_provider.run_modal_training` validates that `MODAL_ORCHESTRATOR_URL` and `MODAL_S3_ENDPOINT_URL` are public HTTP(S) URLs, loads the dataset, and submits `train_image_classifier.remote`. `worker.training.modal_app.train_image_classifier` performs real torchvision transfer learning, reports metrics and summaries during training, posts a final evaluation, and completes the job.

## Key Files To Inspect

- `services/worker/worker/main.py`: worker registration and polling loop.
- `services/worker/worker/jobs.py`: template dispatch and dataset profiling job.
- `services/worker/worker/orchestrator_client.py`: backend API calls used by the worker.
- `services/worker/worker/datasets/storage.py`: `s3://` parsing and download.
- `services/worker/worker/datasets/cache.py`: dataset archive/extraction cache layout.
- `services/worker/worker/datasets/profiler.py`: image-folder profile shape, artifact detection, dataset traits.
- `services/worker/worker/datasets/annotations.py`: helper-only split-file, Pascal VOC XML, and annotation JSON parsing.
- `services/worker/worker/datasets/exemplars.py`: deterministic class-balanced visual exemplar generation with downscale/compression and byte/image caps.
- `services/worker/worker/training/providers.py`: provider switch.
- `services/worker/worker/training/local.py`: local simulator, synthetic scoring, evaluation payload shape.
- `services/worker/worker/training/modal_provider.py`: Modal submission and public URL checks.
- `services/worker/worker/training/modal_app.py`: real transforms, data split, model build, training loop, evaluation, cost/latency estimates.
- `services/worker/worker/exporting/artifacts.py`: worker-owned export manifest/checkpoint helpers with guarded TorchScript/ONNX export paths.
- `services/worker/worker/exporting/inference.py`: TorchScript demo inference helper that returns ranked predictions or deterministic pending/error payloads.
- `services/orchestrator/internal/api/router.go`: canonical endpoint list.
- `services/orchestrator/internal/jobs/model.go`, `plans/model.go`, `runs/model.go`, `datasets/model.go`, `workers/model.go`: worker-visible contracts.
- `apps/mission-control/src/types.ts` and `apps/mission-control/src/App.tsx`: frontend expectations for worker, job, profile, summary, evaluation, and champion data.

## Supported Options After Recent Work

Backend-planned experiment fields now pass through to worker job config: `image_size`, `resolution_strategy`, `preprocessing`, `optimizer`, `scheduler`, `weight_decay`, `augmentation`, `augmentation_policy`, `class_balancing`, `sampling_strategy`, `early_stopping_patience`, `pretrained`, `freeze_backbone`, and `fine_tune_strategy`.

Validated or documented options:

- `resolution_strategy`: `fixed`, `low_latency`, `compare_224_256`, `high_resolution_ablation`.
- `preprocessing.resize_strategy`: `squash`, `preserve_aspect_pad`, `center_crop`, `random_resized_crop`, `bbox_crop_if_available`.
- `preprocessing.normalization`: `imagenet`, `dataset`, `none`.
- `preprocessing.crop_strategy`: `none`, `center_crop`, `random_resized_crop`, `bbox_crop_if_available`, `bbox_crop_ablation`.
- `preprocessing.bbox_mode`: `ignore`, `crop_if_available`, `crop_and_compare_full_image`, `use_boxes_as_metadata`.
- `augmentation_policy`: `none`, `light`, `moderate`, `strong`, `custom`.
- `augmentation` keys: `horizontal_flip`, `vertical_flip`, `color_jitter`, `random_crop`, `random_rotation`, `random_erasing`.
- `optimizer`: `adamw`, `adam`, `sgd`.
- `scheduler`: `none`, `cosine`, `step`.
- `class_balancing`: `none`, `weighted_loss`, `class_weighted_loss`, `class_balanced_sampler`, `weighted_random_sampler`, `focal_loss`.
- `sampling_strategy`: `none`, `class_balanced_sampler`, `weighted_random_sampler`.
- `fine_tune_strategy`: currently used as `head_only`, `last_block`, or `full`.

Worker implementation details:

- Modal currently applies ImageNet normalization unless `normalization: "none"` is requested; `normalization: "dataset"` is future-facing.
- Modal now computes dataset normalization metadata when `normalization: "dataset"` or `use_dataset_normalization` is requested and applies the resulting mean/std in transforms.
- Modal supports `preserve_aspect_pad`, `center_crop`, random resized crop, ImageFolder train/val/test split, weighted sampler, weighted/focal loss, Adam/AdamW/SGD, cosine/step scheduler, early stopping, pretrained/frozen/last-block/full fine-tuning.
- Worker helper tests cover transform construction, sampler/loss selection, profile artifact detection, normalization metadata, and pure export/demo payload shapes.
- Worker helper tests also cover export manifests, dependency-guarded demo inference, visual exemplar generation, split-file parsing, Pascal VOC XML, and annotation JSON.
- Modal model support includes MobileNetV3 small/large, EfficientNet B0/B1/B2, RegNet Y 400MF, ResNet18/34, ConvNeXt Tiny, Swin T, and ViT B/16 via torchvision.
- Local simulates expected quality/latency/cost effects for the same contract but does not train real models.

## Deferred Worker Work

Keep these as explicit future work, not accidental partial implementations:

- Wire parsed split files and annotations into real training behavior.
- Implement real bbox crop vs full-image paired ablations.
- Extend dataset normalization beyond the current bounded mean/std helper if future training paths need cached stats or explicit provenance.
- Wire explicit split-file training instead of always using the deterministic random split.
- Add advanced augmentation policies beyond the current safe string subset.
- Wire champion export helpers into a backend-scheduled worker job that saves artifacts to configured storage.
- Wire lightweight champion inference into a backend-scheduled worker job or controlled runtime endpoint with top-k labels, confidence, latency, and true-label support when available.
- Generate model-card/export metadata: input size, normalization, class labels, latency, model size, training config, and limitations.
- Current worker slice provides export metadata, manifest/checkpoint helpers, dependency-guarded TorchScript/ONNX export paths, and TorchScript inference helpers. Backend job wiring and artifact upload remain deferred.
- Generate visual exemplar packs already exists as a helper; backend/profile upload wiring remains deferred.

## Backend And Frontend Contracts

Backend endpoints the worker uses:

- `POST /workers/register`: registers a project-scoped worker.
- `POST /workers/:id/poll`: assigns one queued job or returns `{"job": null}`.
- `GET /datasets/:id`: returns `storage_uri`, profile, status, and metadata.
- `POST /datasets/:id/profile`: stores profile JSON and may trigger initial planning.
- `POST /jobs/:id/metrics`: epoch metrics; metrics are upserted by `(job_id, epoch)`.
- `POST /jobs/:id/training-run-summary`: upserts summary fields for Mission Control and planner context.
- `POST /jobs/:id/training-run-evaluation`: upserts objective/per-class/confusion/model/holistic evaluation data.
- `POST /jobs/:id/complete`: marks terminal success; train jobs can trigger monitor/planner loops.
- `POST /jobs/:id/fail`: marks terminal failure; train jobs can still trigger review/planning logic.
- `GET /projects/:id/champion/demo-predictions`: Mission Control reads durable prediction history.
- `POST /projects/:id/champion/demo-predictions`: backend currently creates `RUNTIME_UNAVAILABLE` audit rows; a future worker-backed path should report `SUCCEEDED`/`FAILED` prediction records without bypassing backend validation.

Job config is produced by executing an experiment plan. The orchestrator injects `plan_id`, `dataset_id`, `experiment_index`, `experiment_template`, `model`, `epochs`, `batch_size`, `learning_rate`, `target_metric`, `provider`, `gpu_type`, plus optional planned experiment fields.

Mission Control consumes, defensively:

- Dataset profile keys such as `images_per_class`, `class_distribution`, `total_images`, `class_count`, `imbalance_ratio`, `corrupt_image_count`, dimensions, `metadata_summary`, and planner recommendations.
- Job status and worker status for timeline/operations panels.
- Training summaries for run tables and champion comparison.
- Training evaluations for latency, model size, confusion matrix, per-class metrics, and holistic scores.
- Project champion records from backend persistence; worker export/inference should attach clean metadata that fits this surface.

## Safe Implementation Boundaries

- Preserve the control boundary: agents and planners propose; the Go orchestrator validates, schedules, stores, and marks jobs; workers execute assigned work only.
- Do not let worker code create projects/plans/jobs or bypass backend validation.
- Keep payloads backward-compatible and additive where possible; Mission Control often reads optional generic records.
- Treat local training as contract simulation. Do not make frontend/backend correctness depend on local-only behavior that Modal cannot produce.
- Keep Modal remote reachability checks. Modal cannot call localhost URLs without an explicit tunnel/public URL.
- Keep dataset cache behavior predictable; avoid destructive cache cleanup outside `.cache/datasets`.
- Do not introduce new training options without updating backend validation, execution config pass-through, worker implementation, docs, and tests together.
- Prefer small focused worker tests before touching real training behavior, especially transforms, samplers, class weights, model heads, and evaluation payload shape.

## Supporting Docs To Read

- `README.md`: current architecture, worker startup, and control boundary.
- `docs/data_preprocessing_agent_report.md`: preprocessing/profile schema, supported options, and deferred bbox/dataset-normalization work.
- `docs/model_express_agentic_upgrade_roadmap.md`: PR sequence, worker backlog, champion export/demo, and visual exemplar plans.
- `docs/smarter_agentic_orchestration_plan.md`: holistic evaluation, expanded model families, champion selection, and worker-facing metrics.
- `docs/orchestrator_system_design_report.md`: polling architecture, endpoint behavior, reliability gaps, and future eventing/lease work.
- `docs/frontend_mission_control_report.md`: frontend panels and backend/worker data still needed for Mission Control.
- `docs/agentic.md`: LLM subagent safety model and worker-requirement/auto-execution context.
