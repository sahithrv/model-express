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

- `profile_dataset`: fetch dataset metadata from `GET /datasets/:id`. When the worker is running with `GPU_TYPE=modal` or `MODEL_EXPRESS_DATASET_PROFILE_PROVIDER=modal`, submit a Modal profile function that downloads/extracts inside Modal temporary storage and returns profile JSON. Otherwise download the `s3://` archive through MinIO/S3 settings, extract locally, run `profile_image_folder`, post profile JSON to `POST /datasets/:id/profile`, then complete the job.
- `train_experiment`: call `run_training_job`, which selects `config.provider` with default `local`.
- `export_champion`: produce a controlled export manifest/artifact result when a worker-visible artifact/model is available, then report through `/jobs/:id/champion-export-result`.
- `champion_demo_prediction`: run dependency-guarded demo inference from a worker-owned export manifest, then report `SUCCEEDED`, `FAILED`, or `RUNTIME_UNAVAILABLE` through `/jobs/:id/champion-demo-prediction-result`.
- `generate_visual_exemplars`: generate capped class-balanced exemplars and persist the accepted profile patch through `/datasets/:id/visual-exemplars`.
- `label_quality_audit`: create a report-only dataset profile audit artifact for `audit_type` of `label_noise_audit` or `hard_example_audit`, merge it into `datasets.profile`, and complete the job locally without Modal execution or label mutation.
- Unknown templates fail closed instead of fake-running.

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
- `services/worker/worker/champion_jobs.py`: export/demo/exemplar job handlers and backend result payload construction.
- `services/orchestrator/internal/api/router.go`: canonical endpoint list.
- `services/orchestrator/internal/jobs/model.go`, `plans/model.go`, `runs/model.go`, `datasets/model.go`, `workers/model.go`: worker-visible contracts.
- `apps/mission-control/src/types.ts` and `apps/mission-control/src/App.tsx`: frontend expectations for worker, job, profile, summary, evaluation, and champion data.

## Supported Options After Recent Work

Backend-planned experiment fields now pass through to worker job config: `image_size`, `resolution_strategy`, `preprocessing`, `optimizer`, `scheduler`, `weight_decay`, `augmentation`, `augmentation_policy`, `augmentation_policy_config`, `class_balancing`, `class_balancing_config`, `sampling_strategy`, `early_stopping_patience`, `pretrained`, `freeze_backbone`, and `fine_tune_strategy`.

Validated or documented options:

- `resolution_strategy`: `fixed`, `low_latency`, `compare_224_256`, `high_resolution_ablation`.
- `preprocessing.resize_strategy`: `squash`, `preserve_aspect_pad`, `center_crop`, `random_resized_crop`, `bbox_crop_if_available`.
- `preprocessing.normalization`: `imagenet`, `dataset`, `none`.
- `preprocessing.crop_strategy`: `none`, `center_crop`, `random_resized_crop`, `bbox_crop_if_available`, `bbox_crop_ablation`.
- `preprocessing.bbox_mode`: `ignore`, `crop_if_available`, `crop_and_compare_full_image`, `use_boxes_as_metadata`.
- `augmentation_policy`: `none`, `light`, `moderate`, `strong`, `custom`, `basic`, `randaugment`, `trivialaugment`, `trivialaugmentwide`, `autoaugment`, `mixup`, `cutmix`.
- `augmentation` keys: `horizontal_flip`, `vertical_flip`, `color_jitter`, `random_crop`, `random_rotation`, `random_erasing`.
- `augmentation_policy_config`: optional structured augmentation object with `policy_type`, `magnitude`, `num_ops`, `num_magnitude_bins`, `probability`, and `alpha`.
  - Worker image-transform policies now supported: `basic`, `randaugment`, `trivialaugment`, `autoaugment` when available from torchvision.
  - Worker mixed-sample policies now supported in the real training loop: `mixup` and `cutmix`. They are normalized with capped `alpha` and `probability`, applied only inside training batches, and evaluation uses unmixed labels/images.
  - Legacy `augmentation_policy` strings and explicit `augmentation` map keys remain supported.
- `optimizer`: `adamw`, `adam`, `sgd`.
- `scheduler`: `none`, `cosine`, `step`.
- `class_balancing`: `none`, `weighted_loss`, `class_weighted_loss`, `class_balanced_sampler`, `weighted_random_sampler`, `focal_loss`, plus worker aliases for effective-number class-balanced loss: `effective_number`, `effective_number_loss`, `effective_number_class_balanced_loss`, `class_balanced_loss`, and `class_balanced_effective_number`.
- `sampling_strategy`: `none`, `class_balanced_sampler`, `weighted_random_sampler`.
- `fine_tune_strategy`: currently used as `head_only`, `last_block`, or `full`.

Worker implementation details:

- Dataset upload from Mission Control streams a zip directly to object storage; workers should continue treating `datasets.storage_uri` as the canonical dataset artifact.
- Local dataset profile cache under `.cache/datasets/:dataset_id` is cleaned after profiling by default. Set `MODEL_EXPRESS_PERSIST_DATASET_CACHE=1` only when intentionally debugging/reusing local extracted data. `MODEL_EXPRESS_DATASET_CACHE_ROOT` can override the root for local dataset jobs.
- Modal dataset profiling uses a per-call `tempfile.TemporaryDirectory`, so Modal-only workflows do not populate the user's local `.cache/datasets` tree.
- Modal now computes dataset normalization metadata when `normalization: "dataset"` or `use_dataset_normalization` is requested and applies the resulting mean/std in transforms.
- Modal supports `preserve_aspect_pad`, `center_crop`, random resized crop, ImageFolder train/val/test split, weighted sampler, weighted/focal/effective-number class-balanced loss, Adam/AdamW/SGD, cosine/step scheduler, early stopping, pretrained/frozen/last-block/full fine-tuning, bounded structured image augmentation policies (`basic`, `randaugment`, `trivialaugment`, `autoaugment`), and training-only mixed-sample policies (`mixup`, `cutmix`) through `augmentation_policy_config`.
- Modal supports bbox crop execution for `preprocessing.crop_strategy` of `bbox_crop_if_available` or `bbox_crop_ablation`, and `preprocessing.bbox_mode` of `crop_if_available` or `crop_and_compare_full_image`, when Pascal VOC XML or annotation JSON bounding boxes are available in the extracted dataset. `bbox_crop_ablation` / `crop_and_compare_full_image` also report a crop-vs-full-image heldout comparison in evaluation metadata. Missing requested bbox annotations fail locally with a clear worker error; backend validation remains the main scheduling gate.
- Modal and local evaluation payloads can include a report-only `label_quality_audit` object for `mechanism: label_noise_audit`, `mechanism: hard_example_audit`, or explicit `label_quality_audit` config. Dedicated worker jobs with template `label_quality_audit` also update the dataset profile with a report-only `label_quality_audit` / capped `label_quality_audits` artifact path. Both paths report suspicious/hard examples and never mutate labels or datasets.
- Worker helper tests cover transform construction, sampler/loss selection, profile artifact detection, normalization metadata, and pure export/demo payload shapes.
- Worker helper tests also cover export manifests, dependency-guarded demo inference, visual exemplar generation, split-file parsing, Pascal VOC XML, and annotation JSON.
- Modal model support includes MobileNetV3 small/large, EfficientNet B0/B1/B2, RegNet Y 400MF, ResNet18/34, ConvNeXt Tiny, Swin T, and ViT B/16 via torchvision.
- Local simulates expected quality/latency/cost effects for the same contract but does not train real models.

## Deferred Worker Work

Keep these as explicit future work, not accidental partial implementations:

- Wire parsed split files into real training behavior.
- Extend dataset normalization beyond the current bounded mean/std helper if future training paths need cached stats or explicit provenance.
- Wire explicit split-file training instead of always using the deterministic random split.
- Generate model-card/export metadata: input size, normalization, class labels, latency, model size, training config, and limitations.
- Add production storage upload for worker-local `file://` export/exemplar artifacts.
- Add real model reconstruction/export from completed training runs when no worker-visible artifact exists.

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
- `POST /jobs/:id/champion-export-result`: reports validated export job results to the backend.
- `POST /jobs/:id/champion-demo-prediction-result`: reports validated demo prediction results to the backend.
- `POST /datasets/:id/visual-exemplars`: persists capped exemplar/demo image metadata into canonical profile JSON.
- `POST /datasets/:id/profile`: also used by the local `label_quality_audit` job to persist additive report-only audit metadata into canonical profile JSON.
- `GET /projects/:id/champion/demo-predictions`: Mission Control reads durable prediction history.
- `POST /projects/:id/champion/demo-predictions`: backend creates audit rows and queues worker prediction jobs when a `READY` export exists.

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
