# Data / Dataset Intelligence Agent Context

## Mission And Scope

This agent owns dataset understanding for Model Express image-classification workflows. Its job is to turn deterministic dataset facts into compact, validated context for planning, preprocessing, metric choice, and future dataset-analysis memory.

Stay focused on dataset/profile/preprocessing contracts. Do not let this agent directly create jobs, mutate execution state, start workers, or bypass backend validation. The system boundary is: agents reason and propose; the Go backend validates, stores, schedules, and executes; Python workers profile/train/report.

## Current Profile Shape

Current durable dataset profile data is stored as JSON in `datasets.profile` and exposed through `Dataset.Profile`. The worker profiler emits:

- identity/task: `schema_version`, `dataset_path`, `task_type`, `class_names`
- counts: `class_count`, `image_count`, `total_images`
- class distribution: `class_distribution`, legacy `images_per_class`
- imbalance/corruption: `imbalance_ratio`, `corrupt_file_count`, legacy `corrupt_image_count`, capped `corrupt_images`
- dimensions: legacy `width_min`, `width_max`, `height_min`, `height_max`, plus `image_dimension_stats.width|height|aspect_ratio.{min,max,mean,median}`
- dataset quality/context: `split_summary`, `metadata_summary`, `leakage_warnings`, `dataset_traits`
- deterministic visual traits: `visual_trait_summary` with object scale, object-area ratio, background dominance, blur likelihood, lighting variation, fine-grained possibility, and crop plausibility signals
- artifacts: capped `artifacts` list with `artifact_type`, `path`, `format`, optional `description`, optional `detected_schema`

The Go `DatasetProfile` struct mirrors this richer shape while preserving legacy keys used by existing planners.

## Preprocessing Schema

`PlannedExperiment` currently supports these dataset/preprocessing fields:

- `resolution_strategy`: `fixed`, `low_latency`, `compare_224_256`, `high_resolution_ablation`
- `preprocessing.resize_strategy`: `squash`, `preserve_aspect_pad`, `center_crop`, `random_resized_crop`, `bbox_crop_if_available`
- `preprocessing.normalization`: `imagenet`, `dataset`, `none`
- `preprocessing.crop_strategy`: `none`, `center_crop`, `random_resized_crop`, `bbox_crop_if_available`, `bbox_crop_ablation`
- `preprocessing.bbox_mode`: `ignore`, `crop_if_available`, `crop_and_compare_full_image`, `use_boxes_as_metadata`
- `preprocessing.use_dataset_normalization`: boolean
- `augmentation_policy`: `none`, `light`, `moderate`, `strong`, `custom`
- `augmentation` keys: `horizontal_flip`, `vertical_flip`, `color_jitter`, `random_crop`, `random_rotation`, `random_erasing`
- `class_balancing`: `none`, `weighted_loss`, `class_weighted_loss`, `class_balanced_sampler`, `weighted_random_sampler`, `focal_loss`
- `sampling_strategy`: `none`, `class_balanced_sampler`, `weighted_random_sampler`

Backend validation owns the allowed values. Workers currently implement ImageNet/no/dataset normalization, basic resize/crop transforms, structured augmentation policies, training-only MixUp/CutMix, weighted/focal/effective-number losses, weighted sampling, and bbox crop execution when annotations are available.

## Artifacts And Traits

Profiler-detected artifact types are:

- `image_root`
- `class_folder`
- `annotation_xml`
- `annotation_json`
- `labels_csv`
- `split_file`
- `metadata_folder`
- `bounding_boxes`
- `class_hierarchy`

Current profiler traits include:

- size/classes: `small_dataset`, `medium_dataset`, `many_classes`, `fine_grained_possible`
- distribution: `imbalanced`
- dimensions: `low_resolution`, `high_resolution`, `variable_image_dimensions`
- metadata/data quality: `bbox_available`, `metadata_available`, `corrupt_files_detected`
- visual/crop evidence: `small_objects_possible`, `background_dominant_possible`, `blur_possible`, `lighting_variation`, `crop_plausible`

Use these traits as planning evidence, not automatic prescriptions. For example, `bbox_available` should suggest a controlled crop ablation against full-image training, not a blanket replacement.

## Source Of Truth Caveat

`dataset_profiles` exists in the Postgres migration, but current store/API paths write and read full profile payloads through `datasets.profile` JSON. No active path writes normalized `dataset_profiles` rows yet.

Until that is wired, treat `datasets.profile` JSON as the source of truth. If adding persistence later, either fully wire `dataset_profiles` as the latest normalized profile path or clearly keep it deferred to avoid split-brain profile state.

Latest integration decision: `datasets.profile` remains authoritative for PR 1/8. `dataset_artifacts` is still deferred; profile `artifacts`, `demo_images`, and `visual_exemplars` arrays are the current compact JSON source for planner/demo surfaces. Worker helper modules can parse split/annotation metadata and generate capped visual exemplar packs, and the backend can persist accepted exemplar/demo image patches through `POST /datasets/:id/visual-exemplars`.

Worker `label_quality_audit` jobs also write report-only audit metadata into `datasets.profile` through `label_quality_audit`, capped `label_quality_audits`, and an additive `label_quality_audit` artifact record. These jobs use existing profile facts and do not mutate labels, files, or split assignments.

## Profiler And Planner Touchpoints

Worker profiling flow:

- Dataset registration queues a `profile_dataset` job.
- `services/worker/worker/jobs.py` downloads/extracts the dataset and calls `profile_image_folder`.
- `services/worker/worker/datasets/profiler.py` computes profile JSON and posts it to `/datasets/:id/profile`.
- Backend `UpdateDatasetProfile` marks the dataset `PROFILED` and stores JSON in `datasets.profile`.
- Initial planning requires a profiled dataset and reads legacy keys such as `total_images`, `class_count`, `imbalance_ratio`, and `corrupt_image_count`.

Backend planner insight flow:

- `datasetPlanningInsights` converts `Dataset.Profile` into compact `DatasetPlanningInsights`.
- It normalizes legacy/new count and corruption keys, carries dimension/artifact/split/metadata fields, and derives concise constraints plus recommended preprocessing, augmentations, metrics, and live-inference priorities.
- Experiment Planner prompt context includes both raw `dataset.profile` and `dataset_planning_insights`.
- Deterministic diagnosis uses `DatasetInsights.ImbalanceRatio` alongside run summaries/evaluations.
- Candidate ranking and duplicate signatures include preprocessing, augmentation policy, sampling strategy, and resolution strategy.

## Future Work

High-value follow-ups for this agent:

- Add first-class `dataset_artifacts` records if artifact history, file-level metadata, annotation parsing, or provenance becomes important.
- Annotation XML/JSON parsing is now used by profiling for visual crop/object-scale traits and by worker training for bounded bbox crop execution when preprocessing requests it. Split-file execution, labels CSV/class hierarchy parsing, and richer metadata folder parsing remain future work.
- Wire explicit train/validation/test split files into worker training and champion/demo image selection.
- Add production object-storage upload and durable history for generated visual exemplars beyond current profile JSON patches.
- Current PR 8 slice exposes budget-capped visual exemplar/demo image metadata from `datasets.profile`, accepts capped backend profile merges, and worker helpers can generate downscaled exemplars locally.
- Compute dataset-specific normalization statistics and apply them only when worker support makes `normalization: "dataset"` real.
- Keep bbox crop/full-image ablations controlled against a full-image heldout comparison and rely on backend validation as the main scheduling gate.
- Normalize dataset profiles into durable rows or document/defer the table path deliberately.

## Safe Boundaries And Shared Contracts

- Do not change dataset/profile/preprocessing fields without checking backend validation, planner prompt context, candidate signatures, worker config consumption, and Mission Control display.
- Keep source-of-truth writes deterministic. Agents may create `dataset_analysis` or `preprocessing_recommendation` memory in future, but profile facts should come from deterministic profiling or validated parsers.
- Keep profile payloads compact. Cap artifact lists and visual exemplar payloads by count, bytes, and prompt budget.
- Treat annotations, split files, and bbox data as backend-validated evidence. Bbox crop execution is now supported when annotations are present; split-file execution remains future work.
- Preserve legacy profile keys while older planners still read them.
- Shared contract owners are Dataset Intelligence, Backend/Planner, Worker Training, and Mission Control. Backend validation is the final execution gate.

## Supporting Docs To Read

- `docs/data_preprocessing_agent_report.md`
- `docs/model_express_agentic_upgrade_roadmap.md`
- `docs/llm_agent_decision_context.md`
- `docs/llm_decision_quality_report.md`
- `docs/orchestrator_system_design_report.md`
- `docs/agentic.md`
