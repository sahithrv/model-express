# Data Preprocessing / Dataset Intelligence Report

## Implemented

- Added first-class dataset profile structs in `services/orchestrator/internal/datasets/model.go` for:
  - `class_count`, `image_count`, `class_distribution`, `imbalance_ratio`
  - `image_dimension_stats`, `corrupt_file_count`, `split_summary`
  - `metadata_summary`, `leakage_warnings`, `dataset_traits`
  - dataset artifacts such as `annotation_xml`, `annotation_json`, `labels_csv`, `split_file`, `metadata_folder`, `bounding_boxes`, and `class_hierarchy`
- Expanded the worker dataset profiler to emit the richer profile shape while preserving legacy keys such as `total_images`, `images_per_class`, and `corrupt_image_count`.
- Added optional structured experiment fields:
  - `resolution_strategy`
  - `preprocessing`
  - `augmentation_policy`
  - `sampling_strategy`
- Added deterministic backend validation for the new fields, plus validation for optimizer, scheduler, class balancing, sampling, and allowed augmentation keys.
- Passed structured preprocessing fields through experiment execution config so validated plans become worker-visible job config.
- Added worker-side support for:
  - `augmentation_policy`: `light`, `moderate`, `strong`
  - resize strategies: `squash`, `preserve_aspect_pad`, `center_crop`, `random_resized_crop`
  - normalization: `imagenet`, `none`, and bounded dataset-computed mean/std when `normalization: "dataset"` or `use_dataset_normalization` is requested
  - class balancing: `weighted_loss`, `class_weighted_loss`, `focal_loss`
  - sampling: `class_balanced_sampler`, `weighted_random_sampler`
- Kept `datasets.profile` JSON as the active source of truth. The existing `dataset_profiles` table remains deferred to avoid split-brain profile reads/writes.
- Added safe visual exemplar/demo-image API support from capped `datasets.profile.visual_exemplars` and `datasets.profile.demo_images` metadata, plus backend-capped profile merge for worker-generated exemplar packs through `POST /datasets/:id/visual-exemplars`; generic `dataset_artifacts` persistence remains deferred.
- Added focused backend tests for structured preprocessing validation and unsupported augmentation rejection.

## Supported Experiment Options

```json
{
  "resolution_strategy": "fixed|low_latency|compare_224_256|high_resolution_ablation",
  "preprocessing": {
    "resize_strategy": "squash|preserve_aspect_pad|center_crop|random_resized_crop|bbox_crop_if_available",
    "normalization": "imagenet|dataset|none",
    "crop_strategy": "none|center_crop|random_resized_crop|bbox_crop_if_available|bbox_crop_ablation",
    "bbox_mode": "ignore|crop_if_available|crop_and_compare_full_image|use_boxes_as_metadata",
    "use_dataset_normalization": false
  },
  "augmentation_policy": "none|light|moderate|strong|custom",
  "augmentation": {
    "horizontal_flip": true,
    "color_jitter": true,
    "random_crop": true,
    "random_rotation": true,
    "random_erasing": true
  },
  "class_balancing": "none|weighted_loss|class_weighted_loss|class_balanced_sampler|weighted_random_sampler|focal_loss",
  "sampling_strategy": "none|class_balanced_sampler|weighted_random_sampler"
}
```

## Planner Guidance

The Experiment Planner should use these options as hypotheses, not as automatic prescriptions.

- If `dataset_traits` includes `imbalanced`, prefer `macro_f1`, inspect per-class metrics, and test `weighted_loss`, `focal_loss`, or `class_balanced_sampler` as controlled ablations.
- If `dataset_traits` includes `variable_image_dimensions`, compare `preserve_aspect_pad` or `random_resized_crop` against the default square resize.
- If `metadata_summary.bbox_available` or `dataset_traits` includes `bbox_available`, propose `bbox_crop_if_available` only as an ablation against full-image training.
- If the project objective is low-latency live inference, use `resolution_strategy: "low_latency"` or `fixed` image sizes before trying higher resolution.
- If the dataset is small, prefer `augmentation_policy: "moderate"` plus early stopping before larger models.

## Example Configs

Class imbalance ablation:

```json
{
  "template": "efficientnet_transfer",
  "model": "efficientnet_b0",
  "epochs": 12,
  "batch_size": 16,
  "learning_rate": 0.0003,
  "reason": "Test whether class-balanced training improves minority recall.",
  "image_size": 224,
  "resolution_strategy": "fixed",
  "optimizer": "adamw",
  "scheduler": "cosine",
  "weight_decay": 0.01,
  "augmentation_policy": "moderate",
  "class_balancing": "weighted_loss",
  "sampling_strategy": "none",
  "early_stopping_patience": 4,
  "pretrained": true,
  "freeze_backbone": true,
  "fine_tune_strategy": "head_only"
}
```

BBox/crop ablation spec:

```json
{
  "template": "mobilenet_transfer",
  "model": "mobilenet_v3_large",
  "epochs": 10,
  "batch_size": 16,
  "learning_rate": 0.0002,
  "reason": "Compare bbox-aware cropping against full-image training when annotations are available.",
  "image_size": 224,
  "resolution_strategy": "fixed",
  "preprocessing": {
    "resize_strategy": "bbox_crop_if_available",
    "normalization": "imagenet",
    "crop_strategy": "bbox_crop_ablation",
    "bbox_mode": "crop_and_compare_full_image"
  },
  "augmentation_policy": "light",
  "class_balancing": "none",
  "sampling_strategy": "none",
  "early_stopping_patience": 3,
  "strategy": "bbox crop ablation against full image context"
}
```

## Intentionally Deferred

- No new database table for `dataset_artifacts`; artifacts and exemplar metadata are currently represented in profile JSON and Go structs.
- No full annotation parser or bbox crop implementation yet. The schema supports bbox/crop hypotheses, and the profiler detects likely bbox XML, but crop execution should come in a later dedicated PR.
- Dataset normalization currently computes bounded image mean/std in the worker. Cached/provenanced normalization rows are still deferred.
- Split-file execution is not wired yet. The profiler detects split files, but training still uses the existing deterministic split path.
- Real downscaled exemplar generation and persisted invocation audit fields are still deferred.
