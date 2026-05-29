# Backend-Validation-Gated Experiment Methods

Date: 2026-05-27

This doc lists planner or visual-analysis methods that may appear in LLM proposals, but cannot be scheduled unless the backend can validate the required evidence and concrete runnable configuration.

## Source Of Truth

- Planner mechanism taxonomy: `services/orchestrator/internal/agents/experiment_planner_llm.go:851`
- Planner prompt rule that visual hypotheses are advisory only until backend validation: `services/orchestrator/internal/agents/experiment_planner_llm.go:884`
- Follow-up scheduling gate: `services/orchestrator/internal/api/handlers.go:1473`
- Stored decision mechanism contract: `services/orchestrator/internal/api/handlers.go:5804`
- Dataset/mechanism validation gate: `services/orchestrator/internal/api/handlers.go:5829`
- Visual hypothesis downgrade/sanitization for unsupported backend evidence: `services/orchestrator/internal/api/handlers.go:7351`
- Worker preprocessing metadata requirements: `services/worker/worker/training/preprocessing_registry.py:17`

## Important Distinction

`needs_backend_validation` on a visual-analysis card means the observation can be retained as advisory evidence. It does not mean the backend can schedule an experiment from that observation alone.

To schedule, the experiment planner must convert the observation into:

- a supported planner mechanism
- a concrete `PlannedExperiment`
- supported preprocessing/augmentation/class-balancing fields
- backend-verifiable evidence from the dataset profile, run metrics, accepted visual analysis, or deployment objective

## Gated Methods

| Method | Can Be Proposed As | Scheduling Requirement | Why It Is Blocked Without Backend Validation | Backend Support Needed |
| --- | --- | --- | --- | --- |
| `bbox_crop_ablation` | Planner mechanism; visual preprocessing hypothesis; `preprocessing.crop_strategy: bbox_crop_ablation`; bbox-aware resize/crop modes | Must include bbox crop preprocessing and the dataset profile must prove bbox/annotation evidence exists. | Visual comments about subject/background are not enough. The backend must know bounding boxes or compatible annotations exist before queueing a crop-based training run. | Dataset profile fields such as `bbox_available`, `annotations_available`, `bbox_annotations_count`, `bbox_count`, bbox-like `artifact_counts`, bbox-like artifacts, COCO/VOC annotation evidence, or equivalent. Worker must be able to load those boxes. |
| `resolution_crop` | Planner mechanism; crop/resize/image-size experiment | Must include a real image-size, resolution-strategy, or preprocessing change. If the proposal uses crop/high-resolution/aspect-ratio behavior, it also needs object-scale, fine-grained, dimension, crop, or accepted visual-trait evidence. | A generic “crop may help” note is not schedulable if no concrete config or evidence explains why crop/resolution should matter. | Dataset profile dimensions, scale/crop traits, accepted visual-analysis hypothesis/trait evidence, or run diagnosis tying resolution/crop to expected improvement. |
| `augmentation_auto` | Planner mechanism; `augmentation_policy` or `augmentation_policy_config` | Must use structured RandAugment, TrivialAugment, TrivialAugmentWide, or AutoAugment policy/config. | A text recommendation for “stronger augmentation” does not identify a runnable backend policy. | `augmentation_policy` set to one of the supported auto policies, or `augmentation_policy_config.policy_type` set to `randaugment`, `trivialaugment`, `trivialaugmentwide`, or `autoaugment` with valid numeric bounds. |
| `augmentation_mixed_sample` | Planner mechanism; MixUp/CutMix experiment | Must use structured MixUp or CutMix policy/config. | Visual evidence about similar classes/background variation is advisory only until the proposal names a runnable mixed-sample policy. | `augmentation_policy: mixup` or `cutmix`, or `augmentation_policy_config.policy_type: mixup` / `cutmix`, with valid `probability` and `alpha` ranges. |
| `class_imbalance` | Planner mechanism; class weighting/sampling experiment | Must configure `class_balancing` or `sampling_strategy`, and must have imbalance, minority-class, per-class-error, or macro-F1-vs-accuracy evidence. | The backend blocks class-balancing mechanisms if they are not tied to actual class distribution or per-class failure evidence. | Dataset profile `imbalance_ratio`, `class_distribution`, `images_per_class`, or training/evaluation evidence showing minority/per-class weakness. |
| `minority_targeting` | Planner mechanism; minority class sampling/loss experiment | Same gate as `class_imbalance`: class balancing or sampling plus minority/per-class evidence. | The proposal cannot target “minority” behavior unless the backend can verify minority/per-class support. | Same as `class_imbalance`, plus any per-class evaluation diagnostics from completed runs. |
| `label_noise_audit` | Planner mechanism | Must be scheduled as report-only `label_quality_audit`, not as a normal training experiment. | The backend blocks it if the planner tries to turn label audit into a training job or dataset mutation. | A concrete audit job template and report-only payload. No label mutation or training template. |
| `hard_example_audit` | Planner mechanism | Must be scheduled as report-only `label_quality_audit`, not as a normal training experiment. | Same as `label_noise_audit`: audit methods are not model-training experiments. | A concrete audit job template and report-only payload. |
| `deployment_latency` | Planner mechanism; compact/latency/cost experiment | Evidence must mention deployment constraints such as latency, runtime, cost, edge/live objective, compact/mobile model, or similar. | A latency-oriented method is blocked if nothing in the project objective, run metrics, or evidence shows latency/cost matters. | Project objective or run evidence with latency/runtime/cost/edge/live/mobile constraints. |
| `distillation` | Planner mechanism; future option | Currently always blocked. | The validator explicitly marks distillation as not schedulable until teacher-artifact validation and worker support exist. | Teacher model/artifact selection, artifact access/format validation, and worker-side distillation training support. |

## BBox-Specific Notes

The worker registry already recognizes bbox-aware modes, but marks them as requiring bounding-box metadata:

- `preprocessing.resize_strategy: bbox_crop_if_available`
- `preprocessing.crop_strategy: bbox_crop_if_available`
- `preprocessing.crop_strategy: bbox_crop_ablation`
- `preprocessing.bbox_mode: crop_if_available`
- `preprocessing.bbox_mode: crop_and_compare_full_image`
- `preprocessing.bbox_mode: use_boxes_as_metadata`

The orchestrator therefore blocks `bbox_crop_ablation` unless both conditions are true:

- the proposed experiment includes bbox crop preprocessing
- the backend profile shows bbox/annotation evidence

If these conditions are not met, the visual hypothesis should remain evidence-only.

## Methods That Are Usually Schedulable If Basic Shape Is Valid

These planner mechanisms are in the taxonomy and do not currently have extra dataset-evidence validation in `validateMechanismDatasetEvidence`:

- `baseline_control`
- `architecture_challenge`
- `capacity_finetune`
- `optimizer_scheduler`
- `regularization`
- `augmentation_basic`

They still must pass the generic experiment validator: supported model/template, valid numeric ranges, supported optimizer/scheduler/preprocessing values, non-empty mechanism contract, non-empty evidence, and novelty filtering.

## Practical Interpretation

The visual-analysis agent can propose hypotheses like “bbox-aware crop may help” or “MixUp may help similar classes.” The scheduler only accepts those once the experiment planner adds the missing backend-validating details.

In short:

- Visual hypothesis: useful observation.
- Planner proposal: concrete experiment design.
- Backend validation: proof that the exact design is supported and schedulable.
