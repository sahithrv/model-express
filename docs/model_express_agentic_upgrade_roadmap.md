# Model Express Agentic Upgrade Roadmap

Status: planning document only. This roadmap describes the next implementation sequence for making Model Express plan image-classification experiments like a senior data scientist instead of repeatedly trying random model families or longer epoch counts.

The core safety boundary remains unchanged:

```text
LLM proposes structured JSON -> backend validates -> backend stores/schedules -> workers execute
```

Do not let an LLM directly create jobs, run Modal work, mutate datasets, pick arbitrary files, bypass backend validation, or weaken duplicate and stop guards.

## Current Position

The system already has useful ingredients:

- `PlannedExperiment` supports model, epochs, batch size, learning rate, image size, `resolution_strategy`, `preprocessing`, optimizer, scheduler, weight decay, augmentation, `augmentation_policy`, class balancing, sampling, early stopping, pretrained/freeze flags, and `fine_tune_strategy`.
- The planner prompt accepts `candidate_hypotheses`, `planning_mode`, diagnosis evidence, rejected options, score components, and compact `planner_context_snapshot`.
- Deterministic diagnosis covers overfitting, underfitting, plateau, instability, class imbalance, minority-class failure, cost, latency, and improvement stagnation.
- Backend novelty checks reject exact repeats and minor-only same-mechanism follow-ups.
- Backend stop guards can select a champion when bounded metrics are near ceiling or repeated follow-ups fail to beat the champion.
- Modal/local worker paths already understand several execution knobs such as transforms, optimizer/scheduler, class weights, focal loss, weighted sampling, dataset normalization, and fine-tuning depth.

The important gap is not that image classification has too few real experiment options. The gap is that Model Express has not made the experiment action space explicit enough. The planner still sees a loose bag of fields, so it can collapse back to architecture shopping:

```text
Try EfficientNet.
Try ResNet.
Train longer.
```

The next upgrade should make every follow-up experiment pass through:

```text
diagnosis -> mechanism -> intervention -> controlled config -> backend validation -> outcome memory
```

## Codebase Findings

### Existing Planner Contract

Key files:

- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/diagnosis.go`
- `services/orchestrator/internal/agents/candidate_ranking.go`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/plans/model.go`
- `services/worker/worker/training/modal_app.py`

The planner already has broad fields, but there is no first-class `mechanism` field on `plans.PlannedExperiment`. A mechanism is inferred from config text/signatures. That is good enough for duplicate filtering, but not strong enough for an agentic search policy.

### Existing Backend Gates

Backend validation already knows allowed values for:

- optimizers: `adamw`, `adam`, `sgd`
- schedulers: `none`, `cosine`, `step`
- resolution strategies: `fixed`, `low_latency`, `compare_224_256`, `high_resolution_ablation`
- resize strategies: `squash`, `preserve_aspect_pad`, `center_crop`, `random_resized_crop`, `bbox_crop_if_available`
- normalization: `imagenet`, `dataset`, `none`
- crop strategies: `none`, `center_crop`, `random_resized_crop`, `bbox_crop_if_available`, `bbox_crop_ablation`
- bbox modes: `ignore`, `crop_if_available`, `crop_and_compare_full_image`, `use_boxes_as_metadata`
- augmentation policy: `none`, `light`, `moderate`, `strong`, `custom`
- augmentation keys: `horizontal_flip`, `vertical_flip`, `color_jitter`, `random_crop`, `random_rotation`, `random_erasing`
- class balancing: `none`, `weighted_loss`, `class_weighted_loss`, `class_balanced_sampler`, `weighted_random_sampler`, `focal_loss`
- sampling: `none`, `class_balanced_sampler`, `weighted_random_sampler`
- fine-tuning: `head_only`, `last_block`, `full`

These fields are enough to support more thoughtful experiments, but the planner should be forced to name the mechanism and the evidence that makes that mechanism appropriate.

### Current Risk

The planner can still produce superficially varied plans that are really the same search move:

- same model family with only epochs or learning-rate changes
- several architecture challengers without a diagnosis
- augmentation changes when no overfitting, blur, crop, or visual-variance signal exists
- class balancing when macro-F1 and per-class recall are already strong
- high-resolution challengers without object-scale or crop evidence
- more experiments when the champion is already near the metric ceiling
- a terminal champion selection followed by new autonomous follow-up experiments

Backend validation should keep these from becoming scheduled work.

### Newly Observed Bug: Champion Selection Is Not Terminal

A run-through showed the system selecting a champion and then later proposing more experiments. That is a control-plane bug, not a model-quality issue.

Expected behavior:

- Once a project has a persisted `SELECT_CHAMPION` decision or project champion, autonomous follow-up scheduling should stop.
- Existing stale `ADD_EXPERIMENTS` decisions should not create new follow-up plans after champion selection.
- The only way to continue after champion selection should be an explicit user action such as "start a new exploration round" or "reopen experimentation", with a new source decision/event.
- If an autonomous scheduler encounters a selected champion, it should create no plan/jobs and record a clear blocked event such as `backend_stop_guard: champion_selected_guard`.

Likely areas to inspect:

- `schedulePlannerDecision`
- `plannerFollowUpStopReason`
- `ensureFollowUpPlan`
- `experimentPlannerDecisionForPlan`
- `actionDecisionForPlan`
- `followUpSourceDecision`
- `followUpPlanForDecision`
- project champion persistence and lookup paths

## Research-Backed Experiment Levers

These are the levers worth exposing as first-class, diagnosis-driven mechanisms.

### 1. Capacity And Fine-Tuning

Use when there is underfitting, low training quality, or a strong sign that the frozen head cannot adapt.

Controls:

- model family or capacity tier
- `fine_tune_strategy`: `head_only`, `last_block`, `full`
- `freeze_backbone`
- optimizer and scheduler
- learning rate range by fine-tuning depth

Backend rule:

- More epochs alone is not a capacity mechanism unless the curve is still improving and the run is not near ceiling.

### 2. Regularization And Optimization

Use when there is overfitting, instability, noisy validation curves, or a small dataset.

Controls:

- weight decay
- dropout if supported by model head later
- label smoothing
- early stopping
- AdamW vs SGD
- cosine schedule vs step schedule
- Sharpness-Aware Minimization as a later advanced optimizer option

Research signal:

- Label smoothing is a known classification regularizer in Inception-style training.
- SAM targets flatter minima and can improve generalization, but should be deferred until the worker has focused tests because it changes optimizer behavior.

### 3. Augmentation Policy

Use when the dataset has small sample count, lighting variation, background variation, viewpoint variation, blur, or overfitting.

Controls:

- current safe policy: `none`, `light`, `moderate`, `strong`, `custom`
- structured policy later: `basic`, `randaugment`, `trivialaugment`, `autoaugment`
- transform knobs: flips, color jitter, rotation, random crop, random erasing
- policy strength/magnitude and probability caps

Research signal:

- RandAugment reduces the augmentation search space and is meant for practical automated augmentation.
- Torchvision supports AutoAugment, RandAugment, TrivialAugmentWide, MixUp, and CutMix in the transforms API.

### 4. Mixed-Sample Augmentation

Use when overfitting, small data, class boundary ambiguity, or calibration/generalization is the likely issue.

Controls:

- MixUp
- CutMix
- alpha/probability
- apply only to training batches
- avoid for tasks where localized class evidence makes mixing misleading

Backend rule:

- Treat MixUp/CutMix as distinct mechanisms from ordinary augmentation. They affect labels and batching, not just image transforms.

Research signal:

- MixUp trains on convex combinations of examples and labels.
- CutMix replaces image regions and mixes labels, preserving localizable visual evidence better than plain regional dropout.

### 5. Class Imbalance And Minority Failure

Use when imbalance ratio is high, accuracy exceeds macro-F1, or per-class recall/F1 has a weak minority class.

Controls:

- weighted cross entropy
- effective-number class-balanced loss
- focal loss
- weighted random sampler
- class-balanced sampler
- targeted augmentation for minority classes
- target metric switch toward macro-F1 or worst-class recall

Backend rule:

- Class-balancing mechanisms need actual imbalance or per-class weakness evidence. Do not schedule them when all per-class scores are already strong.

Research signal:

- Focal loss was introduced to address extreme class imbalance.
- Class-Balanced Loss reweights by effective number of samples instead of raw inverse frequency.

### 6. Resolution, Crop, And Object Scale

Use when visual evidence or dataset profile suggests variable image dimensions, small objects, aspect ratio distortion, background dominance, or object localization issues.

Controls:

- `image_size`
- `resolution_strategy`
- `preserve_aspect_pad` vs `squash`
- `center_crop` vs `random_resized_crop`
- bbox crop if annotations exist
- train/eval crop consistency
- high-resolution ablation only when expected gain justifies latency cost

Backend rule:

- Higher image size is not meaningful by itself. It must be tied to object scale, fine-grained classes, crop mismatch, or visual evidence.

Research signal:

- FixRes-style work highlights train/test resolution and crop discrepancies in image classification.

### 7. Label Noise And Hard-Example Review

Use when heldout/test metrics look suspicious, the confusion matrix has asymmetric class confusions, validation losses are unstable, or high-confidence mistakes cluster by class.

Controls:

- high-loss sample audit
- low-confidence correct sample audit
- high-confidence wrong sample audit
- label noise scorecard
- robust loss later, such as generalized cross entropy
- quarantine/report only at first; do not mutate labels automatically

Backend rule:

- Initial label-noise mechanisms should create an audit/review artifact, not automatically relabel or drop samples.

Research signal:

- Maximum softmax confidence is a simple baseline for detecting misclassified/OOD examples.
- Co-teaching and generalized cross entropy are research-backed robust training paths, but they are larger worker changes and should be later PRs.

### 8. Deployment And Compression

Use when a larger model beats the compact champion but violates latency/cost goals.

Controls:

- compact model challenger
- latency-constrained ranking
- distillation from a stronger teacher to a smaller student later
- final validation against heldout/test images

Backend rule:

- Quality-only improvements should not displace a champion if latency, model size, or runtime violate the project objective.

Research signal:

- Knowledge distillation is a standard way to transfer a larger model or ensemble into a smaller deployable model, but it is a later execution-plane feature.

## Target Mechanism Taxonomy

Add a first-class mechanism vocabulary before adding more model names.

Recommended mechanisms:

- `stop_select_champion`
- `baseline_control`
- `architecture_challenge`
- `capacity_finetune`
- `optimizer_scheduler`
- `regularization`
- `augmentation_basic`
- `augmentation_auto`
- `augmentation_mixed_sample`
- `class_imbalance`
- `minority_targeting`
- `resolution_crop`
- `bbox_crop_ablation`
- `label_noise_audit`
- `hard_example_audit`
- `deployment_latency`
- `distillation`

Each mechanism should have:

- allowed diagnosis triggers
- required evidence fields
- allowed config knobs
- disallowed cosmetic changes
- expected metric target
- expected tradeoff
- backend novelty signature
- worker support status
- UI display label

Example:

```json
{
  "mechanism": "resolution_crop",
  "hypothesis": "Variable image dimensions and object scale are causing avoidable validation errors.",
  "evidence_used": [
    "dataset_card.image_dimension_stats shows high width/height variance",
    "visual_evidence reports background-dominant examples"
  ],
  "intervention": "random_resized_crop with ImageNet normalization at 256px",
  "expected_effect": "Improve robustness to object scale and crop variation",
  "expected_tradeoffs": ["slower inference", "may reduce accuracy if crops remove key context"]
}
```

## Data The LLM Should Receive

The planner should not receive raw table dumps. It should receive compact, decision-ready cards.

### Dataset Decision Card

Include:

- class count and images per class summary
- imbalance ratio and minority classes
- image dimension and aspect-ratio summary
- corrupt image count and leakage/split warnings
- detected artifacts: bbox annotations, split files, labels CSV, metadata, class hierarchy
- visual exemplar traits: object scale, background dominance, blur, lighting, fine-grained classes
- recommended metrics and preprocessing

Exclude:

- raw file lists
- full profile JSON
- all visual exemplar metadata
- full prior memory blobs

### Training Dynamics Card

Include:

- best/final metric
- train/validation loss gap
- metric slope over last N epochs
- plateau and instability scores
- whether more epochs are justified
- early stopping recommendation

### Per-Class Error Card

Include:

- worst classes by recall/F1
- top confusion pairs
- accuracy vs macro-F1 gap
- whether imbalance or minority failure is active

### Deployment Card

Include:

- latency, throughput, parameter count, model size
- objective weights
- compact champion vs quality challenger tradeoff
- whether the proposed experiment can realistically beat the champion after cost/latency penalties

### Mechanism Coverage Card

Include:

- tried mechanisms
- blocked mechanisms
- mechanisms still eligible
- best result by mechanism
- failed mechanism lessons
- shallow repeat warnings

This card is the key to avoiding repeated ResNet/EfficientNet loops.

## Recommended PR Sequence

### PR 1: Mechanism Taxonomy And Contract

Goal: make the experiment action space explicit without changing worker behavior.

Primary files:

- `services/orchestrator/internal/plans/model.go`
- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/candidate_ranking.go`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/api/handlers_test.go`
- `docs/agents_context/llm_decision_intelligence/context.md`
- `docs/agents_context/orchestrator_backend/context.md`

Implementation slices:

1. Add first-class mechanism fields to planner-facing structures.
   - Preferred: `mechanism` and `intervention` on `CandidateHypothesis`.
   - Consider adding `Mechanism string` to `plans.PlannedExperiment` only if it is persisted and useful to workers/UI. If not, store mechanism in the decision payload and derive execution config from existing fields.

2. Add backend allowed-mechanism validation.
   - Reject unknown mechanisms.
   - Reject `ADD_EXPERIMENTS` if every candidate lacks a mechanism.
   - Require mechanism-specific evidence for high-cost or high-risk proposals.

3. Update candidate ranking.
   - Score mechanism novelty separately from model novelty.
   - Penalize model changes that do not change mechanism.
   - Penalize mechanisms that conflict with diagnosis.

4. Update planner prompt.
   - Require `diagnosis -> mechanism -> intervention -> expected_effect`.
   - Make model family a parameter inside a mechanism, not the mechanism itself.

Tests:

- Planner output with no mechanism is rejected.
- Same model with more epochs cannot pass as a mechanism unless training-dynamics evidence says undertrained.
- Architecture-only challengers are penalized when no diagnosis supports them.
- Candidate ranking rewards a diagnosis-matched non-model mechanism.

### PR 2: Mechanism Coverage And Final Scheduling Gates

Goal: prevent stale or shallow decisions from scheduling work even if they came from older stored payloads.

Primary files:

- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/api/handlers_test.go`
- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/candidate_ranking.go`

Implementation slices:

1. Build a `mechanism_coverage` card from all prior project plans.
   - attempted mechanisms
   - best score by mechanism
   - no-improvement mechanisms
   - blocked repeats
   - eligible next mechanisms

2. Add final validation in follow-up plan creation.
   - Revalidate stale `ADD_EXPERIMENTS` decisions against current mechanism coverage.
   - Block same-mechanism minor-only repeats across all prior project plans.
   - Block high-cost mechanisms when near-ceiling stop guard is active.
   - Block all autonomous follow-up creation after a champion has already been selected for the project unless there is an explicit user-initiated reopen/new-exploration action.

3. Add blocked decision events.
   - If all experiments are filtered out, create no plan/jobs.
   - Record a clear backend event with `backend_validation_status: blocked`.
   - For post-champion follow-up attempts, record `backend_stop_guard: champion_selected_guard`.

Tests:

- Stale architecture-repeat decision cannot create a plan.
- All-filtered proposal creates no plan/jobs.
- Repeated `resnet18` or `efficientnet_b1` with only epoch/LR changes is blocked.
- After a `SELECT_CHAMPION` decision or persisted project champion exists, stale `ADD_EXPERIMENTS` decisions cannot create a follow-up plan.
- After champion selection, autonomous scheduler paths create no jobs and record a blocked event.
- Explicit user-initiated reopening/new-exploration behavior is the only accepted way to continue experimentation after champion selection.
- Valid diagnosis-matched new mechanism still schedules in dry-run/local tests.

### PR 3: Planner Context Cards V2

Goal: give the LLM less context but better context.

Primary files:

- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/experiment_planner_llm_test.go`
- `services/orchestrator/internal/agents/training_monitor_llm.go`
- `services/orchestrator/internal/agents/training_monitor_llm_test.go`
- `docs/agents_context/llm_decision_intelligence/context.md`

Implementation slices:

1. Add decision-ready cards to `planner_context_snapshot`.
   - `training_dynamics_card`
   - `per_class_error_card`
   - `deployment_card`
   - `mechanism_coverage_card`
   - `label_quality_card`

2. Compact Training Monitor input.
   - Replace raw-ish context with a compact run-evaluation card.
   - Keep full run data in backend storage.
   - Store the compact monitor prompt context in `agent_invocations.input_context`.

3. Add prompt budget telemetry.
   - Record approximate input sizes for each invocation.
   - Do not add vector DB yet.

Tests:

- Planner context excludes raw history and includes mechanism coverage.
- Training Monitor context excludes full epoch/profile dumps while preserving trend and per-class summary.
- Snapshot includes worst-class and top-confusion facts when available.

### PR 4: Advanced Augmentation Contract

Goal: expand beyond `light|moderate|strong` without turning augmentation into free-form JSON.

Primary files:

- `services/orchestrator/internal/plans/model.go`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/api/handlers_test.go`
- `services/worker/worker/training/modal_app.py`
- worker tests under `services/worker/tests`
- `docs/agents_context/python_worker_training/context.md`

Implementation slices:

1. Add structured augmentation object support.
   - `policy_type`: `basic`, `randaugment`, `trivialaugment`, `autoaugment`, `mixup`, `cutmix`
   - `magnitude`, `probability`, `alpha`, and safe caps by policy type

2. Keep policy support explicit.
   - No arbitrary transform names.
   - No unbounded transform strengths.
   - No worker execution unless backend validation accepts the policy.

3. Implement one safe vertical slice first.
   - Recommended first slice: RandAugment or TrivialAugmentWide for image transforms.
   - Recommended second slice: MixUp/CutMix because they require batch/label handling.

Tests:

- Backend rejects unknown policy types and unsafe parameters.
- Worker transform builder creates expected transform objects.
- MixUp/CutMix are applied only in training and only with compatible labels.
- No Modal jobs are run in tests.

### PR 5: Class Imbalance, Minority Targeting, And Label Quality

Goal: make class-specific failures actionable.

Primary files:

- `services/orchestrator/internal/agents/diagnosis.go`
- `services/orchestrator/internal/agents/candidate_ranking.go`
- `services/orchestrator/internal/api/handlers.go`
- `services/worker/worker/training/modal_app.py`
- `services/worker/worker/datasets/profiler.py`
- worker/backend tests

Implementation slices:

1. Add richer per-class diagnosis.
   - worst recall/F1 classes
   - confusion pairs
   - accuracy vs macro-F1 gap
   - minority class support

2. Add mechanism-specific controls.
   - `class_imbalance`
   - `minority_targeting`
   - `label_noise_audit`
   - `hard_example_audit`

3. Add label-quality audit artifact path.
   - First version should report suspicious samples only.
   - Do not relabel, delete, or resample files automatically.

4. Add effective-number class-balanced loss as a later slice if focal/weighted loss is not enough.

Tests:

- Class-balancing proposals require imbalance or minority-failure evidence.
- Label-noise audit proposals do not create training jobs unless explicitly represented as an audit job type.
- Backend rejects class-balancing repeats after a no-improvement outcome.

### PR 6: Resolution, Crop, And Visual Evidence Mechanisms

Goal: let the planner use image structure, not just scalar metrics.

Primary files:

- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/api/handlers.go`
- `services/worker/worker/datasets/profiler.py`
- `services/worker/worker/datasets/annotations.py`
- `services/worker/worker/training/modal_app.py`
- `services/worker/worker/datasets/exemplars.py`
- backend/worker tests

Implementation slices:

1. Add visual trait summary to planner context.
   - object scale
   - background dominance
   - lighting variation
   - blur
   - fine-grained classes
   - bbox/crop plausibility

2. Add bbox/crop ablation only when annotations exist.
   - full image vs crop
   - crop_if_available
   - crop_and_compare_full_image

3. Add resolution/crop guardrails.
   - Higher image size requires visual or per-class evidence.
   - High-resolution candidates must declare latency tradeoff.

Tests:

- Bbox mechanism is rejected when no bbox artifact exists.
- High-resolution mechanism is rejected without object-scale/crop evidence.
- Visual evidence appears only as evidence and never as execution authority.

### PR 7: Outcome Learning By Mechanism

Goal: teach future planner calls which mechanisms worked, not just which model names won.

Primary files:

- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/candidate_ranking.go`
- `services/orchestrator/internal/store/*`
- `services/orchestrator/internal/api/handlers.go`
- `docs/agents_context/llm_decision_intelligence/context.md`

Implementation slices:

1. Extend strategy scorecards with mechanism fields.
   - mechanism
   - intervention
   - diagnosis triggers
   - expected vs actual delta
   - cost and runtime
   - outcome status

2. Retrieve mechanism lessons into compact context.
   - Successful mechanism on similar dataset traits: bonus.
   - Failed/no-improvement mechanism on similar traits: penalty.
   - Rejected mechanism: blocked repeat warning.

3. Make candidate ranking mechanism-aware.
   - Penalize repeating failed mechanisms.
   - Do not penalize a mechanism when the new intervention is materially different and evidence-backed.

Tests:

- Failed `architecture_challenge` scorecard penalizes another architecture-only challenger.
- Successful `resolution_crop` scorecard boosts a similar dataset with variable dimensions.
- Strategy memory remains compact and does not dump raw decision payloads.

### PR 8: Mission Control Mechanism Visibility

Goal: make agent behavior inspectable before spending credits.

Primary files:

- `apps/mission-control/src/App.tsx`
- `apps/mission-control/src/types.ts`
- `apps/mission-control/src/styles.css`
- `services/orchestrator/internal/api/handlers.go`

Implementation slices:

1. Add mechanism coverage UI.
   - tried
   - eligible
   - blocked
   - best result
   - no-improvement outcomes

2. Add planner decision card.
   - diagnosis
   - selected mechanism
   - intervention
   - expected effect
   - rejected alternatives
   - backend validation status

3. Add dry-run review state.
   - Show proposed plan without running workers.
   - Make it obvious when auto scheduling/execution are disabled.

Checks:

- `npm run build`
- UI smoke with auto execution disabled.

### PR 9: Model Routing, Prompt Caching, And Retrieval Later

Goal: reduce token cost and improve reasoning only after the planner has a clean decision contract.

Do later, not first:

- route high-value experiment planning to a stronger model only when needed
- keep Training Monitor and JSON repair on cheaper models
- add prompt token/cached-token telemetry
- stabilize static prompt prefixes for provider-side prompt caching
- consider pgvector for cross-project similarity after mechanism outcome memory is clean

Reason:

- A stronger model will still repeat weak experiments if the backend action space is vague.
- Vector retrieval is only useful after the stored memories have high-quality mechanism/outcome labels.
- Prompt caching helps cost, but it does not fix bad experiment semantics.

## Acceptance Criteria

The system is ready to re-enable autonomous Modal execution only when all of these are true:

- Every `ADD_EXPERIMENTS` decision names a first-class mechanism and evidence-backed intervention.
- Backend rejects unknown, unsupported, stale, or diagnosis-mismatched mechanisms.
- Same-family and same-mechanism minor-only repeats are blocked across all project plans.
- Near-ceiling champions become `SELECT_CHAMPION`, not more training.
- Persisted champion selection is terminal for autonomous follow-up scheduling unless the user explicitly reopens experimentation.
- If all candidates are filtered out, no plan and no jobs are created.
- Planner context remains compact and decision-ready.
- Mission Control shows the mechanism, evidence, expected effect, backend validation result, and rejected alternatives.
- Regression tests prove stale `ADD_EXPERIMENTS` payloads cannot bypass current validation.
- No Modal jobs are run in automated tests.

## Verification Before Re-Enabling Modal

Use these settings while testing planner logic:

```env
MODEL_EXPRESS_AUTO_EXECUTE_PLANS=false
MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS=false
```

Recommended verification loop:

1. Run backend unit tests from `services/orchestrator`.
2. Use in-memory/store fixtures to create prior plans with repeated mechanisms.
3. Inspect `agent_invocations.input_context` and confirm the LLM sees compact cards, not raw history.
4. Inspect `agent_decisions.payload` and confirm mechanism, evidence, intervention, expected effect, rejected options, and backend validation fields are present.
5. Confirm blocked proposals create no jobs and record a clear event.
6. Confirm that a persisted champion blocks stale or new autonomous `ADD_EXPERIMENTS` scheduling.
7. Review Mission Control in dry-run mode.
8. Restart the dev server before testing against the UI so compiled code paths are current.
9. Only then re-enable auto scheduling/execution for a low-cost controlled dataset.

## Research References

- Torchvision transforms v2 docs: https://docs.pytorch.org/vision/stable/transforms.html
- Torchvision CutMix/MixUp docs: https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.CutMix.html
- RandAugment: https://arxiv.org/abs/1909.13719
- MixUp: https://arxiv.org/abs/1710.09412
- CutMix: https://arxiv.org/abs/1905.04899
- Focal Loss: https://arxiv.org/abs/1708.02002
- Class-Balanced Loss: https://arxiv.org/abs/1901.05555
- Label smoothing in Inception-style classification training: https://arxiv.org/abs/1512.00567
- Bag of Tricks for Image Classification: https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_With_Convolutional_Neural_Networks_CVPR_2019_paper.pdf
- Sharpness-Aware Minimization: https://arxiv.org/abs/2010.01412
- Fixing the Train-Test Resolution Discrepancy: https://arxiv.org/abs/1906.06423
- Maximum softmax confidence baseline for misclassification/OOD detection: https://arxiv.org/abs/1610.02136
- Co-teaching for noisy labels: https://arxiv.org/abs/1804.06872
- Generalized Cross Entropy for noisy labels: https://arxiv.org/abs/1805.07836
- Knowledge distillation: https://arxiv.org/abs/1503.02531
