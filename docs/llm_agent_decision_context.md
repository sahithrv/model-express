# LLM Agent Decision Context

This document summarizes how Model Express currently uses LLM agents, what memory and database records they read/write, how LLM outputs become backend decisions or experiments, and where future schemas, algorithms, and prompts can make the system smarter.

The goal is to give another planning conversation enough context to propose better decision-making strategies without needing to inspect the codebase directly.

## Current System Shape

Model Express is an agentic image-classification experiment orchestrator with these major parts:

- Electron / Mission Control UI for project setup, execution, metrics, agent decisions, memory, and champions.
- Go orchestrator for projects, datasets, experiment plans, jobs, workers, agent decisions, and automation.
- Python worker for local / Modal training execution.
- Postgres for durable metadata, metrics, decisions, invocations, and memory.
- MinIO for uploaded dataset objects.
- Optional LLM calls for specialized agent reasoning.

The system is intentionally hybrid:

- Deterministic backend logic owns validation, scheduling, worker orchestration, and safety.
- LLM agents propose structured reasoning and experiment strategy.
- The backend validates every LLM output before storing decisions or scheduling jobs.

## Runtime LLM Configuration

LLM behavior is controlled by environment variables and automation settings.

Relevant environment / setting concepts:

```text
MODEL_EXPRESS_LLM_ENABLED=true
MODEL_EXPRESS_LLM_PROVIDER=openai
MODEL_EXPRESS_LLM_MODEL=<model name>
MODEL_EXPRESS_LLM_BASE_URL=https://api.openai.com/v1
MODEL_EXPRESS_LLM_API_KEY=<secret>
MODEL_EXPRESS_AGENT_MODE=propose | autonomous
MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS=true
MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS=true
MODEL_EXPRESS_AUTO_EXECUTE_PLANS=true
MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS=2
```

Supported provider styles are:

- `openai`
- `openai_compatible`
- `local`

The current client uses an OpenAI-compatible Chat Completions request shape:

```text
POST {base_url}/chat/completions
```

The LLM is expected to return JSON text. The backend parses that JSON into typed Go structs and validates it.

## Agent Modes

There are two important modes:

### Propose Mode

The LLM may recommend actions and experiments, but the backend does not automatically execute follow-up plans.

This is useful when testing prompts and agent behavior.

### Autonomous Mode

The LLM may recommend follow-up experiments, and if automation settings allow it, the backend can create and execute a new follow-up plan automatically.

Even in autonomous mode, the LLM does not directly enqueue jobs or start workers. It only provides structured recommendations. The backend remains the gatekeeper.

## Current LLM Agents

There are currently two LLM-backed specialized agents.

## 1. Training Monitor / Evaluation Agent

Agent name:

```text
training_monitor
```

Purpose:

The Training Monitor evaluates one completed or failed training job. It analyzes the run quality, training dynamics, cost, risks, and whether that individual run looks useful.

It does not create new experiment batches. That is intentionally reserved for the Experiment Planner.

Main inputs:

- Source experiment plan.
- Finished job config.
- Training run summary.
- Rich training evaluation, if available.
- Epoch metrics.
- Project objective context.
- Related memory records.

The agent sees information such as:

- Accuracy.
- Macro-F1.
- Per-class metrics.
- Confusion matrix.
- Train / validation gap.
- Plateau or instability signals.
- Cost.
- Runtime.
- Model size / latency profile if available.
- User goal context, such as live inference or cost sensitivity.
- Prior memory from the same project / dataset.

Expected output shape:

```json
{
  "summary": "Short explanation of what happened.",
  "recommended_action": {
    "type": "RANK_MODELS",
    "confidence": 0.82,
    "rationale": "Why this run matters."
  },
  "quality_summary": "How good the run was.",
  "training_dynamics": "Overfitting, underfitting, plateau, etc.",
  "cost_summary": "Cost and runtime interpretation.",
  "risks": ["Possible issue"],
  "findings": ["Useful observation"],
  "rank_score": 0.71,
  "tags": ["mobilenet_v3_small", "stable", "low_latency"]
}
```

Allowed action types for this agent:

```text
CHANGE_PREPROCESSING
PRUNE_RUN
RANK_MODELS
STOP_PROJECT
```

Validation rules:

- `summary` is required.
- `recommended_action.type` must be one of the allowed action types.
- `confidence` must be between `0` and `1`.
- `rationale` is required.
- `rank_score` must be between `0` and `1`.
- Missing list fields are normalized to empty arrays.

What gets stored:

- A full `agent_invocations` row.
- A distilled `agent_memory_records` row with kind `training_evaluation`.

How this helps:

This creates run-level memory. Later planning agents can see which models trained stably, which plateaued, which overfit, and which were low-value.

## 2. Experiment Planner Agent

Agent name:

```text
experiment_planner
```

Purpose:

The Experiment Planner reviews a completed experiment plan as a batch. It decides whether to:

- Add more experiments.
- Select a champion.
- Stop the project.
- Wait.

Unlike the Training Monitor, this agent can propose new experiments.

Main inputs:

- Project and dataset.
- Source experiment plan.
- All jobs in the source plan.
- Training summaries for those jobs.
- Rich evaluations for those jobs.
- Epoch metrics grouped by job.
- Dataset planning insights.
- User goal / objective context.
- Supported model catalog.
- Current champion.
- Champion before the source plan started.
- Deltas from baseline champion.
- Number of no-improvement rounds.
- Stop signals.
- Successful prior strategy memory.
- Failed prior strategy memory.
- Prior plans / jobs / evaluations.
- Prior memory.
- Existing experiment signatures.
- Maximum experiment count.
- Maximum follow-up rounds.
- Current follow-up round.

Expected output shape:

```json
{
  "summary": "What the planner believes should happen next.",
  "decision_type": "ADD_EXPERIMENTS",
  "rationale": "Evidence-based explanation.",
  "confidence": 0.78,
  "planning_mode": "champion_challenge",
  "hypothesis": "Why these experiments may beat the current champion.",
  "dataset_preprocessing_rationale": "How dataset traits influence preprocessing.",
  "changed_variables": ["model", "image_size", "fine_tune_strategy", "augmentation"],
  "success_criteria": "Improve macro-F1 by at least 0.03 without increasing latency too much.",
  "deployment_tradeoff": "Balances validation quality with live inference speed.",
  "proposed_experiments": [
    {
      "template": "image_classification",
      "model": "efficientnet_b0",
      "epochs": 18,
      "batch_size": 32,
      "learning_rate": 0.0003,
      "image_size": 224,
      "optimizer": "adamw",
      "scheduler": "cosine",
      "weight_decay": 0.0001,
      "augmentation": "moderate",
      "class_balancing": "weighted_loss",
      "early_stopping_patience": 5,
      "strategy": "challenge current low-latency champion with better balanced accuracy",
      "pretrained": true,
      "freeze_backbone": false,
      "fine_tune_strategy": "last_block",
      "reason": "Tests whether a slightly larger but still practical model improves per-class performance."
    }
  ],
  "champion_job_id": "",
  "why_can_beat_champion": "The current champion is fast but appears plateaued.",
  "expected_delta_vs_champion": 0.03,
  "stop_reason": "",
  "risks": ["May increase inference latency."],
  "expected_tradeoffs": ["Higher quality, slightly higher cost."],
  "novelty_notes": "Changes model family and preprocessing, not just epochs.",
  "tags": ["champion_challenge", "preprocessing", "macro_f1"]
}
```

Allowed decision types:

```text
ADD_EXPERIMENTS
SELECT_CHAMPION
STOP_PROJECT
WAIT
```

Important prompt rules:

- For `ADD_EXPERIMENTS`, return `1` to `5` complete experiments.
- Use only model names from the supported model catalog.
- Do not repeat existing experiment signatures.
- Do not propose only tiny changes like more epochs or a slightly different learning rate.
- Prefer meaningful changes involving model family, image size, optimizer, scheduler, fine-tuning strategy, augmentation, class balancing, or deployment tradeoffs.
- Use successful and failed memory.
- Use the project goal.
- Use dataset planning insights.
- Explain why the proposed batch can beat the current champion.

Validation rules:

- `summary`, `rationale`, and `confidence` are required.
- `confidence` must be between `0` and `1`.
- For `ADD_EXPERIMENTS`, proposed experiments are required.
- For `ADD_EXPERIMENTS`, the planner must provide:
  - `planning_mode`
  - `hypothesis`
  - `dataset_preprocessing_rationale`
  - `success_criteria`
  - `deployment_tradeoff`
  - `why_can_beat_champion`
- `changed_variables` must include at least two meaningful variables.
- A batch that only changes minor variables such as `epochs`, `learning_rate`, or `batch_size` is rejected.
- A batch with three or more experiments all using the same model is rejected.
- Every proposed experiment must pass backend validation.

What gets stored:

- A full `agent_invocations` row.
- An `agent_decisions` row if the decision is actionable.
- A distilled `agent_memory_records` row with kind `planning_feedback`.
- Later, once a follow-up plan finishes, a `planning_outcome` memory record is created.

How this helps:

The planner is the main bridge from LLM reasoning to new experiments. It looks at plan-level results and proposes the next batch, but the backend validates those proposals before scheduling.

## Deterministic Reviewer Fallback

There is also a deterministic reviewer.

It can return the same high-level decisions:

```text
WAIT
SELECT_CHAMPION
ADD_EXPERIMENTS
STOP_PROJECT
```

This fallback exists so the system can still make progress without LLM calls, or if an LLM invocation fails validation.

The deterministic reviewer is less creative. It mostly follows rule-based improvements and guardrails.

## End-to-End Agent Flow

The current flow looks like this:

```text
1. User creates project and uploads dataset.
2. Backend creates an experiment plan.
3. Plan execution enqueues experiment jobs.
4. Workers run training jobs locally or through Modal.
5. Workers report epoch metrics.
6. Workers report final training summary.
7. Workers optionally report rich evaluation data.
8. Job completes or fails.
9. Training Monitor LLM evaluates that individual job.
10. Backend stores training-monitor memory.
11. Backend checks whether the source plan is fully complete.
12. If the whole plan is complete, Experiment Planner LLM reviews the plan.
13. Backend validates planner output.
14. Backend stores planner decision and memory.
15. If autonomous mode and scheduling are enabled, backend creates a follow-up plan.
16. If auto-execution is enabled, backend enqueues and executes follow-up jobs.
17. Once the follow-up plan finishes, backend records planning outcome memory.
18. Future planner calls use that outcome memory.
```

The key design principle:

```text
Training Monitor reasons about one run.
Experiment Planner reasons about a completed batch.
The backend validates and executes.
```

## Database Tables Used By LLM Agents

## `agent_invocations`

This is the full audit trail of an LLM call.

Purpose:

- Preserve the exact context sent to the agent.
- Preserve the raw output.
- Preserve parsed output.
- Track validation success or failure.
- Link future outcomes back to the original LLM recommendation.
- Support future fine-tuning data generation.

Important fields:

```text
id
project_id
dataset_id
plan_id
job_id
agent_name
agent_version
prompt_version
provider
model
input_messages jsonb
input_context jsonb
raw_output text
parsed_output jsonb
validation_status
validation_error
accepted_for_memory
human_feedback jsonb
downstream_outcome jsonb
created_at
```

How LLMs use it:

LLMs do not normally read full invocation records directly during prompts. Instead, invocation records are the source-of-truth trace. Distilled memory records are what future prompts consume.

Why it matters:

This table is the best foundation for future fine-tuning datasets because it contains prompt input, model output, validation status, and downstream outcome.

## `agent_memory_records`

This is the reusable memory layer.

Purpose:

- Store concise lessons and observations from agent work.
- Avoid stuffing full raw traces into future prompts.
- Let agents use prior outcomes and strategy memory.

Important fields:

```text
id
invocation_id
project_id
dataset_id
plan_id
job_id
agent_name
kind
summary
payload jsonb
tags jsonb
created_at
```

Current memory kinds:

```text
dataset_analysis
preprocessing_recommendation
training_evaluation
planning_feedback
planning_outcome
model_ranking
```

Currently active kinds:

- `training_evaluation`
- `planning_feedback`
- `planning_outcome`

Reserved / future kinds:

- `dataset_analysis`
- `preprocessing_recommendation`
- `model_ranking`

How LLMs use it:

- Training Monitor receives recent related memory for the same project / dataset.
- Experiment Planner receives prior memory, including training evaluations, planning feedback, and planning outcomes.
- Planner outcome memories are split into successful and failed strategies so the LLM can learn from what actually happened.

Important distinction:

```text
agent_invocations = full trace
agent_memory_records = distilled reusable lessons
```

## `agent_decisions`

This table stores project-level agent decisions.

Purpose:

- Record durable decisions such as add experiments, select champion, or stop project.
- Provide a source decision ID for follow-up plans.
- Drive Mission Control decision display.

Important fields:

```text
id
project_id
plan_id
decision_type
rationale
payload jsonb
created_at
```

Decision types:

```text
WAIT
SELECT_CHAMPION
ADD_EXPERIMENTS
STOP_PROJECT
```

How LLMs affect it:

The Experiment Planner can produce a decision. The backend validates it, then stores it here.

How the backend uses it:

- `ADD_EXPERIMENTS` can become a follow-up plan.
- `SELECT_CHAMPION` can persist a champion record.
- `STOP_PROJECT` prevents new scheduling.
- `WAIT` is generally not persisted as an actionable decision.

## `experiment_plans`

This table stores planned experiment batches.

Important fields:

```text
id
project_id
dataset_id
status
source_decision_id
target_metric
recommended_workers
estimated_minutes
experiments jsonb
warnings jsonb
created_at
```

How LLMs affect it:

An accepted `ADD_EXPERIMENTS` decision can be converted into a new follow-up plan. The LLM does not insert the plan directly; the backend creates the plan from validated proposed experiments.

The `source_decision_id` links the follow-up plan back to the LLM decision that proposed it.

Why this matters:

That link allows the backend to later ask:

```text
Did the LLM's proposed strategy actually improve the champion?
```

## `experiment_jobs`

This table stores individual executable jobs derived from a plan.

Important fields:

```text
id
project_id
worker_id
template
status
config jsonb
mlflow_run_id
error
created_at
updated_at
```

How LLMs affect it:

LLMs propose experiment configs. The backend validates them, puts them into a plan, and then creates jobs from that plan.

The LLM does not directly create jobs.

## `epoch_metrics`

This table stores per-epoch metrics.

Important fields:

```text
job_id
epoch
metrics jsonb
created_at
```

How LLMs use it:

Training Monitor and Experiment Planner can use these metrics to infer:

- Plateauing.
- Overfitting.
- Underfitting.
- Instability.
- Whether a model is still improving.
- Whether more epochs are justified.

## `training_run_summaries`

This table stores final run-level summaries.

Important fields:

```text
job_id
project_id
plan_id
dataset_id
model
provider
gpu_type
status
runtime_seconds
estimated_cost_usd
best_macro_f1
best_accuracy
final_train_loss
final_val_loss
epochs_completed
modal_function_call_id
modal_input_id
created_at
updated_at
```

How LLMs use it:

This is the main source for quick run comparison: quality, cost, runtime, and completion status.

## `training_run_evaluations`

This table stores richer evaluation outputs.

Important fields:

```text
job_id
project_id
plan_id
dataset_id
objective_profile jsonb
per_class_metrics jsonb
confusion_matrix jsonb
model_profile jsonb
holistic_scores jsonb
recommendation_summary
created_at
updated_at
```

How LLMs use it:

This is where more holistic model-ranking evidence should live:

- Per-class performance.
- Confusion matrix.
- Objective-specific profile.
- Model size.
- Latency / deployment profile.
- Holistic scores.

Why it matters:

This is the path away from choosing winners based only on accuracy or Macro-F1.

## `project_champions`

This table stores the selected champion model for a project.

Important fields:

```text
id
project_id
dataset_id
plan_id
job_id
source_decision_id
selection_reason
metrics jsonb
evaluation jsonb
deployment_profile jsonb
created_at
updated_at
```

How LLMs affect it:

If the Experiment Planner or deterministic reviewer selects a champion, the backend persists the champion here after validation.

The champion record can include:

- Summary metrics.
- Rich evaluation.
- Deployment profile.
- Objective context.
- Model-card-like notes.
- Export status.

## `automation_settings`

This table stores project automation behavior.

Important fields:

```text
auto_review_experiments
auto_schedule_followups
auto_execute_plans
max_followup_rounds
default_training_provider
default_gpu_type
llm_enabled
agent_mode
llm_provider
llm_model
updated_at
```

How LLMs use it:

This controls whether LLMs run and whether validated LLM decisions are allowed to schedule / execute new work.

## Memory Retrieval Behavior

Current retrieval is mostly project- and dataset-scoped.

Current pattern:

- Fetch recent memory for the same project.
- Filter by dataset when possible.
- Provide a limited number of records to prompts.

Current Training Monitor memory behavior:

- Fetches a small number of relevant records for the same project / dataset.

Current Experiment Planner memory behavior:

- Fetches more project / dataset memory.
- Separates successful and failed planning outcomes.
- Gives the planner examples of what previously worked or failed.

Important limitation:

Memory retrieval is not yet based on semantic similarity between datasets. A future system could retrieve lessons from similar datasets, not only the exact same dataset.

## How LLM Experiment Proposals Become Backend Work

The LLM does not create plans, jobs, workers, or database rows directly.

The pipeline is:

```text
LLM JSON output
-> parse into typed Go struct
-> validate agent-specific rules
-> validate proposed experiment configs
-> reject duplicates and unsupported options
-> create agent decision
-> optionally create follow-up plan
-> optionally enqueue jobs
-> backend ensures workers
```

## Proposed Experiment Schema

The current proposed experiment shape supports:

```json
{
  "template": "image_classification",
  "model": "mobilenet_v3_small",
  "epochs": 16,
  "batch_size": 32,
  "learning_rate": 0.0003,
  "reason": "Why this experiment is worth running.",
  "image_size": 224,
  "optimizer": "adamw",
  "scheduler": "cosine",
  "weight_decay": 0.0001,
  "augmentation": "moderate",
  "class_balancing": "weighted_loss",
  "early_stopping_patience": 5,
  "strategy": "champion challenge",
  "pretrained": true,
  "freeze_backbone": false,
  "fine_tune_strategy": "last_block"
}
```

Backend validation includes:

- `template` is required.
- `model` is required.
- Model must exist in the supported model catalog.
- `epochs` must be within allowed bounds.
- `batch_size` must be within allowed bounds.
- `learning_rate` must be positive and within allowed bounds.
- `image_size` must be within allowed bounds.
- `weight_decay` must be within allowed bounds.
- `early_stopping_patience` must be within allowed bounds.
- `fine_tune_strategy` must be one of:
  - `head_only`
  - `last_block`
  - `full`
- Duplicate experiment signatures are rejected.

## Supported Model Catalog

The planner must choose from the backend's supported catalog.

Current catalog includes:

```text
mobilenet_v3_small
mobilenet_v3_large
efficientnet_b0
regnet_y_400mf
efficientnet_b1
efficientnet_b2
resnet18
resnet34
convnext_tiny
swin_t
vit_b_16
```

Each model has metadata such as:

- Family.
- Deployment tier.
- Default image size.
- Minimum recommended image count.
- Whether transfer learning is supported.
- Expected latency class.
- Recommended use.
- Supported fine-tuning modes.

Why this matters:

The LLM can reason more aggressively, but only inside a catalog the worker can actually execute.

## Project Goal / Objective Context

When the user creates a project, they can provide a goal or extra context.

The backend converts that free text into an objective context.

Example goal:

```text
I want a model that can classify images accurately and quickly for a live service.
```

The backend may infer:

```json
{
  "primary_objective": "low_latency_live_service",
  "metric_preferences": ["macro_f1", "accuracy", "latency_ms", "model_size_mb"],
  "deployment_priorities": ["low_latency", "compact_model", "stable_predictions"],
  "constraints": ["prefer mobile/edge friendly architectures"],
  "ranking_weights": {
    "macro_f1": 0.35,
    "accuracy": 0.2,
    "latency": 0.25,
    "cost": 0.1,
    "stability": 0.1
  }
}
```

How LLMs use it:

- Training Monitor interprets a run relative to the user's actual goal.
- Experiment Planner chooses follow-up experiments that match the intended deployment use case.
- Champion selection can weigh speed, cost, and stability instead of only validation accuracy.

Why this matters:

Two projects with the same dataset may need different winners. A live service may prefer a slightly less accurate but much faster model.

## Planning Outcome Memory

Planning outcome memory is one of the most important parts of the learning loop.

When an LLM proposes a follow-up plan, the backend later checks what actually happened after that plan completes.

It records:

- Source decision ID.
- Source plan ID.
- Follow-up plan ID.
- Baseline champion.
- Actual best run.
- Expected delta.
- Actual delta.
- Total cost.
- Total runtime.
- Number of successful / failed jobs.
- Proposed experiments.
- Lesson learned.

Possible outcome statuses:

```text
improved_champion
minor_improvement
no_improvement
failed
```

How future LLM calls use it:

The planner can see whether prior strategies actually worked. This is the first version of "learning from experience" without fine-tuning.

## Current Backend Guardrails

The backend protects the system from bad LLM output with several guardrails:

- LLM output must be valid JSON.
- Parsed JSON must match expected structs.
- Decision type must be allowed.
- Proposed experiments must be complete.
- Model names must be supported.
- Numeric hyperparameters must be in allowed ranges.
- Duplicate experiment signatures are rejected.
- Batches with too little meaningful variation are rejected.
- Batches that over-focus on one model are rejected.
- Maximum follow-up rounds prevent infinite loops.
- Autonomous execution depends on explicit settings.
- Workers and jobs are managed by backend execution logic, not by LLMs.

## Current Limitations

The current system is functional, but there are known limits.

### Dataset Understanding Is Still Thin

The planner receives dataset-level insights, but there is not yet a full Dataset Analysis / Preprocessing Agent.

Missing or underdeveloped areas:

- Class imbalance analysis.
- Image dimension distribution.
- Corrupt file summary.
- Object size / crop analysis.
- Annotation metadata usage.
- Split quality / leakage detection.
- Dataset-specific augmentation strategy.
- Dataset-specific preprocessing strategy.

### Metadata Is Not Yet First-Class

Some datasets, such as Stanford Dogs, may include extra metadata folders or annotations.

The current system primarily treats datasets as image classification folders. Future support should preserve and profile extra metadata, such as:

- Bounding boxes.
- XML / JSON annotations.
- CSV labels.
- Class hierarchy files.
- Train / validation / test split files.
- Image-level attributes.

This could help the LLM make better decisions, such as testing crop-based training when bounding boxes are available.

### Memory Retrieval Is Not Yet Semantic

The system mostly retrieves memory by project / dataset.

Future retrieval should support:

- Similar datasets.
- Similar class counts.
- Similar imbalance profiles.
- Similar image sizes.
- Similar deployment goals.
- Similar failed / successful strategies.

### The Planner Can Still Be Too Conservative

Even with validation requiring meaningful changes, the LLM may still propose experiments that are not strategically strong.

This can be improved with:

- Better prompt examples.
- Stronger output schemas.
- Strategy scoring.
- Explicit exploration / exploitation controls.
- More detailed dataset profiles.
- More detailed outcome memory.

## Future Schema Ideas

## Dataset Artifact Schema

A future table could track all meaningful dataset files, not only images.

Possible table:

```text
dataset_artifacts
```

Possible fields:

```text
id
dataset_id
artifact_type
path
format
description
detected_schema jsonb
created_at
```

Example artifact types:

```text
image_root
class_folder
annotation_xml
annotation_json
labels_csv
split_file
metadata_folder
bounding_boxes
class_hierarchy
```

Why this helps:

The Dataset Analysis Agent could tell the planner:

```text
This dataset includes bounding boxes, so test crop-based preprocessing against full-image training.
```

## Dataset Profile Schema

Possible table:

```text
dataset_profiles
```

Possible fields:

```text
id
dataset_id
profile_version
class_count
image_count
class_distribution jsonb
image_dimension_stats jsonb
corrupt_file_count
split_summary jsonb
metadata_summary jsonb
imbalance_summary jsonb
leakage_warnings jsonb
recommended_metrics jsonb
created_at
```

Why this helps:

LLM agents need objective dataset facts before proposing strategy.

## Preprocessing Recommendation Schema

Possible memory payload:

```json
{
  "recommended_image_size": 224,
  "normalization": "imagenet",
  "augmentation_policy": "moderate",
  "class_balancing": "weighted_loss",
  "crop_strategy": "bbox_crop_if_available",
  "split_strategy": "stratified",
  "metrics": ["macro_f1", "accuracy", "per_class_recall"],
  "rationale": "Dataset is imbalanced and has object-centered images with useful bounding boxes.",
  "risks": ["Aggressive crops may remove context."]
}
```

## Strategy Outcome Scorecard

A future memory schema could score each planning strategy.

Possible payload:

```json
{
  "strategy_id": "bbox_crop_efficientnet_low_latency",
  "strategy_type": "preprocessing_plus_model_family",
  "dataset_traits": ["many_classes", "fine_grained", "bbox_annotations"],
  "objective": "low_latency_live_service",
  "expected_delta": 0.03,
  "actual_delta": 0.011,
  "cost_usd": 0.18,
  "runtime_seconds": 1260,
  "winner_model": "mobilenet_v3_large",
  "outcome": "minor_improvement",
  "lesson": "Crop strategy helped recall on similar-looking classes but increased preprocessing complexity."
}
```

Why this helps:

The LLM should learn strategy-level lessons, not just individual run facts.

## Future Algorithm Ideas

## Multi-Objective Ranking

Champion ranking should consider more than one metric.

Useful factors:

- Validation accuracy.
- Macro-F1.
- Per-class recall.
- Confusion matrix quality.
- Train / validation gap.
- Stability across epochs.
- Plateau behavior.
- Inference latency.
- Model size.
- Training cost.
- Training time.
- Deployment goal fit.

A future ranking score could combine these with project-specific weights.

For example:

```text
score =
  0.35 * macro_f1
+ 0.20 * accuracy
+ 0.15 * per_class_balance
+ 0.10 * stability
+ 0.10 * latency_score
+ 0.05 * model_size_score
+ 0.05 * cost_score
```

For live inference, latency and model size should weigh more.

For offline batch classification, quality may weigh more.

## Champion Challenge Protocol

Once a strong champion exists, the planner should stop broad random exploration and use a champion challenge protocol.

A challenge batch might include:

- One direct champion refinement.
- One nearby model with better expected tradeoff.
- One preprocessing challenge.
- One class-imbalance challenge.
- One higher-capacity challenger only if budget allows.

This avoids running ten small variants of the same model while still exploiting what works.

## Exploration / Exploitation Balance

The planner should choose a mode:

```text
explore
exploit
champion_challenge
preprocessing_ablation
stop_or_select
```

Each mode should have different rules.

Examples:

- `explore`: test different model families and preprocessing strategies.
- `exploit`: refine a known good family.
- `champion_challenge`: only run experiments with a plausible path to beating the champion.
- `preprocessing_ablation`: isolate preprocessing decisions.
- `stop_or_select`: choose a champion if improvement has stalled.

## Validation-Set Overfitting Protection

If the system keeps optimizing on validation results, it can overfit to the validation split.

Future improvements:

- Hold out a true test split.
- Only evaluate final champions on test data.
- Track how many times a dataset has been optimized.
- Penalize excessive small validation gains after many rounds.
- Prefer robust improvements across multiple metrics.

## Future Prompt Improvements

The Experiment Planner prompt should increasingly force evidence-based strategy.

Useful required fields:

```json
{
  "evidence_used": [
    "Current champion has strong macro-F1 but weak class recall on minority classes.",
    "EfficientNet-B1 underperformed at 224 but may have been undertrained.",
    "Prior higher-resolution MobileNet did not improve enough to justify cost."
  ],
  "rejected_options": [
    {
      "option": "Run another MobileNet with only more epochs",
      "reason": "Prior runs plateaued and this is unlikely to be meaningful."
    }
  ],
  "hypothesis": "Class-weighted loss plus moderate augmentation may improve minority-class recall.",
  "expected_failure_modes": [
    "Weighted loss may hurt majority-class precision.",
    "Higher image size may increase latency beyond the project goal."
  ],
  "stop_condition": "If no experiment improves macro-F1 by 0.02 or improves recall without latency regression, select the current champion."
}
```

Prompt examples should show bad and good behavior.

Bad behavior:

```text
Run the same model with 2 more epochs and a tiny learning-rate change.
```

Good behavior:

```text
Challenge the current champion with a different preprocessing hypothesis, a controlled class-balancing experiment, and one deployment-friendly architecture alternative.
```

## Future Fine-Tuning Path

The current database design can support future OpenAI fine-tuning, but raw memory should not be used directly without curation.

Best source tables:

- `agent_invocations`
- `agent_memory_records`
- `agent_decisions`
- `training_run_summaries`
- `training_run_evaluations`
- `project_champions`

Good fine-tuning examples would include:

- The input context given to the LLM.
- The accepted output.
- Whether backend validation passed.
- The downstream outcome.
- Human feedback, if available.

Important:

Do not fine-tune only on accepted LLM outputs. Some accepted outputs may later prove strategically weak. The downstream outcome matters.

Better training label:

```text
This recommendation was valid JSON and passed validation, but the follow-up plan produced no improvement.
```

That kind of record is useful for training the model not to repeat weak strategies.

## Most Important Next Improvements

The highest-leverage next improvements are:

1. Build the Dataset Analysis / Preprocessing Agent.
2. Make metadata and annotations first-class dataset artifacts.
3. Improve memory retrieval using dataset similarity and strategy outcomes.
4. Strengthen the planner schema with evidence, rejected options, expected failure modes, and stop conditions.
5. Add explicit strategy modes such as explore, exploit, champion challenge, and preprocessing ablation.
6. Improve champion ranking with multi-objective scoring.
7. Use downstream outcomes as feedback for future prompts and eventual fine-tuning.

## Short Summary For Another Chat

Model Express currently uses LLM agents as structured decision-makers, not as direct executors. The Training Monitor evaluates individual completed runs and writes `training_evaluation` memory. The Experiment Planner reviews completed experiment batches and can propose `ADD_EXPERIMENTS`, `SELECT_CHAMPION`, `STOP_PROJECT`, or `WAIT`. The backend parses and validates all LLM JSON, rejects unsupported models or duplicate/minor experiment proposals, stores decisions in Postgres, and only then schedules follow-up plans if autonomous settings allow it.

The important DB split is:

```text
agent_invocations = full LLM trace for audit/fine-tuning
agent_memory_records = distilled reusable lessons for future prompts
agent_decisions = durable actionable project decisions
planning_outcome memory = whether prior LLM strategies actually worked
```

The main gap is not whether the LLM can propose experiments. It can. The main gap is giving it richer dataset understanding, better strategy memory, stronger prompts, multi-objective ranking, and outcome-aware learning so it stops making shallow tweaks and starts making evidence-based experiment strategies.
