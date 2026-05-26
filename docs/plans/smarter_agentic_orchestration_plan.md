# Smarter Agentic Orchestration Plan

## Purpose

This plan covers the next Model Express upgrades needed to make the agentic loop behave more like a practical ML experimentation system:

- Scale workers to the amount of queued work.
- Use the project `Goal` as first-class LLM context.
- Move from single-metric model choice toward holistic model ranking.
- Expand supported model families and training templates.
- Surface selected champions clearly in Mission Control with deployment-relevant diagnostics.

The guiding principle is that LLM agents may reason and propose, but deterministic orchestration still validates, schedules, and records outcomes.

## Current Gaps

1. Worker scaling is too coarse.
   - Plans can queue multiple jobs, but worker requirements do not yet clearly express the number of workers needed relative to queued jobs.
   - Existing workers should keep polling and pick up the next queued job after finishing, but the system should also request more workers when queued work exceeds available capacity.

2. The project goal is underused.
   - `Project.Goal` exists and is already passed into some planner context, but it is not interpreted into objective preferences.
   - A user goal like "fast live image classification" should influence model choice, ranking, metrics, and deployment tradeoffs.

3. Ranking is still too narrow.
   - Accuracy or Macro-F1 alone is not enough.
   - The system needs to consider per-class behavior, overfitting, runtime, cost, model size, and inference latency.

4. Model search space is too small.
   - Current supported families are mostly MobileNetV3-small, EfficientNet-B0/B1, and ResNet18/34.
   - Transfer learning should remain the default, but the planner should be able to choose from additional lightweight, medium, and stronger model families.

5. Champion selection is not visible enough.
   - Agent decisions show `SELECT_CHAMPION`, but Mission Control should have a dedicated champion panel.
   - A selected champion should include metrics and use-case diagnostics, not just a model name.

## Phase 1: Worker Scaling To Queued Jobs

### Goal

When an experiment plan queues jobs, Model Express should make sure enough workers are available to make progress. Existing workers should continue polling after each job, and the system should request additional workers when queued jobs exceed current active/starting capacity.

### Backend Changes

Files:

- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/store/store.go`
- `services/orchestrator/internal/store/memory.go`
- `services/orchestrator/internal/store/postgres.go`
- `services/orchestrator/internal/execution/model.go`

Implementation:

1. Add a deterministic worker scaling policy.
   - Compute `queuedTrainingJobsForPlan`.
   - Compute `activeOrStartingWorkersForProject`.
   - Desired workers should be:
     - at least `1` when jobs are queued
     - at most queued jobs
     - at most plan `recommended_workers`
     - optionally capped by `MODEL_EXPRESS_MAX_AUTO_WORKERS`

2. Update worker requirements after jobs are queued and after jobs finish.
   - If `queued_jobs > available_capacity`, increase `WorkerRequirement.TargetCount`.
   - If workers are already active and enough, mark requirement `ACTIVE`.
   - If no workers are available, mark requirement `PENDING` or `STARTING`.

3. Record clearer execution events:
   - `JOBS_QUEUED`
   - `WORKERS_REQUIRED`
   - `WORKERS_STARTING`
   - `WORKERS_ACTIVE`
   - possible new event: `WORKER_SCALING_UPDATED`

### Mission Control Changes

Files:

- `apps/mission-control/src/App.tsx`
- `apps/mission-control/src/types.ts`
- `apps/mission-control/electron/main.cjs`

Implementation:

1. Make the worker requirement poller satisfy `target_count`, not just "start one worker."
2. Start additional workers when:
   - requirement status is `PENDING` or `STARTING`
   - active known workers are fewer than `target_count`
3. Keep manual worker controls intact.
4. Show queued jobs, active workers, and target workers in the automation timeline or worker panel.

### Important Behavior

Workers should continue polling after completing a job. If enough workers are already running, no new worker should be started. If queued jobs remain and worker count is too low, the UI/Electron layer should start additional workers.

## Phase 2: Goal-Aware Agent Context

### Goal

The user-entered project `Goal` should shape planning and ranking. The system should extract objective preferences from it and pass those preferences to LLM agents.

Example:

> "I need a model that predicts accurately and quickly for a live-time service."

This should translate into:

- prioritize low inference latency
- prefer compact models when quality is close
- evaluate accuracy, Macro-F1, cost, runtime, and inference speed
- require deployment tradeoff explanations

### Data Model

Add a lightweight computed context struct, no migration required at first:

```go
type ProjectObjectiveContext struct {
  GoalText string
  PrimaryObjective string
  MetricPreferences []string
  DeploymentPriorities []string
  Constraints []string
  RankingWeights map[string]float64
}
```

This can be generated deterministically from keyword heuristics first:

- "live", "real-time", "fast", "instant" -> latency priority
- "cheap", "budget" -> cost priority
- "accurate", "best quality" -> quality priority
- "imbalanced", "minority classes" -> Macro-F1/per-class priority
- "mobile", "edge" -> model size/inference priority

### Backend Changes

Files:

- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/training_monitor_llm.go`

Implementation:

1. Build `ProjectObjectiveContext` from `Project.Goal`.
2. Add it to:
   - Experiment Planner prompt context
   - Training Monitor prompt context
   - future model ranking context
3. Persist it in planner decision payloads for auditability.
4. Require the planner to explain how its proposal satisfies the goal.

### Agent Prompt Changes

The planner should receive:

- raw goal text
- interpreted objective context
- dataset planning insights
- champion and delta context
- successful/failed strategy memory

The planner output should include:

- `goal_alignment`
- `deployment_tradeoff`
- `success_criteria`
- `ranking_metrics_to_watch`

## Phase 3: Holistic Multi-Metric Evaluation

### Goal

Stop choosing models based on one metric. Every completed run should be evaluated with a holistic score and explanation.

### Metrics To Track

Quality:

- validation accuracy
- Macro-F1
- per-class precision/recall/F1
- confusion matrix
- train/validation loss gap
- overfitting indicators
- stability across epochs

Efficiency:

- training runtime
- estimated training cost
- model parameter count
- model size on disk
- estimated inference latency
- throughput/images per second

Goal alignment:

- quality-first
- low-latency live service
- low-cost experimentation
- balanced class performance

### Data Model Options

Preferred: add a separate run evaluation table.

```sql
training_run_evaluations (
  job_id text primary key,
  project_id text not null,
  plan_id text not null,
  dataset_id text not null,
  objective_profile jsonb not null,
  per_class_metrics jsonb not null,
  confusion_matrix jsonb not null,
  model_profile jsonb not null,
  holistic_scores jsonb not null,
  recommendation_summary text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
)
```

Why separate table:

- avoids bloating `training_run_summaries`
- supports future export/model-card work
- lets us keep summary rows stable for simple UI tables

### Worker Changes

Files:

- `services/worker/worker/training/modal_app.py`
- `services/worker/worker/training/local.py`
- `services/worker/worker/orchestrator_client.py`

Implementation:

1. During evaluation, compute:
   - predictions
   - true labels
   - confusion matrix
   - per-class report via `sklearn.metrics.classification_report`
2. Add a small inference benchmark:
   - run N forward passes on representative batch
   - report median latency and throughput
3. Estimate model size:
   - parameter count
   - serialized state dict size if feasible
4. Post evaluation payload to orchestrator.

### Orchestrator Changes

Add endpoints:

- `POST /jobs/:id/training-run-evaluation`
- `GET /jobs/:id/training-run-evaluation`
- `GET /projects/:id/training-run-evaluations`

The Training Monitor and Experiment Planner should consume these richer evaluations when available.

## Phase 4: Expanded Model And Template Support

### Goal

Give the planner a wider but still controlled search space.

### Model Families To Add

Fast/live candidates:

- `mobilenet_v3_large`
- `efficientnet_b0`
- `regnet_y_400mf`

Balanced candidates:

- `efficientnet_b1`
- `efficientnet_b2`
- `resnet18`
- `resnet34`

Higher-quality challengers:

- `convnext_tiny`
- `swin_t`
- `vit_b_16` only for larger datasets or explicit quality-first goals

### Template Types

Current templates can remain compatible, but planner config should include more training mode fields:

```json
{
  "template": "image_transfer_classification",
  "model": "convnext_tiny",
  "pretrained": true,
  "freeze_backbone": true,
  "fine_tune_strategy": "head_only|last_block|full",
  "image_size": 224,
  "optimizer": "adamw",
  "scheduler": "cosine",
  "weight_decay": 0.01,
  "augmentation": {},
  "class_balancing": "weighted_loss"
}
```

### Worker Changes

Files:

- `services/worker/worker/training/modal_app.py`
- `services/worker/worker/training/local.py`

Implementation:

1. Add `_build_model` support for new torchvision models.
2. Add model metadata:
   - family
   - default image size
   - deployment tier
   - expected latency class
3. Add fine-tuning strategies:
   - head-only
   - last block
   - full fine-tune
4. Validate unsupported model/template combinations early.

### Planner Guardrails

The planner should not be allowed to pick arbitrary model names. Add a backend-supported model catalog:

```go
type SupportedModelSpec struct {
  Name string
  Family string
  DeploymentTier string
  DefaultImageSize int
  MinRecommendedImages int
  SupportsTransfer bool
}
```

The LLM receives this catalog and proposals are validated against it.

## Phase 5: Champion Selection And Dashboard

### Goal

When a champion is selected, Mission Control should show a dedicated champion panel with the model’s practical strengths and weaknesses.

### Backend Data Model

Add a champion record:

```sql
project_champions (
  id text primary key,
  project_id text not null,
  dataset_id text not null,
  plan_id text not null,
  job_id text not null,
  source_decision_id text not null,
  selection_reason text not null,
  metrics jsonb not null,
  evaluation jsonb not null,
  deployment_profile jsonb not null,
  created_at timestamptz not null default now()
)
```

Why a separate table:

- champion selection is a project-level milestone
- supports replacing champions later
- gives the UI a stable endpoint

### Backend Flow

When a decision is `SELECT_CHAMPION`:

1. Resolve `champion_job_id`.
2. Load:
   - training summary
   - run evaluation if available
   - model config/job config
   - project objective context
3. Create or update project champion.
4. Emit `CHAMPION_SELECTED` execution event.

### UI Changes

Files:

- `apps/mission-control/src/App.tsx`
- `apps/mission-control/src/types.ts`
- `apps/mission-control/src/styles.css`

Champion panel should show:

- model name
- job id / plan id
- status
- validation accuracy
- Macro-F1
- cost
- runtime
- epochs
- model size
- inference latency
- deployment recommendation
- selection rationale
- goal alignment

For image classification diagnostics:

- confusion matrix
- per-class F1 table
- top confused class pairs

If evaluation details are not available yet, show the summary metrics and mark diagnostics as pending.

## Phase 6: Better Agent Decisions

### Planner Improvements

The planner should reason in modes:

- `preprocessing_search`
- `champion_refinement`
- `architecture_challenge`
- `regularization_search`
- `cost_latency_tradeoff`
- `final_validation`
- `select_or_stop`

Each `ADD_EXPERIMENTS` decision should include:

- hypothesis
- changed variables
- expected improvement
- success criteria
- goal alignment
- deployment tradeoff
- why this is not repeating failed strategy memory

### Training Monitor Improvements

The Training Monitor should evaluate:

- whether a run is improving or plateauing
- overfitting/underfitting
- metric stability
- class-specific failures
- latency/cost concerns
- whether a model is a candidate champion

### Dataset Analysis Agent

Add a real Dataset Analysis Agent after deterministic profiling:

Input:

- dataset profile
- project goal
- supported model catalog

Output:

- preprocessing recommendations
- augmentation recommendations
- metric recommendations
- split-quality warnings
- deployment implications

This agent can write memory before the first experiment plan is generated.

## Suggested Implementation Order

1. Worker scaling.
   - This improves reliability immediately.
   - It also makes larger experiment batches practical.

2. Project objective context.
   - Cheap and high leverage.
   - Makes the existing planner more aligned with user intent.

3. Holistic run evaluation endpoint.
   - Needed for real champion selection and smarter planner feedback.

4. Expanded model catalog and Modal templates.
   - Add more search power, but only after ranking can judge tradeoffs.

5. Champion record and Mission Control panel.
   - Makes the system feel like it is progressing toward a usable final model.

6. Dataset Analysis Agent.
   - Best added once objective context and richer evaluations are available.

## Additional Insights

### Add A Held-Out Test Split

The current train/validation approach is enough for iteration, but champion selection should ideally use a held-out test split or final validation pass. This avoids selecting a model that overfit repeated validation feedback.

### Record Model Cards

When a champion is selected, generate a model-card style record:

- intended use
- dataset summary
- metrics
- known limitations
- latency/cost profile
- recommended preprocessing
- export instructions

### Add Budget-Aware Planning

The planner should see remaining budget or user budget preferences. A live-service goal may justify extra quality experiments; a cheap prototype goal may not.

### Avoid Over-Exploration

Expanded model support can become expensive. The planner should use model families in stages:

1. cheap baseline sweep
2. targeted preprocessing search
3. challenger models
4. champion refinement
5. final validation

### Treat Latency As A First-Class Metric

For live image classification, a model that is 0.5% worse but 3x faster may be the better recommendation. The ranking system should be able to explain that.

## Tests

Backend:

- Worker requirement target count scales with queued jobs.
- Existing active workers reduce the required additional count.
- Project goal produces expected objective context.
- Planner receives objective context and dataset insights.
- Unsupported model proposals are rejected.
- Holistic ranking changes when goal prioritizes latency over quality.
- `SELECT_CHAMPION` creates or updates a champion record.

Worker:

- New model names build successfully.
- Evaluation payload includes confusion matrix and per-class metrics.
- Inference benchmark payload shape is stable.

Frontend:

- Worker requirement panel shows target vs active workers.
- Champion panel renders summary-only state.
- Champion panel renders confusion matrix/per-class metrics when available.

Build:

- `go test ./...` in `services/orchestrator`
- `npm run build` in `apps/mission-control`
- `python -m py_compile` for worker training files
