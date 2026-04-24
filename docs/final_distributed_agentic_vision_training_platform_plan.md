# Distributed Agentic Vision Training Platform for Gaming AI

## Project Summary

This project is a distributed, agentic computer vision training platform designed to produce strong image-based classifiers for a Gaming AI application.

The system should help turn labeled gameplay screenshots into usable vision models by:

- profiling gaming image datasets
- planning model experiments with specialized agents
- dispatching training jobs across local or cloud GPU workers
- tracking runs, metrics, and artifacts with MLflow
- pruning bad experiments early using mid-training statistics
- adapting future experiments based on shared results
- selecting the best model
- exporting optimized models for low-latency local inference in the Gaming AI app

The project should be built as a real ML systems project, not as a generic AutoML clone.

---

## One-Sentence Pitch

A distributed agentic vision training platform that uses cloud GPU workers, MLflow tracking, and agent-driven diagnostics to automatically train, prune, improve, and export gaming-specific computer vision classifiers for a Gaming AI application.

---

## Core Motivation

The Gaming AI application needs specialized vision models to understand gameplay context.

Possible classifier goals:

- combat vs non-combat
- menu vs gameplay
- loading screen vs active gameplay
- low-health vs normal-health
- enemy visible vs not visible
- inventory/map/menu open vs closed
- game-state classification from screenshots

Manually training, comparing, and improving these classifiers is slow.

This platform exists to make that workflow faster, more structured, distributed, reproducible, and more explainable.

---

## What This Project Is

This project is:

- a distributed ML experiment orchestration system
- an agentic AI project
- a computer vision training platform
- an ML infrastructure project
- a practical internal tool for the Gaming AI app

It combines:

```text
Agents for reasoning
+
Orchestrator for control
+
Workers for training
+
MLflow for tracking
+
Gaming AI export pipeline for product integration
```

---

## What This Project Is Not

This project is not:

- a generic AutoML clone
- an MLflow clone
- a dashboard-only project
- a tool that promises to train any model on any dataset
- a system where LLMs freely generate arbitrary training code
- a real-time cloud inference system as the primary MVP

---

## Key Design Philosophy

### Agents reason. The orchestrator controls. Workers execute. MLflow tracks. The Gaming AI consumes exported models.

This separation is important.

Agents should make structured recommendations.

The orchestrator should validate those recommendations, schedule jobs, enforce budgets, monitor training, prune bad runs, and decide what happens next.

Workers should execute controlled training templates.

MLflow should handle experiment tracking, metrics, artifacts, and model records.

The Gaming AI app should use the exported optimized model for local inference.

---

## Why Not Just Use MLflow / AutoML / LangGraph?

Existing tools provide important infrastructure, but they do not fully solve this product-specific problem.

### MLflow provides:

- experiment tracking
- metrics
- artifacts
- model registry
- run comparison
- experiment dashboard

### AutoML provides:

- automatic model training
- model search
- hyperparameter search

### LangGraph provides:

- agent/workflow graph infrastructure
- stateful workflow management

### This project provides:

- Gaming-AI-specific classifier workflow
- distributed GPU worker orchestration
- agent-driven experiment planning
- mid-training pruning
- adaptive scheduling
- shared learning across workers
- application-specific diagnostics
- model export for local Gaming AI inference
- decision traces explaining why experiments were launched, stopped, or selected

Key distinction:

```text
MLflow records what happened.
AutoML can train models.
This platform decides what should happen next for a real Gaming AI vision use case.
```

---

## Primary MVP Scope

Focus first on image classification for gameplay screenshots.

Do not start with tabular data.

Do not start with video classification.

Do not start with object detection.

Those can come later.

### MVP Task

Train and improve image classifiers for game-state recognition.

Example dataset:

```text
dataset/
  combat/
    img001.png
    img002.png
  exploration/
    img003.png
    img004.png
  menu/
    img005.png
    img006.png
```

The system should support binary and multiclass image classification.

---

## Future Scope

After the image classifier workflow works:

1. Object detection support
   - enemy visible
   - UI element detection
   - health bar detection
   - minimap detection

2. Temporal/video understanding
   - short clip classification
   - event transition detection
   - scene/action recognition

3. Advanced Gaming AI integration
   - active learning from uncertain frames
   - automatic data collection suggestions
   - model confidence monitoring during gameplay

---

## High-Level System Flow

```text
User creates vision training project
↓
User uploads labeled gameplay screenshot dataset
↓
Dataset profiler computes deterministic dataset statistics
↓
Profiler Agent analyzes dataset risks and recommendations
↓
Planner Agent creates initial experiment plan
↓
Orchestrator validates experiment configs
↓
Training jobs are dispatched to local/cloud GPU workers
↓
Workers train models and report metrics every epoch
↓
MLflow logs params, metrics, artifacts, and models
↓
Orchestrator monitors mid-training performance
↓
Bad runs are pruned early
↓
Diagnostics Agent analyzes completed and pruned runs
↓
Planner proposes targeted follow-up experiments
↓
Orchestrator repeats until budget/stopping condition
↓
Best model is selected
↓
Export pipeline creates Gaming AI inference package
↓
Gaming AI app uses optimized model locally
```

---

## Core Architecture

```text
Frontend / Mission Control UI
        ↓
Backend API / Orchestrator
        ↓
Agent Layer
        ↓
Job Queue / Scheduler
        ↓
Local or Cloud GPU Workers
        ↓
MLflow Tracking Server
        ↓
Model Export Package
        ↓
Gaming AI Application
```

---

## Major Components

### 1. Orchestrator

The orchestrator is the core system component.

Responsibilities:

- project state management
- experiment planning lifecycle
- job scheduling
- worker coordination
- concurrency limits
- GPU resource tracking
- budget enforcement
- retries
- mid-training monitoring
- early stopping / pruning
- adaptive scheduling
- model selection
- export workflow
- decision trace logging

The orchestrator should not train models directly.

It should coordinate workers that train models.

---

### 2. Agent Layer

Agents should be specialized and structured.

Use the OpenAI Agents SDK, LangGraph, or another agent framework if helpful, but keep the project value in your orchestration logic.

Agents should return validated JSON outputs.

Agents should not directly mutate system state or execute training jobs.

---

### 3. GPU Workers

Workers execute training jobs.

A worker can run:

- locally
- on a cloud GPU instance
- on a rented GPU provider
- on another machine
- eventually in Docker

Workers should poll for jobs or receive jobs from the orchestrator.

Workers should report:

- status
- epoch metrics
- logs
- errors
- artifacts
- final results

---

### 4. MLflow

MLflow should be used for:

- experiment tracking
- parameter logging
- metric logging
- artifact logging
- model logging
- run comparison
- optional model registry

Do not rebuild MLflow.

Use it as the tracking backend.

---

### 5. Mission Control UI

The UI should not clone MLflow.

It should show what MLflow does not emphasize:

- agent reasoning
- decision trace
- pruning decisions
- adaptive scheduling decisions
- project status
- selected key graphs
- best model export package
- Gaming AI integration notes

---

### 6. Gaming AI Export Pipeline

The export pipeline packages the best model for use in the Gaming AI app.

Exports should include:

- model artifact
- label map
- preprocessing config
- confidence threshold
- inference script/example
- metrics summary
- integration report

---

## Specialized Agents

### 1. Dataset Profiler Agent

Purpose:

Analyze deterministic dataset statistics and identify risks.

Inputs:

- number of classes
- images per class
- class imbalance
- image dimensions
- corrupted image count
- duplicate/near-duplicate summary, optional
- train/validation/test split
- user goal

Outputs:

- task type
- dataset quality assessment
- risks
- recommended preprocessing
- recommended metric
- whether more data is needed

Example output:

```json
{
  "task_type": "image_classification",
  "risks": ["small_dataset", "class_imbalance"],
  "recommended_metric": "macro_f1",
  "recommendations": [
    "use_transfer_learning",
    "enable_augmentation",
    "consider_class_weighted_loss"
  ],
  "reasoning_summary": "The dataset is small and imbalanced, so transfer learning and macro-F1 are more appropriate than training from scratch and optimizing accuracy."
}
```

---

### 2. Experiment Planner Agent

Purpose:

Create initial experiment plans and follow-up experiment plans.

Inputs:

- dataset profile
- available model templates
- compute resources
- budget
- previous experiment results
- diagnostics results

Outputs:

- experiment configs
- priorities
- purpose of each experiment
- expected tradeoffs

Example output:

```json
{
  "experiments": [
    {
      "name": "mobilenet_transfer_augmented",
      "template": "image_mobilenet_transfer",
      "priority": 1,
      "config": {
        "image_size": 224,
        "epochs": 15,
        "learning_rate": 0.0003,
        "freeze_backbone": true,
        "augmentation": true,
        "class_weighted_loss": false
      },
      "purpose": "Establish a strong lightweight transfer-learning baseline."
    },
    {
      "name": "resnet_transfer_class_weighted",
      "template": "image_resnet_transfer",
      "priority": 2,
      "config": {
        "image_size": 224,
        "epochs": 15,
        "learning_rate": 0.0001,
        "freeze_backbone": true,
        "augmentation": true,
        "class_weighted_loss": true
      },
      "purpose": "Test whether a larger model with class weighting improves minority-class recall."
    }
  ]
}
```

---

### 3. Training Diagnostics Agent

Purpose:

Analyze completed, failed, and pruned experiments.

Inputs:

- MLflow run summaries
- training curves
- validation curves
- confusion matrix
- per-class metrics
- pruning events
- resource usage
- user constraints

Outputs:

- diagnosis
- likely causes
- next recommended experiments
- data collection recommendations
- whether to continue

Example output:

```json
{
  "diagnosis": "MobileNet generalized better than ResNet. ResNet overfit after epoch 10, while MobileNet plateaued around epoch 12.",
  "likely_causes": [
    "dataset_size_too_small_for_larger_model",
    "class_imbalance_affecting_combat_recall"
  ],
  "recommended_next_experiments": [
    {
      "template": "image_mobilenet_transfer",
      "config_changes": {
        "max_epochs": 12,
        "class_weighted_loss": true,
        "learning_rate": 0.0001
      },
      "purpose": "Improve minority-class recall while avoiding late-epoch overfitting."
    }
  ],
  "data_recommendation": "Collect more combat screenshots with varied lighting and enemy positions.",
  "continue_experimenting": true
}
```

---

### 4. Pruning Policy Agent or Rule Engine

This can begin as deterministic rules and later include agent explanations.

Purpose:

Decide when to terminate underperforming runs.

Inputs:

- current epoch metrics
- global best metrics
- peer runs at similar epochs
- training/validation gap
- metric plateau
- budget remaining

Outputs:

- continue
- terminate
- reduce max epochs
- flag for diagnostics
- reason

Example output:

```json
{
  "decision": "terminate",
  "reason": "Validation macro-F1 has not improved for 4 epochs and the train/validation gap indicates overfitting.",
  "epoch": 11
}
```

---

### 5. Export / Integration Agent

Purpose:

Generate Gaming AI integration guidance.

Inputs:

- best model
- metrics
- label map
- preprocessing config
- inference latency
- Gaming AI runtime constraints

Outputs:

- recommended confidence threshold
- runtime usage notes
- failure modes
- integration checklist
- sample inference guidance

Example output:

```json
{
  "recommended_confidence_threshold": 0.75,
  "runtime_notes": [
    "Run inference every 5-10 frames instead of every frame.",
    "Use UNKNOWN for predictions below the confidence threshold.",
    "Smooth predictions over a rolling window to avoid flickering state labels."
  ],
  "failure_modes": [
    "Dark combat scenes may be confused with exploration.",
    "Partial menu overlays may be misclassified."
  ]
}
```

---

## Training Templates

Agents should select from controlled templates.

Do not allow unrestricted arbitrary model code generation in the MVP.

### Initial Templates

Start with:

- simple CNN baseline
- MobileNet transfer learning
- ResNet transfer learning
- EfficientNet transfer learning, optional

### Later Templates

Add later:

- ViT fine-tuning
- YOLO/object detection
- temporal/video classification
- lightweight inference-optimized models

---

## Distributed Training Design

The project should support multiple workers training different experiments concurrently.

Example:

```text
Planner creates 5 experiment configs
↓
Orchestrator schedules jobs
↓
Worker A trains MobileNet
Worker B trains ResNet
Worker C trains EfficientNet
Worker D trains Simple CNN
Worker E trains MobileNet with class weighting
↓
Workers stream epoch metrics back
↓
Orchestrator prunes weak runs
↓
Freed workers receive new tasks
```

This is a key feature.

The platform should aim to turn long sequential experimentation into shorter parallel experimentation.

Example goal:

> Run a batch of 5 experiments in parallel across GPU workers and reduce an hour-long local sweep into a 10-15 minute distributed run.

Actual timing depends on model size, dataset size, GPU availability, networking, and worker startup time.

---

## Worker Communication

Workers should report regularly.

At minimum, after each epoch:

```json
{
  "job_id": "job_123",
  "epoch": 7,
  "train_loss": 0.31,
  "val_loss": 0.48,
  "train_accuracy": 0.93,
  "val_accuracy": 0.81,
  "macro_f1": 0.78,
  "per_class_recall": {
    "combat": 0.62,
    "exploration": 0.85,
    "menu": 0.91
  },
  "status": "running"
}
```

The orchestrator stores these updates and may publish global signals.

---

## Early Stopping and Pruning

This is one of the most important differentiators.

The orchestrator should be able to terminate bad experiments before completion.

### Pruning examples

Terminate if:

- validation loss worsens for N epochs
- validation macro-F1 plateaus
- train accuracy rises while validation score drops
- model is clearly worse than peers at same epoch
- minority-class recall is not improving
- model exceeds latency or size constraints
- budget is nearly exhausted and run is unlikely to win

### Adaptive global signals

The orchestrator can learn from workers.

Example:

```json
{
  "global_signal": "recommended_max_epochs",
  "value": 12,
  "reason": "Most transfer-learning runs plateau between epochs 10 and 12 and overfit after epoch 14."
}
```

Future jobs can use this updated setting.

Running jobs may check for signals at the end of each epoch.

---

## Shared Learning Across Workers

Workers should not directly coordinate with each other.

Instead:

```text
Worker → reports metrics → orchestrator/state store → planner/diagnostics reads summary → new policy/configs created
```

Examples of shared learning:

- reduce max epochs for future runs
- prefer MobileNet over heavier models for small datasets
- enable class-weighted loss after class imbalance is detected
- avoid model families that consistently overfit
- adjust learning rates based on validation curves
- add augmentation when generalization is poor

---

## Live Inference Strategy

Cloud inference is not the primary MVP.

For live Gaming AI inference, local inference is preferred because cloud inference introduces latency:

```text
capture frame
↓
upload image
↓
cloud inference
↓
return prediction
```

This can be too slow for real-time game assistance.

The better plan:

```text
Train on cloud GPU workers
↓
Export optimized model
↓
Run inference locally inside Gaming AI
```

### Local inference optimization ideas

- use MobileNet/EfficientNet-lite style models
- downsample screenshots
- run inference every N frames
- use ONNX or TorchScript
- use TensorRT later for NVIDIA optimization
- confidence thresholding
- prediction smoothing across recent frames
- return UNKNOWN for low-confidence predictions

---

## Model Export Format

Each exported model package should include:

```text
export/
  best_model.pt
  model.onnx                 # optional
  label_map.json
  preprocessing.json
  metrics.json
  inference_example.py
  integration_report.md
```

Example prediction output:

```json
{
  "prediction": "combat",
  "confidence": 0.83,
  "all_scores": {
    "combat": 0.83,
    "exploration": 0.12,
    "menu": 0.05
  }
}
```

---

## MLflow Logging Plan

Each run should log:

### Params

- model family
- template name
- image size
- batch size
- epochs
- learning rate
- optimizer
- augmentation enabled
- class weighting enabled
- freeze backbone
- dataset id
- split seed
- worker id
- GPU type, if known

### Metrics

- train loss per epoch
- validation loss per epoch
- train accuracy per epoch
- validation accuracy per epoch
- macro-F1 per epoch
- per-class precision
- per-class recall
- per-class F1
- inference latency
- train/validation gap
- best validation metric
- epoch of best validation metric

### Artifacts

- trained model
- label map
- preprocessing config
- confusion matrix
- training curve plot
- validation curve plot
- classification report
- sample predictions
- final export package

---

## Mission Control UI

The UI should be focused and product-specific.

It should not try to replace MLflow.

### UI should show:

- project status
- dataset summary
- active workers
- running jobs
- pruned jobs
- agent decisions
- early stopping/pruning decisions
- iteration timeline
- selected key graphs
- current best model
- export package
- Gaming AI integration notes

### UI should not focus on:

- full MLflow run comparison
- advanced artifact browsing
- model registry replacement

### Useful screens

#### 1. Project Overview

- project name
- dataset classes
- target metric
- active workers
- jobs running
- best model so far
- budget remaining

#### 2. Dataset Panel

- class distribution
- sample images
- imbalance warnings
- profiler agent summary

#### 3. Distributed Jobs Panel

- worker id
- job id
- model family
- current epoch
- status
- latest validation score
- prune risk

#### 4. Decision Timeline

Example:

```text
Dataset has class imbalance.
↓
Planner selected transfer learning.
↓
5 experiments launched across GPU workers.
↓
ResNet run pruned at epoch 11 due to overfitting.
↓
MobileNet run plateaued at epoch 12.
↓
Orchestrator updated max epoch recommendation to 12.
↓
Follow-up experiment launched with class-weighted loss.
↓
Combat recall improved by 9%.
```

#### 5. Key Graphs Panel

Show:

- train vs validation loss
- train vs validation accuracy
- macro-F1 over epochs
- confusion matrix
- per-class recall
- leaderboard summary

#### 6. Export Panel

- selected model
- label map
- preprocessing config
- confidence threshold
- inference example
- integration notes

---

## Database Concepts

Use your own database for orchestration state and summaries.

Use MLflow for detailed experiment runs.

### Project

- id
- name
- game_name
- classifier_goal
- target_metric
- status
- created_at
- updated_at

### Dataset

- id
- project_id
- dataset_path
- class_names
- profile_json
- created_at

### AgentRun

- id
- project_id
- agent_name
- input_json
- output_json
- status
- created_at

### Worker

- id
- name
- status
- backend_type
- gpu_type
- last_heartbeat
- current_job_id

### ExperimentJob

- id
- project_id
- worker_id
- mlflow_run_id
- template_name
- config_json
- status
- priority
- prune_reason
- created_at
- started_at
- completed_at

### EpochMetric

- id
- job_id
- epoch
- metrics_json
- created_at

### DecisionTrace

- id
- project_id
- event_type
- summary
- metadata_json
- created_at

### ModelExport

- id
- project_id
- job_id
- model_path
- label_map_path
- preprocessing_path
- report_path
- created_at

---

## Recommended Tech Stack

### Backend

- Python
- FastAPI
- PostgreSQL or SQLite for first MVP
- SQLAlchemy
- Alembic
- Redis for queue/pub-sub if needed
- MLflow
- PyTorch
- torchvision
- Pillow/OpenCV
- Pydantic for schemas
- OpenAI Agents SDK or equivalent for agents

### Worker

- Python worker process
- PyTorch training templates
- MLflow client
- heartbeat reporting
- epoch metric reporting
- cancel/prune signal checking

### Frontend

- React
- TypeScript
- Tailwind
- Recharts or Plotly
- Electron/Tauri optional later

### Deployment / Compute

Start local.

Then add remote worker support.

Possible remote worker targets:

- cloud GPU VM
- RunPod/Lambda Labs style provider
- second machine
- university GPU machine
- Docker container on GPU host

---

## Worker Execution Model

Start simple.

### Phase 1 worker

- local worker process
- one job at a time
- reports epoch metrics
- checks for cancellation

### Phase 2 workers

- multiple local/remote workers
- orchestrator assigns jobs
- workers heartbeat
- workers fetch job configs
- workers log to MLflow

### Phase 3 workers

- Dockerized workers
- cloud GPU workers
- resource-aware scheduling

---

## Orchestration State Machine

### Project states

```text
CREATED
DATASET_UPLOADED
PROFILED
PLANNING
JOBS_QUEUED
RUNNING
ANALYZING
ITERATING
EXPORTING
COMPLETED
FAILED
CANCELLED
```

### Job states

```text
PENDING
QUEUED
ASSIGNED
RUNNING
PRUNED
SUCCEEDED
FAILED
RETRYING
CANCELLED
```

### Worker states

```text
IDLE
ASSIGNED
RUNNING
HEARTBEAT_MISSED
OFFLINE
```

---

## Build Phases

### Phase 0: Project Design

Decide:

- first Gaming AI classifier goal
- dataset folder format
- model templates
- agent output schemas
- pruning rules
- MLflow logging format
- worker API
- UI screens

Deliverable:

- architecture diagram
- basic README
- schema draft

---

### Phase 1: Single-Machine Baseline

Build:

- image dataset loader
- dataset profiler
- MobileNet or simple CNN training template
- MLflow logging
- metric computation
- model artifact saving

Goal:

- manually train one classifier and log results to MLflow

---

### Phase 2: Orchestrated Local Jobs

Build:

- FastAPI backend
- project creation
- experiment job creation
- local worker
- job status tracking
- epoch metric reporting
- cancellation/pruning signal support

Goal:

- orchestrator can run multiple queued experiments locally

---

### Phase 3: Agent Planning

Build:

- Dataset Profiler Agent
- Experiment Planner Agent
- structured JSON outputs
- config validation
- automatic job creation from agent plan

Goal:

- agent creates a valid experiment batch

---

### Phase 4: Distributed Worker Support

Build:

- worker registration
- worker heartbeat
- job assignment API
- remote worker polling
- MLflow run linkage
- resource metadata

Goal:

- multiple workers can train different models concurrently

---

### Phase 5: Early Pruning and Adaptive Scheduling

Build:

- epoch-level metric monitoring
- pruning rules
- global signals
- worker cancellation
- job reassignment
- decision trace entries

Goal:

- bad jobs can be stopped early and workers can receive new jobs

---

### Phase 6: Diagnostics Loop

Build:

- summarize completed/pruned MLflow runs
- Diagnostics Agent
- follow-up experiment generation
- stopping condition

Goal:

- system runs at least one feedback-driven iteration

---

### Phase 7: Model Export for Gaming AI

Build:

- best model selection
- export package
- label map
- preprocessing config
- inference example
- integration report

Goal:

- Gaming AI app can consume the exported model locally

---

### Phase 8: Mission Control UI

Build:

- project overview
- dataset panel
- active workers panel
- jobs panel
- decision timeline
- key graphs
- export panel

Goal:

- polished enough for LinkedIn demo and resume screenshots

---

### Phase 9: Polish

Build:

- README
- architecture diagram
- demo video
- screenshots
- resume bullets
- LinkedIn post
- small curated demo dataset

---

## Definition of Done for Resume-Ready MVP

The project is resume-ready when it can:

- accept a labeled gameplay screenshot dataset
- profile the dataset
- use an agent to generate an experiment plan
- launch at least 3 model training jobs
- log all runs to MLflow
- stream epoch metrics to the orchestrator
- prune at least one underperforming/overfitting run
- launch at least one follow-up experiment based on diagnostics
- select a best model
- export the model for Gaming AI local inference
- show agent decisions and pruning events in a UI
- explain clearly why this is more than MLflow/AutoML

---

## Resume Positioning

Potential project title:

**Distributed Agentic Vision Training Platform**

Alternative names:

- VisionForge
- GameVision Trainer
- Agentic Vision Lab
- ModelForge Vision
- GameSense Trainer

Potential resume bullets:

- Built a distributed agentic computer vision training platform that dispatches image-classification experiments across GPU workers and uses MLflow to track metrics, artifacts, and model outputs.
- Designed an orchestration engine for scheduling, monitoring, pruning, and reassigning ML training jobs based on mid-training validation metrics and resource constraints.
- Implemented specialized agents for dataset profiling, experiment planning, training diagnostics, and Gaming AI model export using structured outputs and validated experiment configs.
- Added adaptive experiment pruning to terminate overfitting or underperforming runs early, freeing GPU workers and reducing total experiment time.
- Exported optimized classifier packages with model artifacts, label maps, preprocessing configs, and inference examples for low-latency local use in a Gaming AI application.

---

## Gaming AI Project Positioning

Separate project bullets for the Gaming AI app:

- Built a Gaming AI vision pipeline that uses exported classifiers to detect game states and UI conditions from gameplay screenshots.
- Integrated trained vision models with confidence thresholding and prediction smoothing for local real-time inference.
- Collected and labeled custom gameplay datasets and used a distributed agentic training platform to improve classifier performance through iterative experimentation.

---

## LinkedIn Demo Angle

Possible demo caption:

> I built a distributed agentic vision training platform for my Gaming AI project. It launches multiple CNN/transfer-learning experiments across GPU workers, streams epoch metrics, prunes overfitting runs early, uses agents to diagnose weak classes and propose follow-up experiments, logs everything to MLflow, and exports the best model for local real-time inference.

Demo sequence:

1. Upload labeled gameplay screenshot dataset
2. Agent profiles dataset
3. Agent plans 5 experiments
4. Workers start training concurrently
5. UI shows epoch metrics
6. One model starts overfitting and gets pruned
7. Worker receives a new task
8. Diagnostics agent recommends class-weighted loss / augmentation
9. Follow-up model improves weak class recall
10. Best model is exported for Gaming AI

---

## Important Tradeoffs to Explain in Interviews

### Why not cloud inference?

Cloud inference adds network latency and image upload overhead. For live game assistance, local inference is better.

### Why cloud/distributed training?

Training and evaluation are batch workloads and can be parallelized across GPU workers.

### Why use MLflow?

MLflow is strong for tracking and artifacts. Rebuilding it would be wasted effort.

### Why build a custom orchestrator?

MLflow records runs, but it does not decide how to adapt experiments for a specific Gaming AI use case, prune workers mid-run, or export models into the application workflow.

### Why agents?

Agents are used for high-level reasoning: profiling, planning, diagnostics, and integration guidance. Deterministic code handles training, metrics, state, and execution.

### Why controlled templates instead of arbitrary generated code?

Controlled templates are safer, reproducible, easier to debug, and better for reliable training.

---

## Risks and Mitigations

### Risk: Scope explosion

Mitigation:

- start with image classification only
- delay video, object detection, and ViT until core system works

### Risk: Cloud compute cost

Mitigation:

- start with local workers
- simulate multiple workers
- add remote GPU support later
- use small datasets and short training runs for demo

### Risk: Weak differentiation from AutoML

Mitigation:

- emphasize distributed orchestration
- early pruning
- adaptive scheduling
- Gaming AI integration
- decision traces

### Risk: Agents make bad recommendations

Mitigation:

- validate all outputs
- restrict to supported templates
- use deterministic safety checks

### Risk: Training takes too long

Mitigation:

- use transfer learning
- support pruning
- use small curated datasets
- limit epochs
- use early stopping

### Risk: UI takes too much time

Mitigation:

- build minimal mission-control UI
- rely on MLflow for detailed run inspection

---

## Questions to Answer Before Coding

1. What exact game-state classifier should be built first?
2. What game or dataset will be used for the first demo?
3. What classes are visually distinct enough for the MVP?
4. How many images per class can be collected quickly?
5. Which model template should be implemented first?
6. How should workers communicate with the orchestrator?
7. What pruning rule should be implemented first?
8. What metrics should determine the best model?
9. What should the export package look like?
10. How will the Gaming AI app consume the exported model?
11. What should the UI show in the first demo?
12. How can remote workers be added without overengineering?

---

## Notes for Codex / Coding Assistant

When helping build this project:

1. Prioritize a working vertical slice over broad scaffolding.
2. Start with image classification only.
3. Do not add tabular data in the MVP.
4. Do not build object detection or video classification first.
5. Use MLflow for tracking instead of rebuilding tracking infrastructure.
6. Use controlled PyTorch templates instead of arbitrary generated code.
7. Keep agents narrow and structured.
8. Validate all agent outputs with schemas.
9. Build worker reporting and pruning early because it is a key differentiator.
10. Keep cloud compute optional until the local distributed workflow works.
11. Make the architecture easy to explain for SWE, ML, and systems interviews.
12. Always preserve the distinction: agents reason, orchestrator controls, workers execute, MLflow tracks, Gaming AI consumes.

