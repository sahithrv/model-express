# Model Express

Distributed agentic vision training platform for gaming AI models.

The first MVP focuses on image classification for labeled gameplay screenshots:

1. Profile a dataset.
2. Plan a small batch of controlled experiments.
3. Dispatch jobs to Python GPU workers.
4. Log runs, metrics, and artifacts to MLflow.
5. Prune weak runs early from the Go orchestrator.
6. Export the best local-inference model package.

## Architecture Clarification

Model Express orchestrates experiments across already-running GPU workers. It
does not provision, start, stop, or configure cloud GPU machines.

```text
Manually started GPU workers -> register with orchestrator -> pull jobs -> train
```

The platform is distributed experiment search, not multi-GPU training of one
large model.

## Repository Layout

```text
services/orchestrator   Go control plane: API, scheduler, pruning, worker state
services/worker         Python execution plane: PyTorch training and MLflow logging
apps/mission-control    Future React UI
schemas                 Shared API and JSON schemas
configs                 Local configuration examples
infra                   Local development infrastructure
datasets                Local datasets, gitignored
artifacts               Local generated artifacts, gitignored
exports                 Local exported model packages, gitignored
docs                    Project planning docs
```

## First Build Target

Build the vertical slice before broadening the platform:

```text
Go orchestrator -> local Python worker -> MLflow run -> exported model package
```

## Control Boundary

```text
Agents recommend.
Orchestrator validates, schedules, prunes, and records decisions.
Workers execute controlled training templates.
MLflow tracks runs, metrics, and artifacts.
Memory stores reusable experiment insights.
```

Workers can run locally or on manually launched GPU hosts such as RunPod-style
instances. Each worker should run one training job at a time for the MVP.
