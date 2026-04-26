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

## Local Development

Start the local backing services:

```powershell
docker compose -f infra/compose.yaml up -d postgres minio
```

Then run the orchestrator:

```powershell
cd services/orchestrator
go run ./cmd/orchestrator
```

By default the orchestrator uses:

```text
postgres://model_express:model_express@localhost:5432/model_express?sslmode=disable
```

Override it with `DATABASE_URL` when needed.

MinIO runs at:

```text
API:     http://localhost:9000
Console: http://localhost:9001
Login:   model_express / model_express_password
```

Datasets should live in object storage, not Postgres. Postgres stores metadata
such as `storage_uri`, checksum, size, status, and profile JSON.

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

Worker registrations are scoped to one project. Start a worker with the project
it should serve:

```powershell
cd services/worker
$env:PROJECT_ID="project_1"
$env:WORKER_NAME="local-worker-1"
python -m worker.main
```

This prevents a worker launched for one experiment project from pulling queued
jobs from another project.
