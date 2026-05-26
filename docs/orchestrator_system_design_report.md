# Orchestrator System Design Report

## Scope

This report reviews the current Model Express orchestration path across the Go orchestrator, Python worker, Postgres persistence, Mission Control polling, local/Modal execution flow, LLM agent decision loop, epoch metric ingestion, failure recovery, idempotency, concurrency safety, and observability.

The key safety boundary is already correct: LLM agents propose structured decisions, while deterministic backend code validates, stores, schedules, and executes. LLMs do not directly create workers, mutate datasets, enqueue jobs, or bypass backend validation.

## Implementation Status Update

Since this report was first written, several recommended current-scale hardening items have landed:

- Job leases now track attempt count, owner, expiry, and last heartbeat; expired assigned/running jobs are requeued or failed on poll/manual recovery.
- `GET /projects/:id/events/stream` now exposes durable `execution_events` over SSE as a Mission Control refresh hint.
- Champion export and demo prediction work now flows through backend-scheduled worker jobs and validated worker result callbacks.
- Worker-generated visual exemplar metadata can be persisted through a backend-capped `POST /datasets/:id/visual-exemplars` profile merge.

Remaining reliability work is narrower: durable DB idempotency keys, a standalone lease-recovery ticker, moving LLM automation off terminal job HTTP paths, richer event names/cursors, and production artifact storage.

## Current System Summary

Model Express currently uses a pragmatic request/response orchestrator:

- Mission Control creates projects, registers datasets, creates plans, executes plans, displays metrics, and supervises worker requirements.
- The orchestrator stores projects, datasets, plans, jobs, workers, run summaries, evaluations, agent invocations, agent memory, decisions, champions, strategy scorecards, worker requirements, and execution events in Postgres.
- Workers poll `/workers/:id/poll`, receive one queued job, run local or Modal-backed training, report epoch metrics, upsert final run summaries/evaluations, and complete or fail the job.
- Job assignment is transaction protected with `FOR UPDATE SKIP LOCKED`, which is the right simple primitive for current scale.
- Epoch metrics are idempotent on `(job_id, epoch)` through `ON CONFLICT`.
- Automatic follow-up scheduling is serialized in-process with `autoReviewMu` and guarded by backend validation, maximum follow-up rounds, agent mode, and automation settings.
- Worker startup is coordinated through durable `worker_requirements` plus `execution_events`; Mission Control polls requirements and starts Electron-managed local workers.

This is a good architecture for a desktop-first prototype and small concurrent experiment batches. The main bottlenecks are not exotic distributed-systems problems yet. They are repeated broad reads, incomplete durable idempotency, stale worker/job recovery, and polling-based UI updates.

## Observed Strengths

- Job assignment uses row locks and `SKIP LOCKED`, so multiple workers should not claim the same queued job.
- Metric ingestion is naturally retryable because epoch metrics upsert by primary key.
- Worker requirements are durable and unique per `(project_id, plan_id)`.
- Automatic review and planning paths use backend validation before scheduling.
- Agent invocations and memory records provide an audit trail and a future fine-tuning source.
- Strategy scorecards and planning outcome memory close the loop between LLM recommendations and actual follow-up results.
- Mission Control and manual execution share the same backend plan execution path.

## Pain Points And Bottlenecks

### Repeated Broad Reads

Mission Control refreshes project detail every 2.5 seconds by issuing many parallel requests: datasets, jobs, plans, summaries, evaluations, champion, decisions, workers, requirements, execution events, memory, and scorecards. For a small project this is fine. With many projects or hundreds of jobs, this becomes a polling fan-out tax.

The planner input path also loads broad project-level lists and then filters in memory:

- all project plans
- all project jobs
- all project summaries
- all evaluations
- metrics for each plan job
- recent memory
- scorecards

That is acceptable while plans are small, but it will become the main planner bottleneck before worker assignment does.

### Partial Durable Idempotency

Some operations are idempotent in practice but rely on application-level scans:

- `executeStoredExperimentPlan` lists project jobs and matches by `plan_id`, provider, and `experiment_index` from JSON config.
- `ensureFollowUpPlan` lists all project plans and searches for `source_decision_id`.
- planner outcome recording checks invocation downstream outcome before writing outcome memory.

These are reasonable for now, but concurrent orchestrator instances or retry storms could still create duplicates unless uniqueness moves into database constraints or explicit idempotency keys.

### Stale Worker And Job Recovery

Workers are marked offline when scanned if heartbeat age exceeds the limit, but there is no durable recovery loop that requeues jobs owned by dead workers. If a worker dies after a job is assigned and before reporting failure, the job can remain assigned or running indefinitely with `workers.current_job_id` still set.

This is the biggest reliability gap for unattended runs.

### Blocking LLM Work In Request Path

Completing or failing a training job can synchronously trigger:

- training monitor LLM evaluation
- planner outcome recording
- experiment planner LLM call
- follow-up scheduling
- automatic execution/worker requirement creation
- deterministic fallback review

The code uses timeouts for LLM calls, but the HTTP completion request can still carry too much orchestration work. At current scale this is manageable. As training jobs finish in bursts, this can turn completion endpoints into slow or failure-prone control-plane operations.

### In-Process Automation Lock

`autoReviewMu` is useful inside a single orchestrator process. It does not protect against duplicate planning if multiple orchestrator instances run against the same Postgres database. That is fine for the current deployment shape, but it is a future horizontal-scaling boundary.

### Weak Event Model For UI Freshness

`execution_events` are a useful audit log, but the UI discovers them by polling. There is no event cursor, long poll, SSE, or WebSocket stream. This causes repeated full refreshes and prevents low-latency updates without extra request load.

### Observability Is Mostly Event Rows And Logs

Execution events capture important lifecycle facts, but there is no consistent request correlation ID, job attempt ID, worker lease ID, structured orchestrator logs, or metrics endpoint. Debugging a failed autonomous loop still requires piecing together rows, logs, and worker output.

## Implement Now: Low-Risk Improvements

These changes fit the current architecture and do not require new infrastructure.

### Implemented In This PR Boundary

Added additive Postgres indexes in `services/orchestrator/internal/store/migrations/001_init.sql`:

- `idx_experiment_jobs_project_status_created_at` for worker assignment and project job status scans.
- `idx_experiment_plans_source_decision_id` for follow-up plan lookup by decision.
- `idx_agent_decisions_project_plan_created_at` for plan-scoped decision lookup.
- `idx_worker_requirements_project_status` for Mission Control requirement supervision.
- `idx_agent_invocations_project_dataset_agent_created_at` for filtered invocation history.
- `idx_agent_memory_project_dataset_kind_created_at` for planner/training monitor memory retrieval.

These are low-risk because they are additive `CREATE INDEX IF NOT EXISTS` statements and do not change table shape or behavior.

### Recommended But Not Implemented Here

- Add explicit validation that reported epoch numbers are positive at the API or store boundary.
- Add structured execution events for job assignment, job running, job completion, and job failure, not just automatic execution and agent events.
- Add request-scoped log fields: `project_id`, `plan_id`, `job_id`, `worker_id`, `decision_id`, `invocation_id`.
- Add a bounded `limit` parameter to project jobs, plans, summaries, and evaluations endpoints, plus optional `updated_since`.
- Add endpoint-specific queries such as `ListPlanJobs`, `ListPlanTrainingRunSummaries`, `ListPlanTrainingRunEvaluations`, and `ListPlanMetrics` instead of loading entire project history for planner input.
- Add a job completion guard that treats repeat completion/failure as idempotent if the terminal state and payload match.
- Add tests around duplicate execution attempts for the same plan/provider/experiment index.

## Implement Soon: Reliability And Scalability Changes

These changes are medium effort and should come before adding message brokers.

### Durable Worker Leases And Requeue

Introduce job lease fields:

- `lease_owner_worker_id`
- `lease_expires_at`
- `attempt`
- `max_attempts`
- `last_heartbeat_at`

Assignment should set a lease deadline. Workers should renew while running. A periodic backend recovery task should requeue assigned/running jobs whose lease expired, or mark them failed after max attempts.

This solves abandoned assigned jobs without needing Redis or Kafka.

### Database-Level Idempotency

Add generated or explicit idempotency keys:

- Jobs: `(plan_id, provider, experiment_index)` for training jobs.
- Follow-up plans: `source_decision_id` when non-empty.
- Planner outcome memory: `(invocation_id, kind, plan_id)` for planning outcomes.
- Agent decisions: `(project_id, plan_id, decision_source)` or a source invocation key when LLM-backed.

Some of these require schema columns because key fields currently live inside JSON payload/config. Do not add fragile uniqueness over raw JSON expressions until existing data and migration risks are reviewed.

### Move Automation Off Completion Request Path

Keep completion endpoints fast:

1. Mark job terminal.
2. Write run summary.
3. Append a durable domain event such as `job.completed`.
4. Return HTTP response.
5. Let an internal automation worker process agent review/planning work asynchronously.

This can initially be an in-process goroutine with a Postgres-backed event table and retry state. It does not require a broker yet.

### Planner Input Query Narrowing

Replace broad project reads with targeted store methods:

- `ListJobsForPlan(plan_id)`
- `ListTrainingRunSummariesForPlan(plan_id)`
- `ListTrainingRunEvaluationsForPlan(plan_id)`
- `ListEpochMetricsForPlan(plan_id)`
- `ListRecentPlansForProject(project_id, limit)`
- `ListRecentAgentMemory(project_id, dataset_id, kinds, limit)`

This reduces planner prompt assembly cost and keeps context windows under control.

### UI Long Poll Or SSE

Add a project event stream before considering WebSockets:

- `GET /projects/:id/events/stream`
- Server-sent events for execution events, job state changes, worker requirement changes, and metric updates.
- Mission Control can keep a single stream open and fetch full detail only when an event says the visible resource changed.

SSE is enough because most updates are server-to-client. WebSockets are only needed if Mission Control later needs bidirectional low-latency commands.

### Observability Baseline

Add:

- `/metrics` for Prometheus-style counters and histograms.
- counters for job assignment, completion, failure, LLM invocation, validation failure, follow-up scheduled, worker requirement state changes.
- histograms for poll latency, job queue wait, job runtime, LLM latency, planner input assembly latency.
- structured logs with stable IDs.

## Future Architecture

Do not add Kafka, Redis Streams, or NATS just because the system is agentic. Current scale is better served by Postgres, row locks, targeted queries, and SSE.

### When Postgres Plus SSE Is Enough

Use Postgres plus SSE while:

- worker count is small to moderate
- jobs are minutes-long training runs
- event volume is mostly status and epoch metrics
- there is one orchestrator instance
- Mission Control is the main realtime consumer

Recommended event stream:

- channel/resource: `project_events`
- SSE event names:
  - `job.queued`
  - `job.assigned`
  - `job.running`
  - `job.metric_reported`
  - `job.succeeded`
  - `job.failed`
  - `worker.registered`
  - `worker.heartbeat`
  - `worker.offline`
  - `worker_requirement.pending`
  - `worker_requirement.starting`
  - `worker_requirement.active`
  - `worker_requirement.failed`
  - `plan.created`
  - `plan.executed`
  - `agent.invocation_created`
  - `agent.recommendation_recorded`
  - `agent.outcome_recorded`
  - `champion.selected`

Suggested event payload envelope:

```json
{
  "event_id": "execution_event_123",
  "event_type": "job.metric_reported",
  "project_id": "project_1",
  "plan_id": "plan_1",
  "job_id": "job_1",
  "worker_id": "worker_1",
  "occurred_at": "2026-05-21T12:00:00Z",
  "payload": {}
}
```

### When To Consider Redis Streams Or NATS

Consider Redis Streams or NATS when:

- multiple orchestrator instances need competing consumers for automation work
- worker/job event volume exceeds comfortable Postgres polling/SSE patterns
- retries and dead-letter handling become a core operational need
- Modal/local workers should publish status without synchronous orchestrator HTTP calls
- independent services need to consume job/metric/agent events

Candidate topics/streams:

- `model_express.jobs`
- `model_express.job_metrics`
- `model_express.workers`
- `model_express.plans`
- `model_express.agent_invocations`
- `model_express.execution_events`

Candidate event types:

- `job.created`
- `job.assigned`
- `job.lease_renewed`
- `job.lease_expired`
- `job.metric_reported`
- `job.summary_reported`
- `job.evaluation_reported`
- `job.succeeded`
- `job.failed`
- `plan.created`
- `plan.execution_requested`
- `plan.completed`
- `agent.training_monitor.requested`
- `agent.training_monitor.completed`
- `agent.experiment_planner.requested`
- `agent.experiment_planner.completed`
- `agent.validation_failed`
- `followup.requested`
- `followup.created`
- `champion.selected`

Between Redis Streams and NATS:

- Redis Streams is simpler if the team wants durable stream state, consumer groups, and replay with low operational ceremony.
- NATS is better if low-latency pub/sub and service-to-service messaging become central.
- Kafka is not justified unless event volume, retention, replay, and analytics needs grow far beyond the current product shape.

### Workflow Engine

Consider a workflow engine only after the orchestration lifecycle becomes too complex for explicit state machines:

- retries with backoff across many steps
- cancellation and compensation
- human approval checkpoints
- long-running multi-day workflows
- many provider-specific training/export/deployment branches

Until then, explicit Postgres state plus small background workers is easier to reason about.

## Roadmap

### Phase 1: Current Scale Hardening

- Keep Postgres-backed polling and row locks.
- Add targeted indexes and query limits.
- Add event rows for every job lifecycle transition.
- Add positive epoch validation.
- Add duplicate execution tests.
- Improve structured logs and metrics.

### Phase 2: Reliability Loop

- Add job leases and stale job recovery.
- Add durable idempotency keys.
- Move LLM review/planning work off terminal job HTTP requests.
- Add internal automation worker loop with retry state.
- Add SSE for Mission Control project updates.

### Phase 3: Multi-Instance Orchestration

- Replace in-process `autoReviewMu` as the only coordination layer with database locks or durable work claims.
- Add explicit automation task rows and retry/dead-letter state.
- Add advisory locks or task claim queries using `FOR UPDATE SKIP LOCKED`.
- Scale orchestrator instances horizontally only after durable idempotency is in place.

### Phase 4: External Event Bus If Needed

- Start with Redis Streams if durable event processing becomes necessary.
- Consider NATS if service-to-service pub/sub becomes the primary architecture.
- Avoid Kafka unless the project needs high-volume replayable event analytics or many independent consumers.

## Proposed PR Boundary

This PR should stay focused on orchestrator/system design cleanup and observability:

- Add the system design report.
- Add low-risk Postgres indexes.
- Avoid planner/preprocessing schema changes.
- Avoid frontend changes beyond documenting polling/SSE recommendations.
- Leave durable job leases, idempotency constraints, and event streaming as follow-up implementation work.
