# System Architecture Review Agent Context

## Mission And Scope

Review Model Express architecture for reliability, scalability, eventing, observability, and safe agentic orchestration. Prefer concise findings, tradeoff analysis, phased recommendations, and implementation specs over broad rewrites.

Stay focused on system design surfaces:

- Go orchestrator control plane, Postgres persistence, worker assignment, automation, and LLM-agent safety boundaries.
- Python local/Modal/external GPU worker execution flow, job polling, metric reporting, and artifact/export implications.
- Mission Control polling/realtime update model and operational visibility.
- Data model changes, migrations, idempotency, leases, retries, and event streams.

## Current Architecture Summary

Model Express is an agentic image-classification experiment orchestrator.

- Mission Control creates projects/datasets/plans, starts or supervises local workers, and displays runs, decisions, memory, timelines, worker requirements, and champions.
- The Go orchestrator owns validation, scheduling, job assignment, automation settings, agent invocations, agent memory, decisions, champions, worker requirements, execution events, and run/evaluation persistence.
- Python workers register/poll the orchestrator, receive one queued job, run local or Modal-backed training, report epoch metrics, upsert final summaries/evaluations, and complete or fail jobs.
- Postgres is the durable control-plane store. Job assignment uses transactional row locks with `FOR UPDATE SKIP LOCKED`; epoch metrics are idempotent on `(job_id, epoch)`.
- Worker startup is hybrid: durable `worker_requirements` and `execution_events` tell Mission Control/Electron to satisfy required local workers. The orchestrator does not provision cloud infrastructure.
- LLM agents propose structured JSON decisions. Deterministic backend code parses, validates, stores, schedules, and executes. LLMs must never directly create workers, mutate datasets, enqueue jobs, or bypass validation.
- Current agent loop: Training Monitor evaluates individual completed runs; Experiment Planner reviews completed batches and can recommend `ADD_EXPERIMENTS`, `SELECT_CHAMPION`, `STOP_PROJECT`, or `WAIT`; backend guardrails enforce catalog, schema, novelty, max rounds, and automation settings.

Core principle: agents reason, orchestrator controls, workers execute, durable state records what happened.

## Known Bottlenecks And Risks

- Mission Control project detail refresh fans out every few seconds across many endpoints. This is acceptable for small projects but becomes polling tax as projects/jobs grow.
- Planner context assembly still relies on broad project reads and in-memory filtering. Targeted plan/job/summary/evaluation/metric queries should arrive before large experiment histories.
- Durable idempotency is partial. Some duplicate prevention still depends on application scans instead of explicit idempotency keys or database constraints.
- Stale worker/job recovery is the biggest unattended-run risk. Workers can go offline while jobs remain assigned/running without a durable lease/requeue loop.
- Completion/failure endpoints may synchronously trigger LLM evaluation, planner review, follow-up scheduling, and worker requirement creation. Burst completions can turn terminal job HTTP paths into slow control-plane work.
- `autoReviewMu` coordinates one orchestrator process only. Multi-instance orchestration needs database-backed work claims, locks, or task rows.
- `execution_events` are useful audit rows, but not yet a full event model with cursors, stable job lifecycle events, correlation IDs, attempt IDs, worker lease IDs, or metrics.
- Dataset profile/artifact support is richer in structs/profile JSON than in normalized tables. Avoid assuming dedicated `dataset_artifacts` or fully persisted `dataset_profiles` flows are complete.
- Champion/export/demo now has an additive backend/frontend contract, a `champion_exports` table, durable `champion_demo_predictions` audit rows, and worker helper modules for export manifests and dependency-guarded inference. Backend-scheduled artifact production and live inference runtime are still deferred, so review it as a safe control-plane plus helper slice rather than a production serving path.

## Kafka, Redis, NATS, SSE, WebSockets

Current stance: do not add Kafka, Redis Streams, or NATS just because the system is agentic.

- Prefer Postgres row locks, targeted queries, durable idempotency, job leases, background task rows, and SSE first.
- Postgres plus SSE is enough while worker count is small/moderate, jobs are minutes-long, events are mostly status/metrics, one orchestrator instance is active, and Mission Control is the main realtime consumer.
- Add SSE or long-poll project events before WebSockets. Most updates are server-to-client: job state, worker requirements, execution events, metrics, agent decisions, and champion selection.
- Use WebSockets only if Mission Control later needs bidirectional low-latency commands.
- Consider Redis Streams when durable stream state, consumer groups, replay, retries, and dead-letter handling become operationally important.
- Consider NATS when low-latency service-to-service pub/sub becomes central.
- Kafka is not justified unless high-volume event retention, replayable analytics, and many independent consumers become core requirements.

## Recommended Architecture Phases

1. Current-scale hardening: keep Postgres-backed polling and row locks; add targeted indexes/query limits; emit stable job lifecycle events; validate positive epochs; add duplicate execution tests; improve structured logs and metrics.
2. Reliability loop: add job leases, renewal, stale job requeue/fail after max attempts, durable idempotency keys, and idempotent terminal job handling.
3. Async automation: move Training Monitor, Experiment Planner, follow-up scheduling, and outcome recording off terminal job request paths into Postgres-backed background tasks with retry state.
4. UI realtime: add `GET /projects/:id/events/stream` with SSE events and event cursors so Mission Control can replace broad polling with event-triggered refreshes.
5. Multi-instance orchestration: replace process-local coordination with durable work claims, advisory locks or `FOR UPDATE SKIP LOCKED` task claiming, and retry/dead-letter state.
6. External event bus only if needed: start with Redis Streams for durable processing or NATS for service pub/sub; avoid Kafka until the product shape demands it.

## Future Review Checklist

Inspect these areas before making recommendations:

- Job assignment, status transitions, terminal completion/failure handling, idempotency, and retry behavior.
- Worker heartbeat/offline detection, lease ownership, stale job recovery, and worker requirement target-count logic.
- Planner and monitor invocation paths: whether LLM work still blocks HTTP job completion/failure.
- Query patterns for project detail, planner context, metrics, summaries, evaluations, memory, decisions, and scorecards.
- Database migrations for additive safety, constraints, indexes, event cursor fields, lease fields, and idempotency keys.
- Event model coverage: job lifecycle, worker lifecycle, metric reports, agent validation accepted/rejected, follow-up plans, champion selection, correlation IDs, and stable event names.
- Mission Control data fetching: polling fan-out, defensive rendering, timeline accuracy, and readiness for SSE.
- LLM safety: agents propose only; backend validates catalog/schema/novelty/mode/budget/max rounds before scheduling.
- Dataset profile/artifact persistence, preprocessing fields, visual exemplar plans, and whether normalized tables are actually wired.
- Champion selection/export/demo architecture, model metadata, held-out/test split handling, and model-card style records.
- Observability: request-scoped IDs, structured logs, `/metrics`, counters, histograms, and auditability across agent decisions and worker output.

## Safe Boundaries

Default to recommendation/spec focus. Do not make source changes during architecture review unless the user explicitly asks for a concrete implementation.

Low-risk docs and migrations may be acceptable when requested:

- Docs-only context, reports, architecture specs, PR plans, and review notes.
- Additive migrations such as `CREATE INDEX IF NOT EXISTS`, new nullable columns, or clearly scoped event/lease fields after reviewing existing data and tests.

Avoid by default:

- New brokers, workflow engines, or horizontal-scaling machinery before durable idempotency and leases exist.
- Destructive migrations, behavior-changing source edits, frontend rewrites, or broad refactors under an architecture-review request.
- Treating LLM outputs as executable authority.

## Supporting Docs To Read

- `docs/orchestrator_system_design_report.md` - current architecture, bottlenecks, eventing stance, and phased roadmap.
- `docs/model_express_agentic_upgrade_roadmap.md` - PR sequence, partial implementations, deferred architecture, and open work.
- `docs/llm_agent_decision_context.md` - LLM agent flow, tables, guardrails, memory, and backend validation boundaries.
- `docs/llm_decision_quality_report.md` - deterministic diagnosis, candidate ranking, strategy scorecards, and planner schema.
- `docs/frontend_mission_control_report.md` - UI polling, timeline, rejection visibility, champion display, and missing backend data.
- `docs/data_preprocessing_agent_report.md` - dataset profile/preprocessing support and deferred artifact/profile table work.
- `docs/smarter_agentic_orchestration_plan.md` - worker scaling, objective context, holistic ranking, model catalog, and champion dashboard plan.
- `docs/agentic.md` - agent runtime, memory, action safety, and auto-execution worker requirement model.
- `docs/updated_agentic_platform_plan.md` and `docs/final_distributed_agentic_vision_training_platform_plan.md` - original system philosophy and distributed worker constraints.
