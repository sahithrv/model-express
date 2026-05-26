# Orchestrator / Backend Agent Context

## Mission And Scope

Own the Go orchestrator/backend control plane for Model Express. Keep it reliable, deterministic, and easy for worker, frontend, dataset, and LLM agents to build against.

Primary scope:

- HTTP API contracts, request validation, and response shape stability.
- Postgres and in-memory store behavior, migrations, indexes, constraints, and idempotency.
- Project, dataset, plan, job, worker, run summary/evaluation, champion, automation, memory, decision, scorecard, worker requirement, and execution event state.
- Agentic loop gating: LLM outputs are parsed, validated, stored, and scheduled only through backend guardrails.
- Backend-facing follow-ups for champion export, demo inference APIs, visual exemplar APIs, SSE/eventing, durable job leases, idempotency keys, and future production artifact storage.

Do not own Python training implementation, Mission Control UI layout, or LLM prompt strategy except where backend contracts, validation, or stored context require it.

## Current Backend Responsibilities

Model Express is a distributed image-classification experiment orchestrator, not cloud GPU infrastructure. GPU workers are started externally or by Mission Control/Electron; the orchestrator manages work across already-running workers.

Current backend responsibilities:

- Start from `services/orchestrator/cmd/orchestrator/main.go`, load env, connect Postgres, and serve Gin routes on `:8080`.
- Register projects and datasets, store dataset profile JSON, and create an initial plan after profiling.
- Create experiment plans and execute them into `train_experiment` jobs.
- Register project-scoped workers, accept heartbeats, and assign queued jobs by worker polling.
- Receive epoch metrics, final training summaries, rich run evaluations, and terminal complete/fail signals.
- Persist agent invocations, distilled memory, agent decisions, strategy scorecards, project champions, automation settings, worker requirements, and execution events.
- Run Training Monitor and Experiment Planner LLM flows when enabled, with deterministic fallbacks and backend validation.
- Coordinate auto execution through durable `worker_requirements` plus `execution_events`; Mission Control satisfies local worker starts.

Core principle:

```text
Agents propose -> backend validates/stores/schedules -> workers execute -> MLflow/artifacts track model work.
```

## Major Files And Concepts

- `services/orchestrator/internal/api/router.go`: route map. Key groups: `/projects`, `/datasets`, `/plans`, `/jobs`, `/workers`, `/settings/automation`, worker requirements, execution events, agent decisions/memory/invocations, strategy scorecards, and champions.
- `services/orchestrator/internal/api/handlers.go`: main orchestration logic, validation, plan execution, auto review, planner input assembly, champion persistence, worker requirement events, objective context, model catalog, and utility helpers.
- `services/orchestrator/internal/api/handlers_test.go`: focused backend behavior coverage for validation, automation, planner outcomes, and orchestration flows.
- `services/orchestrator/internal/store/store.go`: backend store interface. Add new durable concepts here before implementing `memory.go` and `postgres.go`.
- `services/orchestrator/internal/store/postgres.go`: Postgres implementation. Job assignment uses transactions and `FOR UPDATE SKIP LOCKED`; metrics and summaries use upsert paths.
- `services/orchestrator/internal/store/memory.go`: in-memory test/dev store that should mirror Postgres semantics closely.
- `services/orchestrator/internal/store/migrations/001_init.sql`: durable schema for projects, workers, datasets, `dataset_profiles`, plans, jobs, metrics, summaries, evaluations, champions, champion exports, champion demo predictions, decisions, automation settings, worker requirements, execution events, invocations, memory, and strategy scorecards.
- `services/orchestrator/internal/projects/model.go`: project goal/status. `Goal` is converted into objective context.
- `services/orchestrator/internal/datasets/model.go`: dataset state and richer `DatasetProfile` structs. Current source of truth is still `datasets.profile` JSON, not normalized `dataset_profiles` rows.
- `services/orchestrator/internal/plans/model.go`: `ExperimentPlan`, `PlannedExperiment`, and preprocessing fields.
- `services/orchestrator/internal/jobs/model.go`: job lifecycle and epoch metric models.
- `services/orchestrator/internal/runs/model.go`: training summaries, rich evaluations, and `ProjectChampion`.
- `services/orchestrator/internal/workers/model.go`: worker status, project scope, heartbeat, and current job.
- `services/orchestrator/internal/execution/model.go`: worker requirements and execution event names.
- `services/orchestrator/internal/memory/model.go`: agent invocations and distilled memory records.
- `services/orchestrator/internal/decisions/model.go`: actionable decision types: `WAIT`, `ADD_EXPERIMENTS`, `SELECT_CHAMPION`, `STOP_PROJECT`, `REOPEN_EXPERIMENTATION`.
- `services/orchestrator/internal/strategies/model.go`: strategy scorecards and planner outcome tracking.
- `services/orchestrator/internal/agents/*`: deterministic reviewer, Training Monitor, Experiment Planner, diagnosis, candidate ranking, objective structs, and tests.
- `services/orchestrator/internal/llm/*`: OpenAI-compatible JSON generation client and agent mode/config normalization.

## Validation And Safety Boundaries

Backend validation is the execution gate.

- LLM agents may only return structured JSON. They cannot directly create jobs, mutate datasets, start workers, choose arbitrary files, or bypass validation.
- `propose` mode records recommendations; `autonomous` mode may schedule/execute only when automation settings permit it.
- Auto follow-up scheduling is currently serialized inside one process with `autoReviewMu`, bounded by `MaxFollowUpRounds`, and guarded by planner decision type, agent mode, and settings.
- Planner outputs must pass agent schema validation, planning-mode rules, experiment shape validation, model catalog checks, novelty/duplicate checks, and max experiment count.
- Supported model names are backend curated: MobileNetV3, EfficientNet B0/B1/B2, RegNet-Y-400MF, ResNet18/34, ConvNeXt Tiny, Swin-T, and ViT-B/16.
- `PlannedExperiment` validation checks template/model presence, epochs, batch size, learning rate, image size, optimizer, scheduler, weight decay, augmentation keys, augmentation policy, class balancing, sampling strategy, early stopping, fine-tune strategy, and preprocessing values.
- Duplicate experiment signatures include model/config plus preprocessing, augmentation policy, sampling strategy, and resolution strategy.
- Job assignment is Postgres transaction protected with row locks and `SKIP LOCKED`.
- Epoch metrics are idempotent at `(job_id, epoch)` through `ON CONFLICT`.
- Worker requirements are unique per `(project_id, plan_id)`.
- Project champions are upserted by unique `project_id`.

Known safety gaps to preserve in context:

- Durable idempotency is partial. Plan execution, follow-up creation, planner outcomes, and agent decisions still rely partly on application-level scans.
- Job lease/requeue support now exists through assignment lease fields, lease renewal, and poll/manual recovery; a standalone recovery ticker remains future work.
- Terminal job endpoints can still trigger LLM review/planning and follow-up work synchronously.
- `autoReviewMu` is single-process only.
- `execution_events` are durable audit rows and are exposed through `GET /projects/:id/events/stream` as SSE refresh hints; they are not a full event-sourcing model.

## Recent Work Already Done

- Dataset/profile/preprocessing support: richer `DatasetProfile` structs, worker profile fields, optional planned experiment fields for `resolution_strategy`, `preprocessing`, `augmentation_policy`, structured `augmentation_policy_config`, and `sampling_strategy`, backend validation, preprocessing-aware duplicate signatures, and worker-visible execution config.
- LLM decision quality: deterministic planner diagnosis, planning modes, candidate hypotheses, deterministic candidate ranking with `score_components`, rejected options memory, strategy scorecards, and planner outcome memory.
- Objective/champion support: project `Goal` is converted into objective context; run evaluations support per-class/confusion/model/holistic payloads plus backend `training_diagnostics`; `SELECT_CHAMPION` persists a project champion with metrics, evaluation, deployment profile, model-card-like pending export notes, and `CHAMPION_SELECTED` event. A terminal `STOP_PROJECT` without `champion_job_id` now falls back to the best successful run so the user still gets a usable champion model.
- Follow-up safety: backend novelty checks reject LLM proposals and filter deterministic follow-up payloads that repeat an already-tested experiment mechanism with only minor tuning changes such as epochs, batch size, or learning rate.
- Planner correction loop: when an LLM planner output passes JSON/schema validation but fails backend proposal validation, the backend records the rejected invocation outcome and retries the planner once with `planner_validation_feedback`; the corrected JSON must still pass the same backend validation gates before any decision/plan is stored or scheduled.
- Mechanism contract: LLM-originated `ADD_EXPERIMENTS` decisions now carry first-class mechanism metadata (`mechanism`, `intervention`, `evidence_used`, `expected_effect`) through `proposal_mechanisms`; the backend validates allowed mechanisms and copies accepted mechanism metadata onto stored `PlannedExperiment`s for API/UI/audit visibility.
- Mechanism evidence gates: follow-up plan creation and stale follow-up execution now revalidate mechanism support against backend-owned dataset/profile evidence. Class imbalance requires balancing config plus imbalance/per-class/minority evidence; bbox crop requires backend-profiled bbox/annotation evidence; high-resolution/crop mechanisms require object-scale/fine-grained/dimension/crop evidence; label-noise and hard-example mechanisms are report-only and must use the `label_quality_audit` job template; MixUp/CutMix require structured mixed-sample augmentation config; distillation remains blocked until worker/support contracts land.
- Strategy scorecards now promote mechanism outcome metadata to first-class store fields in addition to preserving `proposed_changes`: `mechanism`, `intervention`, `diagnosis_triggers`, `evidence_used`, and `expected_effect`. Store creation hydrates these fields from explicit create values or legacy `proposed_changes`/`proposal_mechanisms`, and Postgres migrations backfill existing rows where practical.
- Champion terminal guard: a persisted project champion or any persisted `SELECT_CHAMPION` decision blocks new Experiment Planner calls, autonomous/stale follow-up plan creation, and follow-up execution. Blocked attempts record `backend_stop_guard: champion_selected_guard`. Continuing requires explicit `POST /projects/:id/experimentation/reopen`, which creates `REOPEN_EXPERIMENTATION` and `EXPERIMENTATION_REOPENED`.
- Mission Control visibility: frontend can render timeline, dataset intelligence, agent reasoning, backend gate/rejection cues, champion comparison, and selected champion from existing backend payloads.
- System design cleanup: additive Postgres indexes were added for project/status scans, follow-up lookup, plan-scoped decisions, worker requirements, agent invocations, and memory retrieval.
- Champion export/demo contract: `champion_exports` table plus `GET/POST /projects/:id/champion/exports`, `POST /jobs/:id/champion-export-result`, `champion_demo_predictions` table plus `GET/POST /projects/:id/champion/demo-predictions`, `POST /jobs/:id/champion-demo-prediction-result`, `GET /projects/:id/champion/demo-images`, `GET/POST /datasets/:id/visual-exemplars`, `CHAMPION_EXPORT_REQUESTED`, `CHAMPION_DEMO_PREDICTION`, and positive epoch validation. Export requests are idempotent per project/champion/format at the store/API behavior level. Demo prediction requests queue worker inference when a `READY` export exists and record `RUNTIME_UNAVAILABLE` when artifacts/runtime are missing.

## Follow-Up Backend Work

Champion export:

- Add durable export records/API, such as `champion_exports` and export artifact metadata linked to `project_champions`.
- Validate that a selected champion job is complete and has an exportable model artifact before marking export ready.
- Store model-card/export metadata: format, labels, input size, normalization, training config, model size, expected latency, intended use, known limits, and instructions.
- Coordinate with the worker agent for actual TorchScript/ONNX/framework-native artifact production.
- Current slice records export intent/readiness and validation errors and updates duplicate same-format requests in place. Backend-scheduled artifact production remains worker/runtime-owned.

Demo APIs:

- Add backend endpoints to list held-out/test demo images, audit champion prediction attempts, and return predicted label, true label if known, confidence, top-k predictions, latency, and correctness when a worker runtime reports them.
- Prefer test/held-out split images when available; degrade gracefully when no split exists.
- Prediction audit/history now uses `champion_demo_predictions`.
- Coordinate with worker/runtime ownership for actual inference execution.
- Current slice lists capped demo images from `datasets.profile.demo_images` or `datasets.profile.visual_exemplars`, queues worker prediction jobs when possible, and persists worker result rows.

Visual exemplars:

- Add a class-balanced visual exemplar API or artifact record path for small, downscaled, compressed image examples.
- Cap exemplars by class count, total images, bytes, and prompt budget.
- Expose exemplars as optional planner context only; LLM output remains JSON and backend validation remains unchanged.
- Record whether exemplars were used in an invocation.
- Current slice reads capped `visual_exemplars` from `datasets.profile`, exposes them through API, accepts capped worker-generated exemplar patches through `POST /datasets/:id/visual-exemplars`, and injects evidence-only class summary metadata into planner context. Durable generation/audit history remains deferred.

Idempotency:

- Add durable idempotency keys or constraints for jobs by `(plan_id, provider, experiment_index)`, follow-up plans by `source_decision_id`, planner outcome memory by `(invocation_id, kind, plan_id)`, and agent decisions by source invocation/decision key.
- Avoid fragile uniqueness over raw JSON until existing data and migrations are reviewed.
- Champion export idempotency is currently behavioral rather than backed by a unique constraint to avoid unsafe migration failures on existing duplicate export rows.
- Add duplicate plan execution and idempotent terminal completion/failure tests.

SSE/eventing:

- Continue expanding stable job lifecycle execution events behind the existing `GET /projects/:id/events/stream` endpoint.
- Use SSE before WebSockets; most updates are server-to-client.
- Include event IDs/cursors and resource identifiers: project, plan, job, worker, decision, invocation.
- Mission Control should use events to reduce broad polling fan-out.

Job leases:

- Lease fields `lease_owner_worker_id`, `lease_expires_at`, `lease_last_heartbeat_at`, `attempt`, and `max_attempts` are additive migration fields.
- Assignment and metric reporting renew leases.
- Poll/manual store recovery requeues expired assigned/running jobs or fails them after max attempts.
- A standalone backend recovery ticker remains future work.

Other useful backend follow-ups:

- The reopen/new-exploration path uses existing durable primitives: a user-source `agent_decisions` row plus an `execution_events` row with a clear reopen event type/payload. No dedicated reopen table is required unless product semantics later need separate authorization, expiry, or round-scoped idempotency.
- Wire normalized `dataset_profiles` rows or explicitly keep `datasets.profile` JSON as the only source of truth.
- Add normalized rejection/validation events so failed LLM outputs are visible even without `agent_decisions`.
- Narrow broad planner/project reads with plan-scoped queries and limits.
- Move LLM automation work off terminal job HTTP request paths into durable background tasks.
- Add request-scoped structured logs and `/metrics`.

## No-Conflict Boundaries

- Worker/training agent owns Python local/Modal training, transforms, dataset normalization computation, bbox crop execution, split-file training, model artifact export, inference runtime, and exemplar generation internals. Backend owns API contracts, validation, durable records, and status transitions.
- Frontend/Mission Control agent owns UI layout, section navigation, panels, typed display fields, component/build tests, and demo UX. Backend should provide stable endpoints, events, and payload contracts.
- LLM Decision Intelligence agent owns prompt/schema reasoning quality, diagnosis/ranking strategy, memory shaping, and compact context design. Backend may enforce schemas and pass context, but should not loosen validation to accommodate unsupported LLM ideas.
- Dataset Intelligence agent owns deterministic profile contents, artifact parsing, visual sample selection policy, and preprocessing contract changes. Backend must keep source-of-truth writes deterministic and validate all execution-facing fields.
- System Architecture Review agent owns recommendations and phased design review. Orchestrator/backend agent owns implementation only when explicitly asked.

## Supporting Docs To Read

- `README.md` - repo layout, control boundary, worker lifecycle, local dev.
- `docs/orchestrator_system_design_report.md` - reliability gaps, idempotency, leases, SSE, observability, and eventing stance.
- `docs/model_express_agentic_upgrade_roadmap.md` - implemented work, PR sequence, champion export/demo, visual exemplars, and backend follow-ups.
- `docs/llm_agent_decision_context.md` - agent flow, tables, validation, memory, and decision pipeline.
- `docs/llm_decision_quality_report.md` - deterministic diagnosis, planning modes, candidate ranking, rejected options, and scorecards.
- `docs/data_preprocessing_agent_report.md` - profile/preprocessing schema, supported values, and deferred dataset/profile work.
- `docs/frontend_mission_control_report.md` - what Mission Control can render and backend data still needed.
- `docs/smarter_agentic_orchestration_plan.md` - worker scaling, objective context, holistic evaluation, model catalog, and champion dashboard.
- `docs/agentic.md` and `docs/updated_agentic_platform_plan.md` - original agent runtime, safety, and distributed worker constraints.
