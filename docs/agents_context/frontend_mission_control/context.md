# Frontend / Mission Control Context

## Agent Mission And Scope

Build and maintain the Mission Control desktop UI for Model Express. The app is the operator-facing control plane for creating projects, uploading image-folder datasets, watching profiling/training progress, reviewing agent decisions, supervising workers, and explaining model/champion outcomes.

Scope is frontend-focused:

- Work primarily in `apps/mission-control`.
- Keep UI changes consistent with the existing Electron + Vite + React + TypeScript app.
- Use backend APIs as the source of truth; do not invent frontend-only state that changes orchestration semantics.
- Show agentic behavior transparently: what happened, what evidence the agents used, what was accepted/rejected, and what the backend scheduled.

## Current UI Surface

Mission Control currently has one long project workspace with a sidebar project picker and these main panels:

- Connection/status: orchestrator URL, health, refresh, selected project.
- New project modal: project name, goal, local dataset folder upload, profiling worker start.
- Summary metrics: dataset/job/plan/worker/queued counts.
- Experiment Timeline: synthesized lifecycle from project, dataset, latest plan, jobs, training monitor memory, latest planner decision, backend gate, follow-up plan, and champion.
- Dataset Intelligence: reads `Dataset.profile` plus latest decision planning insights; shows image/class counts, imbalance ratio, class distribution, image size range, corrupt images, artifacts, recommended metrics, preprocessing hints, warnings, and profile readiness.
- Automation Settings: auto review, auto follow-ups, auto execute, LLM enabled, agent mode, follow-up rounds, default provider/GPU, LLM provider/model.
- Selected Champion: chosen job/model, macro-F1, accuracy, cost, runtime, latency, model size, selection reason, confusion preview.
- Training Run Summary: run table, total estimated spend, best macro-F1, total runtime, active runs.
- Champion Comparison: completed run comparison across macro-F1, accuracy, runtime, cost, latency, size, objective-fit, and champion marker.
- Agent Decisions: latest decision, highlights, reasoning cards, backend gate/rejection cards, decision history, strategy scorecards, manual review and follow-up scheduling actions.
- Automation Timeline: worker requirements and execution events.
- Agent Memory: recent distilled memory records.
- Experiment Plan: latest plan, target metric, workers, estimate, experiments, warnings, execute action.
- Manual Job Queue: JSON-first debugging queue form.
- Champion Export / Demo: export status/request controls, use-case fit, preprocessing contract, demo image next/random controls, selected-image prediction action, and durable prediction-history display.
- Workers, Datasets, Recent Jobs, Run Metrics chart with metric tabs.

The app refreshes project detail by polling about every 2.5s and supervising worker requirements about every 3s.

## Key Data Displayed

Project detail is assembled from parallel requests:

- `Dataset[]`
- `Job[]`
- `ExperimentPlan[]`
- `TrainingRunSummary[]`
- `TrainingRunEvaluation[]`
- `ProjectChampion | null`
- `AgentDecision[]`
- `Worker[]`
- `WorkerRequirement[]`
- `ExecutionEvent[]`
- `AgentMemoryRecord[]`
- `StrategyScorecard[]`
- `ChampionExport[]`
- `ChampionDemoImage[]`
- `ChampionDemoPrediction[]`

Job metrics are fetched separately for the selected job.

Important payload conventions:

- Dataset intelligence is defensive over `datasets.profile`; profile JSON is still the current source of truth even though richer Go structs exist.
- Agent reasoning is mostly read from `AgentDecision.payload`, including `summary`, `evidence_used`, `deterministic_diagnosis_used`, `hypothesis`, `proposed_experiments`, `rejected_options`, `candidate_rankings`, `expected_tradeoffs`, `risks`, and validation signals.
- Champion/deployment fields may appear in `ProjectChampion.metrics`, `ProjectChampion.evaluation`, `ProjectChampion.deployment_profile`, `TrainingRunEvaluation.model_profile`, and `TrainingRunEvaluation.holistic_scores`.

## Key Files To Inspect

- `apps/mission-control/src/App.tsx`: main UI, polling, project detail aggregation, panel rendering, derived timeline/dataset/reasoning/champion helpers.
- `apps/mission-control/src/types.ts`: frontend API types. Update when backend contracts become stable.
- `apps/mission-control/src/styles.css`: layout and responsive behavior.
- `apps/mission-control/electron/main.cjs`: Electron shell, backend request bridge, dataset folder upload, worker process management.
- `apps/mission-control/electron/preload.cjs`: `window.missionControl` bridge types exposed to React.
- `services/orchestrator/internal/api/router.go`: endpoint list.
- `services/orchestrator/internal/api/handlers.go`: API response shapes, validation, review/follow-up/champion behavior.
- `services/orchestrator/internal/plans/model.go`: experiment plan and preprocessing fields.
- `services/orchestrator/internal/datasets/model.go`: dataset/profile/artifact shape.
- `services/orchestrator/internal/runs/model.go`: run summaries, evaluations, champions.
- `services/orchestrator/internal/decisions/model.go`: agent decision shape.
- `services/orchestrator/internal/memory/model.go`: agent memory and invocation shape.
- `services/orchestrator/internal/execution/model.go`: worker requirements and execution events.

## Useful Endpoints

- `GET /healthz`
- `GET/PATCH /settings/automation`
- `GET/POST /projects`
- `GET /projects/:id`
- `GET/POST /projects/:id/datasets`
- `GET/POST /projects/:id/jobs`
- `GET /projects/:id/plans`, `POST /projects/:id/plans`
- `POST /plans/:id/execute`
- `GET /projects/:id/training-run-summaries`
- `GET /projects/:id/training-run-evaluations`
- `GET /projects/:id/champion`
- `POST /projects/:id/review-experiments`
- `POST /projects/:id/schedule-follow-up-experiments`
- `GET /projects/:id/agent-decisions`
- `GET /projects/:id/agent-memory?limit=...`
- `GET /projects/:id/agent-invocations?limit=...`
- `GET /projects/:id/strategy-scorecards?limit=...`
- `GET /projects/:id/worker-requirements`
- `PATCH /worker-requirements/:id`
- `GET /projects/:id/execution-events?limit=...`
- `GET /projects/:id/workers`
- `GET /jobs/:id/metrics`
- `GET/POST /projects/:id/champion/exports`
- `GET /projects/:id/champion/demo-images?max_total_images=...&max_per_class=...`
- `GET/POST /projects/:id/champion/demo-predictions`

## Recent Work Already Done

From `docs/frontend_mission_control_report.md` and `docs/model_express_agentic_upgrade_roadmap.md`:

- Dataset upload no longer uses synchronous `adm-zip`, a temp archive file, or whole-archive `readFileSync` in Electron main. `apps/mission-control/electron/main.cjs` now streams a generated zip archive from the selected folder directly into S3 while computing SHA-256 on the stream, and `apps/mission-control/electron/zip-stream.cjs` owns the zip writer.
- New-project dataset profiling now respects the current automation default provider. When `default_training_provider` is `modal`, Mission Control starts the profile worker with `GPU_TYPE=modal`, so the worker submits profiling to Modal instead of downloading/extracting the dataset into the local project cache.
- Added experiment lifecycle timeline.
- Added dataset intelligence panel using current profile payloads.
- Expanded agent decision visibility with reasoning cards.
- Added backend gate/rejection visibility for rejected planner options and candidate-ranking failures.
- Added champion comparison table across completed training summaries/evaluations, including backend train/validation diagnostics, seed variance for repeated seeded runs, and a holistic rank score.
- Added selected-run evaluation details for backend diagnostics, per-class precision/recall/F1, and confusion matrix preview.
- Added sticky section navigation/tabs for Overview, Data, Agents, Runs, Operations, and Export/Demo.
- Added typed display for preprocessing fields on planned experiments and candidate score-component visibility from planner rankings.
- Added Champion Export / Demo panel wired defensively to champion export records, backend demo images, and backend demo prediction requests.
- Added selected demo image next/random controls plus prediction-history rendering for durable `champion_demo_predictions`.
- Runtime-unavailable prediction rows are shown as audit/status records, not as successful live inference.
- Improved responsive behavior for panels, tables, timeline, settings, and reasoning cards.
- Backend/worker work added richer dataset profile structs and worker profiler fields.
- Planner work added deterministic diagnosis, planning modes, candidate hypotheses/ranking, rejected options, strategy scorecards, and richer decision payloads.
- Orchestrator now exposes durable execution events through optional SSE refresh hints; polling remains the fallback. Leases/requeue landed, while durable DB idempotency keys remain future work.

## Backend Data Still Needed

These would make Mission Control less defensive and more useful:

- Normalized, stable dataset profile API fields or persisted `dataset_profiles` rows. Current UI should still read `datasets.profile`.
- A compact latest agent-invocation or validation-failure feed so invalid LLM outputs appear even when no durable `agent_decisions` row is created.
- Normalized rejection event list instead of inferring from `rejected_options`, `candidate_rankings`, and `validation_error`.
- Stable execution event types for dataset upload/profile completion, monitor evaluation, planner accepted/rejected, job lifecycle transitions, and champion persistence.
- Normalized champion export records, export artifact metadata, and export status are now exposed via `/projects/:id/champion/exports` and displayed in the Export/Demo panel.
- Backend demo image listing is displayed when `/projects/:id/champion/demo-images` returns images. Demo prediction requests and history call `/projects/:id/champion/demo-predictions`; worker-backed jobs can return `SUCCEEDED`, `FAILED`, or `RUNTIME_UNAVAILABLE`.
- Consistent deployment fields on evaluations/champions: `estimated_latency_ms`, `model_size_mb`, `parameter_count`, objective-fit score.
- More granular stable job lifecycle events to make SSE refreshes more targeted.

## Safe Implementation Rules

- Preserve the backend safety boundary: LLMs propose; backend validates, stores, schedules, and executes.
- Treat backend responses as nullable/partial; many fields are generic JSON records.
- Do not assume every project has a dataset, plan, decision, evaluation, champion, or worker.
- Avoid frontend-only decisions that enqueue jobs, select champions, mutate datasets, or bypass API validation.
- Keep manual JSON job queue as a debugging path unless replacing it with a safer form.
- Prefer compact, scannable operational UI over marketing-style layout.
- Do not hide rejection/failure states; they are core to agent trust.
- If adding new typed fields, keep `types.ts` aligned with Go structs and update defensive rendering in `App.tsx`.
- Respect the long-scroll issue: future large features should move toward tabs/section navigation instead of adding more full-width panels.

## No-Conflict Boundaries

- Frontend PRs should not modify orchestrator validation, worker execution, migrations, or LLM prompts unless explicitly assigned.
- Backend fields that are not yet normalized should be displayed defensively, not required.
- Do not add new infrastructure assumptions such as Kafka/Redis/NATS for UI freshness; current roadmap prefers Postgres events plus SSE first.
- Champion export/demo UI should wait for real backend export/inference contracts or clearly show empty/pending states.
- Visual exemplar work is backend/LLM-owned; frontend can display metadata later but should not curate prompt images itself.

## Future Task Checklist

- Continue refining section navigation or tabs if workspace density grows.
- Keep typed display for `resolution_strategy`, `preprocessing`, `augmentation_policy`, and `sampling_strategy` aligned with backend plan structs.
- Continue refining candidate `score_components` display.
- Add planner validation/rejection feed when backend exposes one.
- Link scorecards, decisions, invocations, follow-up plans, and outcomes.
- Improve manual job creation with a safer form-based queue builder.
- Champion Export panel is present and calls the backend export API. Demo prediction history renders durable backend audit rows, including successful worker prediction results when a worker-visible manifest/runtime is available.
- Continue using SSE as a refresh hint while preserving polling fallback.
- Add Playwright smoke coverage once tabs/navigation or export/demo flows land.

## Supporting Docs To Read

- `apps/mission-control/README.md`
- `docs/frontend_mission_control_report.md`
- `docs/model_express_agentic_upgrade_roadmap.md`
- `docs/data_preprocessing_agent_report.md`
- `docs/llm_decision_quality_report.md`
- `docs/llm_agent_decision_context.md`
- `docs/orchestrator_system_design_report.md`
- `docs/agentic.md`
