# Model Express PR Completion Report

Generated after the agentic-upgrade implementation pass and updated after the deferred-work completion pass. The safety boundary remains:

```text
LLMs propose structured JSON -> backend validates -> backend stores/schedules -> workers execute
```

## PR 1: Dataset Profiling + Preprocessing Schema + Backend Validation

Owning agents: Data / Dataset Intelligence, Orchestrator / Backend, Integration Coordinator.

What changed:

- Kept `datasets.profile` JSON as the active dataset-profile source of truth.
- Documented `dataset_profiles` rows as deferred to avoid split-brain profile state.
- Kept generic `dataset_artifacts` table deferred; profile JSON `artifacts`, `visual_exemplars`, and `demo_images` are the current compact artifact/exemplar source.
- Preserved alignment for `resolution_strategy`, `preprocessing`, `augmentation_policy`, `sampling_strategy`, class balancing, and validation.

Files changed:

- `docs/agents_context/data_dataset_intelligence/context.md`
- `docs/data_preprocessing_agent_report.md`
- backend/profile files touched by adjacent PRs: `services/orchestrator/internal/datasets/model.go`, `services/orchestrator/internal/api/handlers.go`

Tests/checks:

- Covered by `go test ./...`.

Deferred:

- Normalized `dataset_profiles` activation and backfill.
- Generic `dataset_artifacts` persistence.

## PR 2: Python Worker Preprocessing Support + Preprocessing Ablations

Owning agents: Python Worker / Training, Data / Dataset Intelligence.

What changed:

- Added worker tests for profile artifact detection, dataset normalization metadata, transform construction, sampler selection, loss selection, and export/demo payload helpers.
- Added bounded dataset-computed normalization metadata and Modal transform consumption for `normalization: "dataset"` / `use_dataset_normalization`.
- Exposed pure sampler-selection helper.
- Added worker export metadata, demo prediction payload helpers, and champion job dispatch safety tests.
- Added helper-only split-file, Pascal VOC XML, and annotation JSON parsers.
- Made Modal app importable in local test environments without `modal` installed.
- Unknown worker job templates now fail closed instead of fake-running.

Files changed:

- `services/worker/worker/datasets/profiler.py`
- `services/worker/worker/datasets/annotations.py`
- `services/worker/worker/training/modal_app.py`
- `services/worker/worker/exporting/metadata.py`
- `services/worker/worker/exporting/artifacts.py`
- `services/worker/worker/exporting/inference.py`
- `services/worker/worker/jobs.py`
- `services/worker/worker/champion_jobs.py`
- `services/worker/tests/test_profiler.py`
- `services/worker/tests/test_dataset_annotations.py`
- `services/worker/tests/test_training_modal_helpers.py`
- `services/worker/tests/test_export_metadata.py`
- `services/worker/tests/test_champion_jobs.py`

Tests/checks:

- `python -m py_compile worker/jobs.py worker/orchestrator_client.py worker/champion_jobs.py worker/exporting/artifacts.py` passed.
- `python -m unittest discover -s tests -v` passed; 21 tests run, 5 skipped because local env lacks `torch`/`torchvision`.

Deferred:

- Explicit split-file training.
- Bbox crop/full-image ablations and training use of parsed annotations.
- Advanced augmentation object policies.

## PR 3: LLM Planner Schema/Prompt Upgrades + Planning Modes

Owning agent: LLM Decision Intelligence.

What changed:

- Updated Experiment Planner prompt examples and rules for:
  - `resolution_strategy`
  - `preprocessing`
  - `augmentation_policy`
  - `sampling_strategy`
  - class balancing/loss options
- Reinforced JSON-only output and backend-validation execution gate.
- Added optional evidence-only visual exemplar context to planner input and prompt context.
- Added visual exemplar caps/audit details to planner prompt context.
- Tightened planner instructions so LLM output cannot request exports, inference runs, worker/job creation, arbitrary files, dataset mutation, or backend-validation bypass.
- Added planner tests for preprocessing-aware outputs, visual exemplar prompt handling, caps/audit preservation, and JSON-only/backend-validation wording.

Files changed:

- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/experiment_planner_llm_test.go`
- `docs/agents_context/llm_decision_intelligence/context.md`

Tests/checks:

- `go test ./internal/agents` passed through the full backend run.
- `go test ./...` passed.

Deferred:

- Durable invocation audit fields beyond prompt context for whether visual exemplars were used.
- Semantic retrieval across similar datasets/objectives/strategies.

## PR 4: Strategy Scorecards + Candidate Ranking + Rejected Options Memory

Owning agent: LLM Decision Intelligence, with Integration Coordinator checks.

What changed:

- Verified candidate ranking, rejected options, scorecards, and memory stay compatible with preprocessing/export/exemplar additions.
- Updated candidate ranking so `resolution_strategy`, `preprocessing`, `augmentation_policy`, `sampling_strategy`, crop/bbox, and normalization changes count as meaningful mechanisms.
- Kept experiment ranking holistic rather than single-metric-only by combining quality, latency, runtime, cost, backend diagnostics, and seed stability where available.
- Tightened backend follow-up validation so same-mechanism plans with only epochs/batch size/learning-rate changes are rejected for LLM proposals and filtered from deterministic follow-up payloads before plan creation.
- Added one bounded LLM correction retry when a planner response passes JSON/schema validation but fails backend proposal validation; retry context includes `planner_validation_feedback`, and corrected output still must pass deterministic backend validation.
- Frontend now displays candidate `score_components`, selection/rejection state, and ranking reasons.

Files changed:

- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/candidate_ranking.go`
- `services/orchestrator/internal/agents/experiment_planner_llm_test.go`
- `apps/mission-control/src/App.tsx`

Tests/checks:

- `go test ./...` passed.
- `npm run build` passed.

Deferred:

- Normalized validation/rejection feed independent of stored decisions.

## PR 5: Frontend Mission Control UI Improvements

Owning agent: Frontend / Mission Control.

What changed:

- Added sticky section navigation for Overview, Data, Agents, Runs, Operations, Export/Demo.
- Added typed plan display for preprocessing fields.
- Added candidate score-component UI.
- Added Champion Export / Demo panel with backend export/demo API integration.
- Added selected demo image next/random controls and prediction-history rendering.
- Added defensive display for `RUNTIME_UNAVAILABLE`, prediction errors, true label, top-k, latency, and correctness.
- Expanded Champion Comparison with rank score, train/validation gap status, and seed variance for repeated seeded runs.
- Added selected-run evaluation details for backend diagnostics, per-class precision/recall/F1, and confusion matrix preview.
- Added optional SSE refresh handling through `GET /projects/:id/events/stream` while keeping polling as the durable fallback.

Files changed:

- `apps/mission-control/src/App.tsx`
- `apps/mission-control/src/types.ts`
- `apps/mission-control/src/styles.css`
- `docs/agents_context/frontend_mission_control/context.md`

New UI surfaces:

- Section tabs.
- Candidate score cards.
- Preprocessing chips on experiments.
- Export status/request panel.
- Demo image selector and prediction-history panel with runtime-unavailable state.
- Champion comparison rank/gap/seed-variance columns.
- Selected-run diagnostics/per-class/confusion preview under Run Metrics.
- SSE-triggered refresh hints for project events.

Tests/checks:

- `npm run build` passed. Vite emitted the existing Node CommonJS/ESM experimental warning.

Deferred:

- Successful live predictions depend on a worker-visible export manifest/artifact and local runtime dependencies.
- Download/use instructions beyond current metadata display.

## PR 6: Orchestrator/System Design Cleanup And Observability

Owning agents: Orchestrator / Backend, System Architecture Review.

What changed:

- Added low-risk positive epoch validation at API and store boundaries.
- Preserved current Postgres-first reliability stance.
- Added export-request and demo-prediction execution events.
- Added idempotent champion export request behavior for the same project/champion/format.
- Added additive job lease columns (`attempt`, `max_attempts`, `lease_owner_worker_id`, `lease_expires_at`, `lease_last_heartbeat_at`).
- Added lease renewal during polling/metric reporting and stale job requeue/fail recovery.
- Added `GET /projects/:id/events/stream` SSE over durable `execution_events`.
- Added focused lease recovery tests.
- Added backend `training_diagnostics` enrichment for final training evaluations, stored under `training_run_evaluations.holistic_scores`.
- Raised the deterministic review spend budget to `$10.00` and aligned backend/worker cost-score normalization with that budget.

Files changed:

- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/store/memory.go`
- `services/orchestrator/internal/store/postgres.go`
- `services/orchestrator/internal/execution/model.go`
- `services/orchestrator/internal/api/handlers_test.go`
- `services/orchestrator/internal/jobs/model.go`
- `services/orchestrator/internal/store/migrations/001_init.sql`
- `docs/agents_context/system_architecture_review/context.md`

Tests/checks:

- `go test ./...` passed.

Deferred:

- Durable idempotency keys for plan execution and planner outcomes.
- Async background task runner for LLM automation.
- Standalone lease-recovery ticker; current recovery runs on poll/manual store recovery.

## PR 7: Champion Export + Live Demo

Owning agents: Orchestrator / Backend, Python Worker / Training, Frontend / Mission Control.

What changed:

- Added backend champion export record contract and APIs.
- Added `champion_exports` table and store methods.
- Added `runs.ChampionExport`.
- Made export requests idempotent per project/champion/format without adding an unsafe unique migration over existing rows.
- Export requests now enqueue validated `export_champion` jobs when worker artifact work is needed.
- Added `POST /jobs/:id/champion-export-result`; the backend validates the job template/config before updating export state.
- Added backend demo image listing from capped `datasets.profile` metadata.
- Added `champion_demo_predictions` table, store methods, history API, and execution event.
- Demo prediction requests now persist audit rows and enqueue `champion_demo_prediction` jobs when a `READY` export exists.
- Added `POST /jobs/:id/champion-demo-prediction-result`; worker results can mark predictions `SUCCEEDED`, `FAILED`, or `RUNTIME_UNAVAILABLE`.
- Added terminal `STOP_PROJECT` champion fallback: if a stop decision does not name a champion but successful runs exist, the backend persists the best successful run so far as the project champion.
- Added worker export manifest/checkpoint helpers with guarded TorchScript/ONNX paths.
- Added worker TorchScript demo inference helper that returns ranked predictions when a valid worker-owned artifact exists, and deterministic pending/error payloads otherwise.
- Added worker job handlers for export and demo prediction jobs.
- Added Mission Control export/demo panel wired to export, demo image, and prediction-history APIs.

New APIs:

- `GET /projects/:id/champion/exports`
- `POST /projects/:id/champion/exports`
- `GET /projects/:id/champion/demo-images`
- `GET /projects/:id/champion/demo-predictions`
- `POST /projects/:id/champion/demo-predictions`
- `POST /jobs/:id/champion-export-result`
- `POST /jobs/:id/champion-demo-prediction-result`

New schema/config:

- `champion_exports`
- `runs.ChampionExport`
- `runs.ChampionExportCreate`
- `runs.ChampionExportUpdate`
- `champion_demo_predictions`
- `runs.ChampionDemoPrediction`
- `runs.ChampionDemoPredictionCreate`
- `runs.ChampionDemoPredictionUpdate`
- `jobs.TemplateExportChampion`
- `jobs.TemplateChampionDemoPrediction`
- `CHAMPION_EXPORT_REQUESTED`
- `CHAMPION_DEMO_PREDICTION`

Files changed:

- `services/orchestrator/internal/api/router.go`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/runs/model.go`
- `services/orchestrator/internal/store/store.go`
- `services/orchestrator/internal/store/memory.go`
- `services/orchestrator/internal/store/postgres.go`
- `services/orchestrator/internal/store/migrations/001_init.sql`
- `services/worker/worker/exporting/metadata.py`
- `services/worker/worker/exporting/artifacts.py`
- `services/worker/worker/exporting/inference.py`
- `services/worker/worker/champion_jobs.py`
- `services/worker/worker/jobs.py`
- `services/worker/worker/orchestrator_client.py`
- `services/worker/tests/test_champion_jobs.py`
- `apps/mission-control/src/App.tsx`
- `apps/mission-control/src/types.ts`
- `apps/mission-control/src/styles.css`

Tests/checks:

- `go test ./...` passed.
- `python -m unittest discover -s tests -v` passed with torch/torchvision-dependent skips.
- `npm run build` passed.

Deferred:

- Production model-serving/runtime hosting beyond worker-owned demo inference.
- Real model reconstruction/export when no completed training artifact is available to the worker.
- Production storage upload for worker-local export artifacts.
- Strong DB uniqueness for export idempotency after duplicate-row migration review.

Manual demo steps:

- Select a project with a persisted champion.
- Open Export/Demo.
- Click Request ONNX.
- Confirm the export record appears as `READY` if an artifact URI exists or `PENDING_ARTIFACT` with validation errors if not.
- If demo images are present in dataset profile, click Predict. With a `READY` worker-visible export manifest, the worker reports top-k prediction results; otherwise the audit row records `RUNTIME_UNAVAILABLE`.

## PR 8: Visual Class Exemplars For LLM Planning

Owning agents: Data / Dataset Intelligence, Orchestrator / Backend, LLM Decision Intelligence.

What changed:

- Added `datasets.VisualExemplar`.
- Added `GET /datasets/:id/visual-exemplars`.
- Added `POST /datasets/:id/visual-exemplars` with backend count/per-class/byte caps and merge into canonical `datasets.profile`.
- Added capped visual exemplar extraction from `datasets.profile.visual_exemplars`.
- Added capped demo image extraction from `datasets.profile.demo_images` or `visual_exemplars`.
- Wired evidence-only visual exemplar summaries into Experiment Planner input.
- Added planner exemplar caps/audit details and tests.
- Added worker-side deterministic class-balanced exemplar generation with downscale/compression and byte/image caps.
- Added `generate_visual_exemplars` worker job dispatch and backend-compatible result payloads.

New APIs/schemas:

- `GET /datasets/:id/visual-exemplars`
- `POST /datasets/:id/visual-exemplars`
- `datasets.VisualExemplar`
- `agents.PlannerVisualExemplarContext`
- `agents.PlannerClassExemplar`
- `jobs.TemplateGenerateVisualExemplars`

Files changed:

- `services/orchestrator/internal/datasets/model.go`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/api/router.go`
- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/experiment_planner_llm_test.go`
- `services/worker/worker/datasets/exemplars.py`
- `services/worker/worker/champion_jobs.py`
- `services/worker/worker/orchestrator_client.py`
- `services/worker/tests/test_dataset_exemplars.py`
- `services/worker/tests/test_champion_jobs.py`

Tests/checks:

- `go test ./...` passed.

Deferred:

- Durable exemplar tables and invocation audit fields beyond prompt context.
- Production object-storage upload for exemplar image files beyond worker-local `file://` URIs.
- Multimodal image attachment beyond compact metadata/profile context.

## PR 9: `me_ground_truth.md` System Reference

Owning agent: Integration Coordinator.

What changed:

- Created durable system reference explaining architecture, data model, flows, modules, LLM context, memory, backend validation, export/demo, reliability, and deferrals.
- Updated agent context packs where durable responsibilities/contracts changed.
- Created this completion report.

Files changed:

- `docs/me_ground_truth.md`
- `docs/model_express_pr_completion_report.md`
- `docs/model_express_end_to_end_checklist.md`
- `docs/agents_context/data_dataset_intelligence/context.md`
- `docs/agents_context/orchestrator_backend/context.md`
- `docs/agents_context/python_worker_training/context.md`
- `docs/agents_context/llm_decision_intelligence/context.md`
- `docs/agents_context/frontend_mission_control/context.md`
- `docs/agents_context/system_architecture_review/context.md`
- `docs/agents_context/integration_coordinator/context.md`
- `docs/data_preprocessing_agent_report.md`

Tests/checks:

- Documentation-only.

## PR 10: Compact Planner Context Snapshot

Owning agent: LLM Decision Intelligence.

What changed:

- Implemented `planner_context_snapshot_v1` for Experiment Planner LLM prompts.
- Replaced raw prompt context (`dataset.profile`, `prior_plans`, `prior_jobs`, `prior_evaluations`, raw memory, and epoch metrics) with distilled evidence: dataset card, objective context, failure diagnosis, champion/baseline comparison, completed experiment ledger, search coverage, strategy lessons, blocked repeats, visual evidence, model catalog, retry validation feedback, and follow-up pressure.
- Kept full Postgres history available to backend validation/ranking while only giving the LLM a capped decision brief.
- Updated prompt rules to point the LLM at `planner_context_snapshot` and remind it that backend validation checks the full project history even when only capped signature samples are shown.
- Added regression coverage that long follow-up histories do not leak raw context into planner prompts while preserving champion, ledger, coverage, strategy, blocked-repeat, and validation-feedback evidence.
- Preserved the existing safety boundary: snapshots are reasoning context only, and backend validation/model catalog/duplicate detection/scheduling gates remain unchanged.

Files changed:

- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/experiment_planner_llm_test.go`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/api/handlers_test.go`
- `docs/agents_context/llm_decision_intelligence/context.md`
- `docs/model_express_pr_completion_report.md`

Tests/checks:

- `go test ./internal/agents` passed.
- `go test ./internal/api -run "TestExperimentPlannerPromptContextCompactsLongFollowUpHistory|TestExperimentPlannerRetryUsesValidationFeedbackBeforeScheduling|TestAllFilteredPlannerExperimentsDoNotCreateFollowUpPlan" -count=1` passed.
- `go test ./...` from `services/orchestrator` passed.

Deferred:

- Semantic retrieval across similar datasets/objectives/strategies.
- Durable invocation audit fields beyond compact prompt context.

## PR 11: Modal-First Dataset Upload And Profile Flow

Owning agents: Frontend Mission Control, Python Worker / Training, System Architecture Review.

What changed:

- Replaced synchronous Electron `adm-zip` upload with a streaming zip writer that sends the selected dataset folder directly to S3 while computing SHA-256 on the stream.
- Removed temp archive creation and whole-archive `readFileSync` from the upload path, preventing Electron main-process freezes and local temp zip buildup for large datasets.
- Added Modal dataset profiling for workers launched with `GPU_TYPE=modal` or `MODEL_EXPRESS_DATASET_PROFILE_PROVIDER=modal`.
- Local profile jobs now clean `.cache/datasets/:dataset_id` by default after profiling. Set `MODEL_EXPRESS_PERSIST_DATASET_CACHE=1` only when intentionally keeping local cache data.
- New-project profiling now follows Mission Control's default training provider, so Modal-default projects launch a Modal-mode profile worker.
- Updated future-agent context docs and ground-truth flow notes.

Files changed:

- `apps/mission-control/electron/main.cjs`
- `apps/mission-control/electron/zip-stream.cjs`
- `apps/mission-control/src/App.tsx`
- `services/worker/worker/datasets/cache.py`
- `services/worker/worker/jobs.py`
- `services/worker/worker/training/modal_app.py`
- `services/worker/worker/training/modal_provider.py`
- `services/worker/tests/test_dataset_cache.py`
- `docs/me_ground_truth.md`
- `docs/agents_context/frontend_mission_control/context.md`
- `docs/agents_context/python_worker_training/context.md`
- `docs/agents_context/system_architecture_review/context.md`
- `docs/model_express_pr_completion_report.md`

Tests/checks:

- Streaming zip smoke test created a dataset archive and validated it with Python `zipfile`.
- `python -m pytest tests/test_dataset_cache.py tests/test_training_modal_helpers.py` from `services/worker`: passed, with existing Modal/torch-dependent skips.
- `npm run build` from `apps/mission-control`: passed; existing Node experimental CommonJS/ESM warning from Vite config.

Deferred:

- Progress reporting/cancellation for long uploads.
- Multipart/resumable object-storage upload for unstable networks.
- Moving all exemplar/demo dataset reads to Modal-backed temporary storage.

## PR 12: Near-Ceiling Champion Stop Guard

Owning agents: Orchestrator / Backend, LLM Decision Intelligence.

What changed:

- Hardened backend planner stop criteria so an LLM `ADD_EXPERIMENTS` recommendation is converted to `SELECT_CHAMPION` when the current champion is within the minimum meaningful improvement threshold of a bounded metric ceiling.
- Added stop-signal text for near-ceiling champions so the compact planner context tells the LLM there is no meaningful accuracy/macro-F1 headroom left.
- Prevented follow-up plan/job creation for near-ceiling champions even when the LLM proposes novel challenger experiments or an older stored `ADD_EXPERIMENTS` decision is reused.
- Kept the existing repeated no-improvement guard intact.

Files changed:

- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/api/handlers_test.go`
- `docs/agents_context/llm_decision_intelligence/context.md`
- `docs/model_express_pr_completion_report.md`

Tests/checks:

- `go test ./internal/api -run "TestExperimentPlannerStopCriteriaSelectsNearCeilingChampion|TestNearCeilingChampionBlocksPlannerFollowUpScheduling|TestNearCeilingStaleAddDecisionCannotScheduleFollowUp|TestExperimentPlannerStopCriteriaSelectsChampionAfterStalledFollowUps" -count=1` passed.
- `go test ./internal/agents -count=1` passed.
- `go test ./...` from `services/orchestrator` passed.

## Final Verification

- `go test ./...` from `services/orchestrator`: passed.
- `python -m py_compile worker/jobs.py worker/orchestrator_client.py worker/champion_jobs.py worker/exporting/artifacts.py` from `services/worker`: passed.
- `python -m unittest discover -s tests -v` from `services/worker`: passed; 21 tests run, 5 skipped due missing local `torch`/`torchvision`.
- `npm run build` from `apps/mission-control`: passed; existing Node experimental CommonJS/ESM warning from Vite config.

## Known Risks

- Lease recovery is implemented through poll/manual store recovery, but there is no standalone recovery ticker yet.
- Export/demo can run through worker jobs when artifacts/manifests are worker-visible; production storage and real reconstruction from arbitrary completed training runs remain future hardening.
- Visual exemplars persist into canonical profile JSON, but durable exemplar history tables and production object-storage upload remain future hardening.
- `dataset_profiles` remains dormant; do not read from it as canonical until fully wired.
