# Model Express PR Completion Report

Generated after the agentic-upgrade implementation pass. The safety boundary remains:

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
- Added worker export metadata and demo prediction payload helpers.
- Added helper-only split-file, Pascal VOC XML, and annotation JSON parsers.
- Made Modal app importable in local test environments without `modal` installed.

Files changed:

- `services/worker/worker/datasets/profiler.py`
- `services/worker/worker/datasets/annotations.py`
- `services/worker/worker/training/modal_app.py`
- `services/worker/worker/exporting/metadata.py`
- `services/worker/tests/test_profiler.py`
- `services/worker/tests/test_dataset_annotations.py`
- `services/worker/tests/test_training_modal_helpers.py`
- `services/worker/tests/test_export_metadata.py`

Tests/checks:

- `python -m py_compile worker/datasets/profiler.py worker/datasets/exemplars.py worker/datasets/annotations.py worker/training/modal_app.py worker/training/local.py worker/exporting/metadata.py worker/exporting/artifacts.py worker/exporting/inference.py` passed.
- `python -m unittest discover -s tests -v` passed; 16 tests run, 5 skipped because local env lacks `torch`/`torchvision`.

Deferred:

- Explicit split-file training.
- Bbox crop/full-image ablations and training use of parsed annotations.
- Advanced augmentation policies.
- Backend-scheduled champion export/inference jobs.

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

Tests/checks:

- `npm run build` passed. Vite emitted the existing Node CommonJS/ESM experimental warning.

Deferred:

- Full live predictions until worker/backend inference runtime exists.
- Download/use instructions beyond current metadata display.

## PR 6: Orchestrator/System Design Cleanup And Observability

Owning agents: Orchestrator / Backend, System Architecture Review.

What changed:

- Added low-risk positive epoch validation at API and store boundaries.
- Preserved current Postgres-first reliability stance.
- Added export-request and demo-prediction execution events.
- Added idempotent champion export request behavior for the same project/champion/format.

Files changed:

- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/store/memory.go`
- `services/orchestrator/internal/store/postgres.go`
- `services/orchestrator/internal/execution/model.go`
- `services/orchestrator/internal/api/handlers_test.go`
- `docs/agents_context/system_architecture_review/context.md`

Tests/checks:

- `go test ./...` passed.

Deferred:

- Job leases/requeue.
- Durable idempotency keys for plan execution and planner outcomes.
- SSE/event stream.
- Async background task runner for LLM automation.

## PR 7: Champion Export + Live Demo

Owning agents: Orchestrator / Backend, Python Worker / Training, Frontend / Mission Control.

What changed:

- Added backend champion export record contract and APIs.
- Added `champion_exports` table and store methods.
- Added `runs.ChampionExport`.
- Made export requests idempotent per project/champion/format without adding an unsafe unique migration over existing rows.
- Added backend demo image listing from capped `datasets.profile` metadata.
- Added `champion_demo_predictions` table, store methods, history API, and execution event.
- Changed demo prediction requests to persist audited `RUNTIME_UNAVAILABLE` rows and return `202` while inference runtime is absent.
- Added worker export manifest/checkpoint helpers with guarded TorchScript/ONNX paths.
- Added worker TorchScript demo inference helper that returns ranked predictions when a valid worker-owned artifact exists, and deterministic pending/error payloads otherwise.
- Added Mission Control export/demo panel wired to export, demo image, and prediction-history APIs.

New APIs:

- `GET /projects/:id/champion/exports`
- `POST /projects/:id/champion/exports`
- `GET /projects/:id/champion/demo-images`
- `GET /projects/:id/champion/demo-predictions`
- `POST /projects/:id/champion/demo-predictions`

New schema/config:

- `champion_exports`
- `runs.ChampionExport`
- `runs.ChampionExportCreate`
- `champion_demo_predictions`
- `runs.ChampionDemoPrediction`
- `runs.ChampionDemoPredictionCreate`
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
- `apps/mission-control/src/App.tsx`
- `apps/mission-control/src/types.ts`
- `apps/mission-control/src/styles.css`

Tests/checks:

- `go test ./...` passed.
- `python -m unittest discover -s tests -v` passed with torch/torchvision-dependent skips.
- `npm run build` passed.

Deferred:

- Backend-scheduled worker export and inference jobs.
- Production model-serving/runtime hosting.
- Strong DB uniqueness for export idempotency after duplicate-row migration review.

Manual demo steps:

- Select a project with a persisted champion.
- Open Export/Demo.
- Click Request ONNX.
- Confirm the export record appears as `READY` if an artifact URI exists or `PENDING_ARTIFACT` with validation errors if not.
- If demo images are present in dataset profile, click Predict. Current backend records `RUNTIME_UNAVAILABLE` until worker-backed inference is wired.

## PR 8: Visual Class Exemplars For LLM Planning

Owning agents: Data / Dataset Intelligence, Orchestrator / Backend, LLM Decision Intelligence.

What changed:

- Added `datasets.VisualExemplar`.
- Added `GET /datasets/:id/visual-exemplars`.
- Added capped visual exemplar extraction from `datasets.profile.visual_exemplars`.
- Added capped demo image extraction from `datasets.profile.demo_images` or `visual_exemplars`.
- Wired evidence-only visual exemplar summaries into Experiment Planner input.
- Added planner exemplar caps/audit details and tests.
- Added worker-side deterministic class-balanced exemplar generation with downscale/compression and byte/image caps.

New APIs/schemas:

- `GET /datasets/:id/visual-exemplars`
- `datasets.VisualExemplar`
- `agents.PlannerVisualExemplarContext`
- `agents.PlannerClassExemplar`

Files changed:

- `services/orchestrator/internal/datasets/model.go`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/api/router.go`
- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/experiment_planner_llm_test.go`
- `services/worker/worker/datasets/exemplars.py`
- `services/worker/tests/test_dataset_exemplars.py`

Tests/checks:

- `go test ./...` passed.

Deferred:

- Uploading/storing generated exemplar packs in backend profile/artifact paths.
- Durable exemplar tables and invocation audit fields beyond prompt context.
- Multimodal image attachment beyond compact metadata context.

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

## Final Verification

- `go test ./...` from `services/orchestrator`: passed.
- `python -m py_compile worker/datasets/profiler.py worker/datasets/exemplars.py worker/datasets/annotations.py worker/training/modal_app.py worker/training/local.py worker/exporting/metadata.py worker/exporting/artifacts.py worker/exporting/inference.py` from `services/worker`: passed.
- `python -m unittest discover -s tests -v` from `services/worker`: passed; 16 tests run, 5 skipped due missing local `torch`/`torchvision`.
- `npm run build` from `apps/mission-control`: passed; existing Node experimental CommonJS/ESM warning from Vite config.

## Known Risks

- Worker crash recovery still needs job leases and requeue.
- Export/demo now has manifest/inference helpers and durable audit records, but backend-scheduled worker inference is not wired.
- Visual exemplars can be generated by worker helpers, but backend upload/persistence still depends on profile JSON metadata.
- `dataset_profiles` remains dormant; do not read from it as canonical until fully wired.
