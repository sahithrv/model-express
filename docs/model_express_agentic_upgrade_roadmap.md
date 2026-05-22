# Model Express Agentic Upgrade Roadmap

## Integration Summary

This pass split the agentic experimentation upgrade into PR-sized tracks and kept the core safety boundary intact:

```text
LLMs propose structured plans -> backend validates -> backend stores/schedules -> workers execute
```

The repo now has stronger support for dataset intelligence, preprocessing-aware experiment configs, deterministic planner diagnosis, candidate ranking, strategy scorecards, and Mission Control visibility. The work should still be merged as multiple PRs rather than one large change.

Final implementation status is tracked in:

- `docs/model_express_pr_completion_report.md`
- `docs/model_express_end_to_end_checklist.md`
- `docs/me_ground_truth.md`

This roadmap also includes three additional product goals:

- Export the selected champion model for real use.
- Add a Mission Control champion demo that runs live predictions on held-out/test images.
- Give the LLM planner a small, backend-curated set of downscaled class example images so it can connect dataset statistics to actual visual evidence.
- Create `docs/me_ground_truth.md` as the durable system reference for how Model Express works.

## What Was Implemented

### Dataset Profiling And Preprocessing

Implemented in PR 1 scope:

- Richer dataset profile structs in `services/orchestrator/internal/datasets/model.go`.
- Worker profiler fields for class distribution, image count, imbalance ratio, image dimension stats, corrupt files, metadata/artifacts, leakage warnings, and dataset traits.
- Optional `PlannedExperiment` fields for:
  - `resolution_strategy`
  - `preprocessing`
  - `augmentation_policy`
  - `sampling_strategy`
- Backend validation for supported preprocessing, augmentation, sampling, optimizer, scheduler, and class-balancing options.
- Experiment signatures now include preprocessing fields so duplicates are detected correctly.
- Execution config passes validated preprocessing fields to workers.
- Local and Modal workers understand a safe preprocessing/class-balancing subset.
- Focused backend validation tests and Python compile checks.

### LLM Decision Quality

Implemented in PR 3/4 scope:

- Deterministic planner diagnosis fields already exist and are passed to planner context.
- Explicit planning modes are validated.
- Candidate hypotheses can be ranked deterministically when the LLM leaves `proposed_experiments` empty.
- Candidate rankings now include `score_components` so decisions are explainable.
- Rejected options and strategy scorecards are represented in planner context and decision payloads.
- Planner prompt requires evidence, diagnosis, rejected options, stop conditions, and champion comparison.

### Mission Control UX

Implemented in PR 5 scope:

- Experiment lifecycle timeline.
- Dataset intelligence panel.
- Agent reasoning and backend gate cards.
- Candidate rejection visibility.
- Champion comparison panel.
- Responsive layout cleanup and richer empty/loading states.

### Champion Export And Demo

Implemented as a safe control-plane/helper slice:

- The backend stores idempotent champion export records after validating a selected champion and completed run.
- Missing artifacts are represented as `PENDING_ARTIFACT`, not fake readiness.
- Worker export helpers can create manifest/checkpoint metadata and guarded TorchScript/ONNX artifacts when dependencies and model objects exist.
- The backend stores durable demo prediction audit rows and returns `RUNTIME_UNAVAILABLE` until worker-backed inference is scheduled.
- Mission Control includes an Export/Demo panel with export status, use-case fit, selected demo image, next/random controls, prediction history, top-k, latency, true-label, correctness, and runtime-unavailable display.
- Backend-scheduled export/inference worker jobs and production serving remain deferred.

### LLM Visual Dataset Exemplars

Implemented as a safe evidence-only slice:

- Backend APIs expose capped exemplars from `datasets.profile.visual_exemplars` / `demo_images`.
- Planner context treats visual exemplars as backend-curated evidence only and carries caps/audit details.
- Worker helper code can generate class-balanced downscaled/compressed exemplar packs with image/byte caps.
- LLM output remains JSON only and backend validation remains unchanged.
- Uploading generated exemplars into backend profile/artifact storage and true multimodal image attachment remain deferred.

### Ground Truth System Document

Implemented:

- `docs/me_ground_truth.md` is the durable system reference.
- `docs/model_express_pr_completion_report.md` records PR-by-PR completion details.
- `docs/model_express_end_to_end_checklist.md` records acceptance, verification, manual demo steps, and remaining explicit deferrals.

### Orchestrator/System Design

Implemented in PR 6 scope:

- System design report with current architecture, bottlenecks, reliability gaps, and future eventing guidance.
- Additive Postgres indexes for project/status scans, follow-up plan lookup, plan-scoped decisions, worker requirements, and agent memory/invocation filters.

## Partially Implemented

- `dataset_profiles` exists in the migration, but profile persistence still primarily uses `datasets.profile` JSON. No store/API path writes full `dataset_profiles` rows yet.
- Dataset artifacts are represented in profile JSON and Go structs, not a dedicated `dataset_artifacts` table.
- Bounding box support is schema/spec-level only. Actual bbox crop execution should be a later PR.
- `normalization: "dataset"` is validated as future-facing, but workers do not compute and apply dataset-specific normalization yet.
- The implemented `augmentation_policy` is a safe string subset: `none|light|moderate|strong|custom`. The original target object form with `basic|randaugment|trivialaugment|autoaugment|mixup|cutmix` is not implemented yet.
- The planner prompt examples still need to be updated to show the new PR 1 preprocessing fields explicitly.
- Mission Control can display defensive profile/reasoning fields, but it still depends on generic payloads instead of normalized rejection/invocation summary endpoints.

## Conflict Check

### Schema And Config

- Backend `PlannedExperiment`, backend validation, execution config, candidate signatures, and worker config now agree on the PR 1 preprocessing fields.
- Candidate ranking and API duplicate detection both include preprocessing, augmentation policy, sampling strategy, and resolution strategy in signatures.
- The frontend reads most new data defensively through generic records, so it does not require backend migrations to render.

### Known Mismatches To Resolve

- Augmentation policy shape differs from the original requested object schema. Keep PR 1's string policy as the initial safe contract, then add object-shaped advanced policies in PR 2.
- `DatasetProfile` Go structs and `dataset_profiles` SQL table are not yet wired together. Either wire persistence in PR 1 follow-up or document `datasets.profile` JSON as the current source of truth.
- LLM planner prompt now includes `preprocessing`, `sampling_strategy`, `resolution_strategy`, and the supported value list.
- Mission Control wants validation failures even when no `agent_decisions` row is persisted. That requires an agent invocation summary endpoint or rejection event stream.

## Work Still Needed After Final Pass

### Backend

- Fully promote normalized `dataset_profiles`, or keep the table permanently documented as dormant.
- Add a first-class `dataset_artifacts` table only if artifact tracking needs history, file-level metadata, annotation parsing, or provenance.
- Wire backend-scheduled worker jobs for champion export artifact production and demo inference.
- Prefer test/held-out split images when split parsing is wired into training/profile state.
- Persist/upload worker-generated visual exemplar packs so backend profile/API paths can serve generated images, not only profile metadata.
- Add advanced augmentation policy validation once workers support RandAugment, TrivialAugment, AutoAugment, MixUp, and CutMix.
- Add normalized rejection events for failed LLM validation.
- Add job lease/requeue support and durable idempotency keys.

### Worker

- Implement real bbox crop/full-image paired ablations.
- Wire helper-level annotation XML/JSON and split-file parsing into training/profile behavior; labels CSV, class hierarchy, and metadata folder parsing remain future work.
- Add real advanced augmentation policies beyond the safe current subset.
- Wire explicit split-file training.
- Wire export manifest/checkpoint helpers into real training/export job outputs and object storage.
- Wire lightweight demo inference helper into backend-scheduled worker execution.
- Upload worker-side visual exemplar outputs or embed compact metadata into `datasets.profile`.

### Frontend

- Add a compact planner validation/rejection feed.
- Link scorecards, decisions, invocations, follow-up plans, and outcomes together in the UI.
- Add download/use instructions once artifacts are available.
- Show `SUCCEEDED` live predictions once backend-scheduled inference exists.

### Migrations

- Current additive indexes are safe.
- Future migrations should add, only when needed:
  - `dataset_artifacts`
  - `dataset_visual_exemplars` or equivalent artifact records for class example images
  - persisted normalized `dataset_profiles` writes or latest-profile helper view
  - optional stricter export idempotency constraint after duplicate export row migration review
  - worker inference result/update tables or status fields if a future job path needs them
  - job lease fields and retry attempts
  - idempotency keys/constraints for plan execution and planner outcomes
  - event cursor fields for SSE

### Tests

- Keep existing backend validation tests.
- Add backend tests for future worker-backed export/inference job transitions.
- Add UI component tests or Playwright smoke once Mission Control grows section navigation.
- Add duplicate plan execution/idempotency tests.
- Add stale worker/job recovery tests after lease support lands.

## Recommended PR Sequence

### PR 1: Dataset Profiling + Preprocessing Schema + Backend Validation

Owner: Data Preprocessing / Dataset Intelligence Agent

Files/features:

- `services/orchestrator/internal/plans/model.go`
- `services/orchestrator/internal/datasets/model.go`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/api/handlers_test.go`
- `services/orchestrator/internal/agents/experiment_planner_llm.go` only for dataset insight structs if needed
- `services/orchestrator/internal/agents/candidate_ranking.go` only for signatures if needed
- `services/worker/worker/datasets/profiler.py`
- `services/worker/worker/training/local.py`
- `services/worker/worker/training/modal_app.py`
- `docs/data_preprocessing_agent_report.md`

Dependencies: none.

Defer from PR 1:

- advanced augmentation object schema
- real bbox crop execution
- dataset normalization computation
- dedicated `dataset_artifacts` table

### PR 2: Python Worker Preprocessing Support + Preprocessing Ablations

Owner: Data/Worker follow-up

Files/features:

- worker transform tests
- annotation parsing
- bbox crop vs full-image paired ablations
- dataset-computed normalization
- split-file training
- advanced augmentation policies

Dependencies: PR 1.

### PR 3: LLM Planner Schema/Prompt Upgrades + Planning Modes

Owner: LLM Decision Quality Agent

Files/features:

- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/diagnosis.go`
- planner prompt examples
- tests for required evidence, diagnosis, stop conditions, rejected options, and preprocessing-aware planner outputs
- `docs/llm_decision_quality_report.md`

Dependencies: PR 1 for preprocessing field names and supported values.

### PR 4: Strategy Scorecards + Candidate Ranking + Rejected Options Memory

Owner: LLM Decision Quality Agent

Files/features:

- `services/orchestrator/internal/agents/candidate_ranking.go`
- `services/orchestrator/internal/strategies/model.go`
- store/API scorecard methods
- planner outcome memory updates
- candidate `score_components`
- tests for ranking, outcome idempotency, and rejected memory retrieval

Dependencies: PR 3, plus PR 1 for preprocessing-aware signatures.

### PR 5: Frontend Mission Control UI Improvements

Owner: Frontend / Mission Control UX Agent

Files/features:

- `apps/mission-control/src/App.tsx`
- `apps/mission-control/src/styles.css`
- `apps/mission-control/src/types.ts` when new backend fields are finalized
- `docs/frontend_mission_control_report.md`

Dependencies: PRs 1, 3, and 4 for complete backend payloads. Can merge earlier if defensive rendering remains.

### PR 6: Orchestrator/System Design Cleanup And Observability

Owner: Orchestrator / System Design Agent

Files/features:

- `services/orchestrator/internal/store/migrations/001_init.sql`
- `docs/orchestrator_system_design_report.md`
- optional follow-up code for event rows, query limits, and metrics

Dependencies: none for additive indexes. Later reliability work depends on PRs 3/4 if moving agent work off request paths.

### PR 7: Champion Export + Live Demo

Owner: Backend/Worker + Frontend

Files/features:

- champion export records and API
- exported model artifact metadata
- worker export support for the selected champion
- lightweight inference endpoint/runtime for demo images
- Mission Control Champion Export panel
- Mission Control live demo screen using held-out/test images
- use-case explanation generated from metrics, objective context, deployment profile, and model metadata

Dependencies: PRs 1, 4, and 5.

Defer from PR 7:

- production deployment hosting
- continuous camera/webcam inference
- model registry integrations
- advanced explainability overlays such as Grad-CAM

### PR 8: Visual Class Exemplars For LLM Planning

Owner: Dataset/Backend + LLM Decision Quality

Files/features:

- class-balanced image exemplar sampling
- downscaling/compression and strict prompt budget controls
- optional multimodal planner context
- planner prompt updates explaining how to use image examples as evidence
- backend validation remains unchanged: LLM still proposes JSON and backend still validates
- audit fields recording whether visual exemplars were used in a planner invocation

Dependencies: PRs 1 and 3.

Defer from PR 8:

- semantic image retrieval across projects
- large image batches in prompts
- allowing LLMs to mutate datasets or choose files directly

### PR 9: `me_ground_truth.md` System Reference

Owner: Final Integration / Documentation

Files/features:

- `docs/me_ground_truth.md`
- explain the system architecture, data model, and end-to-end data movement
- document major orchestrator, worker, frontend, planner, memory, export, and demo flows
- document backend validation boundaries and why LLMs cannot execute directly
- keep this document updated as PRs land

Dependencies: can start immediately, but should be refreshed after PRs 1-8 land.

## Future Execution Prompt

Use this prompt in a future Codex session to continue implementation from this roadmap without restating the whole project context.

```text
You are working in the Model Express repo.

Primary roadmap:
- docs/model_express_agentic_upgrade_roadmap.md

Agent context folder:
- docs/agents_context/README.md
- docs/agents_context/frontend_mission_control/context.md
- docs/agents_context/orchestrator_backend/context.md
- docs/agents_context/python_worker_training/context.md
- docs/agents_context/data_dataset_intelligence/context.md
- docs/agents_context/llm_decision_intelligence/context.md
- docs/agents_context/system_architecture_review/context.md
- docs/agents_context/integration_coordinator/context.md

Supporting docs:
- docs/llm_agent_decision_context.md
- docs/data_preprocessing_agent_report.md
- docs/llm_decision_quality_report.md
- docs/frontend_mission_control_report.md
- docs/orchestrator_system_design_report.md
- docs/smarter_agentic_orchestration_plan.md
- docs/agentic.md
- docs/final_distributed_agentic_vision_training_platform_plan.md

Goal:
Finish implementing the remaining roadmap PRs while preserving Model Express's safety boundary:

LLMs propose structured JSON -> backend validates -> backend stores/schedules -> workers execute.

Do not let any LLM directly execute jobs, mutate datasets, create workers, bypass backend validation, choose arbitrary files, or weaken backend guardrails.

Work style:
- Do not implement this as one giant undifferentiated change.
- Split work by PR boundary and agent ownership.
- If a PR is too large, implement the smallest safe vertical slice and document exactly what remains.
- Keep migrations safe and additive unless explicitly justified.
- Preserve existing propose/autonomous modes and max follow-up rounds.
- Prefer existing repo patterns.
- Do not add Kafka, Redis, NATS, WebSockets, or a workflow engine unless the architecture review shows they are truly justified. Prefer Postgres hardening, idempotency, leases, and SSE first.
- Use subagents in parallel where write scopes do not conflict.
- Do not let subagents edit the same files at the same time. If two agents need the same schema/API field, the Integration Coordinator defines the shared contract and assigns one owner.

Before implementation:
1. Read docs/model_express_agentic_upgrade_roadmap.md.
2. Read docs/agents_context/README.md.
3. Each subagent must read its own context file and any supporting docs listed there.
4. Run `git status --short` and preserve user/other-agent changes. Do not revert unrelated work.
5. Create a short execution plan mapping each PR to owner, files, tests, and dependencies.

Use these agents:

1. Data / Dataset Intelligence Agent
   Context:
   - docs/agents_context/data_dataset_intelligence/context.md
   - docs/data_preprocessing_agent_report.md
   - docs/model_express_agentic_upgrade_roadmap.md

   Owns:
   - Dataset profile contract finalization.
   - Deciding whether `datasets.profile` JSON remains the source of truth or whether `dataset_profiles` rows become the active normalized path.
   - `dataset_artifacts` design/implementation if needed.
   - Annotation XML/JSON, labels CSV, split-file, metadata folder, class hierarchy, and bounding-box artifact parsing.
   - Dataset traits and preprocessing recommendations.
   - Visual class exemplar sampling policy: class-balanced, downscaled, compressed, capped by count/bytes/prompt budget.

   Work on:
   - Complete remaining PR 1 follow-ups that are still safe and needed.
   - Support PR 2 where dataset parsing affects worker training.
   - Own the data side of PR 8 visual exemplars.

2. Python Worker / Training Agent
   Context:
   - docs/agents_context/python_worker_training/context.md
   - docs/data_preprocessing_agent_report.md
   - docs/model_express_agentic_upgrade_roadmap.md

   Owns:
   - Worker-side training execution and test coverage.
   - Real transform behavior for preprocessing fields.
   - Bbox crop vs full-image ablations when annotations are available.
   - Dataset-computed normalization.
   - Explicit split-file training.
   - Advanced augmentation policies beyond the current safe string subset.
   - Champion export artifact production.
   - Lightweight champion inference path for demo images.
   - Worker-side visual exemplar generation if backend delegates it.

   Work on:
   - PR 2: Python worker preprocessing support + preprocessing ablations.
   - Worker slice of PR 7: champion export and inference.
   - Worker slice of PR 8: exemplar generation/processing.

3. Orchestrator / Backend Agent
   Context:
   - docs/agents_context/orchestrator_backend/context.md
   - docs/orchestrator_system_design_report.md
   - docs/model_express_agentic_upgrade_roadmap.md

   Owns:
   - API contracts, validation, stores, migrations, idempotency, execution events, and backend safety.
   - Dataset profile/artifact persistence if Data Agent defines the contract.
   - Advanced preprocessing validation after Worker Agent supports it.
   - Champion export records and APIs.
   - Demo image listing and prediction APIs.
   - Visual exemplar APIs and invocation audit fields.
   - Planner validation/rejection feed.
   - Durable idempotency keys, job leases/requeue, and SSE/event stream if included in the current implementation scope.

   Work on:
   - Backend pieces of PR 1 follow-ups, PR 2 validation, PR 7, PR 8, and selected PR 6 reliability/observability follow-ups.
   - Keep all execution-facing changes behind deterministic backend validation.

4. LLM Decision Intelligence Agent
   Context:
   - docs/agents_context/llm_decision_intelligence/context.md
   - docs/llm_decision_quality_report.md
   - docs/llm_agent_decision_context.md
   - docs/model_express_agentic_upgrade_roadmap.md

   Owns:
   - Planner prompt/schema upgrades.
   - Preprocessing-aware prompt examples using the finalized PR 1/2 contract.
   - Visual exemplar prompt/context design for PR 8.
   - Planner tests for visual exemplar budget handling and JSON-only output.
   - Rejected options, candidate scoring, strategy scorecard use, and semantic memory follow-up if scoped.

   Work on:
   - PR 3 remaining prompt/schema updates.
   - PR 4 refinements only where necessary.
   - LLM side of PR 8 visual exemplars.

5. Frontend / Mission Control Agent
   Context:
   - docs/agents_context/frontend_mission_control/context.md
   - docs/frontend_mission_control_report.md
   - docs/model_express_agentic_upgrade_roadmap.md

   Owns:
   - Mission Control UI and TypeScript types.
   - Section navigation/tabs for the long workspace.
   - Typed display for preprocessing fields once backend contracts are stable.
   - Candidate `score_components`, rejected options, scorecards, and invocation/rejection visibility.
   - Champion Export panel.
   - Champion Use Case explanation box.
   - Live champion demo screen showing held-out/test image, predicted label, true label if known, confidence, top-k results, latency, and correctness.

   Work on:
   - PR 5 remaining UX improvements.
   - Frontend slice of PR 7 champion export + live demo.

6. System Architecture Review Agent
   Context:
   - docs/agents_context/system_architecture_review/context.md
   - docs/orchestrator_system_design_report.md
   - docs/model_express_agentic_upgrade_roadmap.md

   Owns:
   - Architecture review and risk analysis.
   - Reviewing idempotency, job leases, SSE/eventing, polling pressure, migration safety, and export/demo architecture.
   - Recommending whether PR 6 reliability work should be implemented now or deferred.

   Work on:
   - Review proposed backend/worker/frontend changes before final integration.
   - Do not add Kafka/Redis/NATS unless clearly justified.
   - Produce notes for the final report.

7. Integration Coordinator Agent
   Context:
   - docs/agents_context/integration_coordinator/context.md
   - docs/agents_context/README.md
   - docs/model_express_agentic_upgrade_roadmap.md

   Owns:
   - Coordinating subagents and PR boundaries.
   - Resolving schema/API conflicts.
   - Checking that frontend expectations match backend responses.
   - Checking that planner proposals are validated by backend and supported by workers.
   - Checking migrations and store implementations.
   - Running final verification.
   - Writing final documentation.

   Work on:
   - Create or update docs/me_ground_truth.md.
   - Create docs/model_express_pr_completion_report.md after implementation.
   - The report must be ordered by PR and include:
     - PR number and title.
     - Owning agent(s).
     - What changed.
     - Files changed.
     - New APIs, schemas, migrations, config fields, and UI surfaces.
     - Tests/build checks run and results.
     - What remains deferred and why.
     - Any known risks or manual demo steps.

Required implementation targets:

PR 1 remaining:
- Resolve or document the `datasets.profile` JSON vs `dataset_profiles` table source-of-truth path.
- Add/complete dataset artifact persistence only if it is needed for PR 2, PR 7, or PR 8.
- Keep preprocessing schema aligned across backend, worker, LLM prompts, and frontend.

PR 2:
- Add worker tests for transform construction, sampler/loss selection, and profile artifact detection.
- Implement the next safe slice of preprocessing ablations:
  - dataset-computed normalization if feasible,
  - explicit split-file training if feasible,
  - bbox crop vs full-image ablation if annotations are available,
  - advanced augmentation policy support only if backend validation and tests land with it.

PR 3:
- Update planner prompt/examples to include finalized preprocessing fields:
  - `resolution_strategy`
  - `preprocessing`
  - `augmentation_policy`
  - `sampling_strategy`
  - class balancing/loss options.
- Keep output valid JSON and strongly typed.
- Add tests for preprocessing-aware planner outputs.

PR 4:
- Ensure candidate ranking, rejected options, scorecards, and memory remain compatible with any new preprocessing, export, or exemplar fields.
- Add tests only where behavior changes.

PR 5:
- Add section navigation/tabs if it improves demo clarity.
- Add UI for candidate score components and validation/rejection feed if backend exposes it.
- Add frontend support for champion export/demo APIs from PR 7.

PR 6:
- Implement only low-risk reliability/observability improvements that fit current scale.
- Strong candidates: event rows for job lifecycle, idempotency tests, positive epoch validation, or SSE if scope allows.
- Defer job leases or async automation if they would balloon the change.

PR 7:
- Add champion export records/API and worker export support.
- Add model metadata/use-case explanation.
- Add lightweight inference/demo API.
- Add Mission Control export/demo panel.
- Demo should use held-out/test images when possible and show image, prediction, true label if available, confidence/top-k, latency, and correctness.

PR 8:
- Add class-balanced downscaled visual exemplars for planner context.
- Add strict budget caps and audit fields.
- Add prompt/context support so LLM can use images as evidence.
- LLM still returns JSON only; backend validation remains the execution gate.

PR 9:
- Create docs/me_ground_truth.md.
- Explain the complete system: architecture, data model, major flows, functions/modules that move data, dataset/profile/training/evaluation/planning/champion/export/demo flow, LLM context, memory, and backend validation.
- Keep it as a durable system reference, not a changelog.

Final integration requirements:
- Run appropriate checks after code changes:
  - `go test ./...` from `services/orchestrator`
  - `npm run build` from `apps/mission-control`
  - Python compile or tests for touched worker files
  - any new focused tests added by agents
- If a check fails, fix it if feasible. If not feasible, document:
  - failing command
  - failing file/test
  - likely cause
  - suggested fix
- Update relevant files in docs/agents_context if a PR changes durable responsibilities or contracts.
- Create docs/model_express_pr_completion_report.md with the final PR-by-PR explanation.
- Final assistant response should summarize:
  - major implemented changes,
  - report file path,
  - tests run,
  - any deferred work.
```

## Deferred Architecture

- Do not add Kafka now.
- Do not add Redis Streams or NATS now.
- Prefer Postgres row locks, targeted queries, durable idempotency, job leases, and SSE first.
- Revisit Redis Streams or NATS only when multiple orchestrator instances, independent consumers, or higher event volume make Postgres-backed events awkward.
