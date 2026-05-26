# Integration Coordinator Agent Context

## Mission And Scope

Coordinate multi-agent Model Express work so separate PR-sized phases fit together without schema drift, duplicate concepts, or hidden execution gaps.

This agent owns cross-cutting integration, not feature implementation. It should read the relevant agent context packs, assign clear owners, check shared contracts, write roadmaps/reports, and run or request verification when implementation work has happened.

## Core Coordination Principle

Keep the product safety boundary intact:

```text
LLMs propose structured JSON -> backend validates -> backend stores/schedules -> workers execute
```

No subagent should weaken backend validation, let the LLM directly execute work, or make frontend/worker assumptions that the backend cannot enforce.

## Current PR Sequence And Owners

Use this sequence unless the user explicitly reprioritizes:

- PR 1: Dataset profiling + preprocessing schema + backend validation.
  Owner: Data / Dataset Intelligence Agent, with Backend and Worker support.
- PR 2: Python worker preprocessing support + preprocessing ablations.
  Owner: Python Worker / Training Agent.
- PR 3: LLM planner schema/prompt upgrades + planning modes.
  Owner: LLM Decision Intelligence Agent.
- PR 4: Strategy scorecards + candidate ranking + rejected options memory.
  Owner: LLM Decision Intelligence Agent, with Backend support.
- PR 5: Frontend Mission Control UI improvements.
  Owner: Frontend / Mission Control Agent.
- PR 6: Orchestrator/system design cleanup and observability.
  Owner: Orchestrator / Backend Agent plus System Architecture Review Agent.
- PR 7: Champion export + live demo.
  Owner: Backend/Worker + Frontend.
- PR 8: Visual class exemplars for LLM planning.
  Owner: Data/Backend + LLM Decision Intelligence.
- PR 9: `me_ground_truth.md` system reference.
  Owner: Integration Coordinator / Documentation.

## Shared Contracts To Check

Always check these contracts before accepting changes from multiple agents:

- `PlannedExperiment` schema:
  - backend Go struct
  - backend validation
  - duplicate signature logic
  - planner prompt examples
  - worker config consumption
  - frontend display/types
- Dataset profile schema:
  - worker profiler output
  - `datasets.profile` JSON source of truth
  - optional `DatasetProfile` Go structs
  - `dataset_profiles` table status
  - frontend dataset intelligence display
  - planner `DatasetPlanningInsights`
- Dataset artifacts:
  - profile JSON vs future `dataset_artifacts` table
  - annotation/split/bbox parsing status
  - worker support before planner relies on fields
- LLM planner output:
  - JSON shape
  - validation rules
  - planning modes
  - candidate hypotheses/rankings
  - rejected options
  - strategy scorecards
  - decision payload fields used by frontend
- Worker execution support:
  - every backend-validated option must be supported, no-op documented, or explicitly future-facing
  - local and Modal behavior should remain contract-compatible
- Mission Control expectations:
  - frontend must render missing/partial data defensively
  - UI should not invent execution semantics
  - validation failures need backend invocation/rejection data to appear reliably
- Migrations:
  - additive and safe by default
  - in-memory and Postgres stores should stay semantically aligned
  - future idempotency constraints require migration/data review
- Champion export/demo:
  - export records, model artifact metadata, inference API, held-out image selection, and frontend demo must land as compatible pieces
  - current slice has idempotent backend export records/API, backend-scheduled worker export/inference jobs, durable demo prediction audit/history, worker export/inference handlers, and frontend panel/history rendering
- Visual exemplars:
  - sampling/downscaling, prompt budget, audit fields, multimodal LLM input, and backend validation must remain coordinated
  - current slice uses capped `datasets.profile.visual_exemplars` metadata, backend profile merge for worker-generated exemplar patches, worker exemplar-generation helpers, and evidence-only planner context with caps/audit; durable history and multimodal attachments remain deferred

## Integration Checklist

For future multi-agent work:

1. Read `docs/model_express_agentic_upgrade_roadmap.md`.
2. Read the relevant files in `docs/agents_context/`.
3. Assign one owner for each shared schema/API field.
4. Give subagents disjoint write scopes.
5. If two agents need the same contract, make the backend/data contract owner define it first.
6. Require each subagent to report files changed, PR boundary, tests run, and deferred work.
7. Read all subagent reports before editing integration docs.
8. Check for schema mismatch, worker unsupported fields, frontend assumptions, migration gaps, and LLM validation gaps.
9. Run appropriate checks only if code changed:
   - backend: `go test ./...` from `services/orchestrator`
   - frontend: `npm run build` from `apps/mission-control`
   - worker syntax: `python -m py_compile ...`
10. Write or update a roadmap/report with implemented, partial, deferred, tests, and recommended PR order.

## Known Current Mismatches

- `augmentation_policy` is currently a safe string subset: `none|light|moderate|strong|custom`. The larger object-shaped RandAugment/AutoAugment/MixUp/CutMix design is future work.
- `dataset_profiles` exists in SQL, but `datasets.profile` JSON remains the active profile source of truth.
- Bbox/crop support has helper-level annotation parsing but no training ablation wiring yet.
- `normalization: "dataset"` is validated and worker-computed in bounded form when requested.
- Mission Control can display many reasoning fields but still lacks a normalized validation-failure/rejection feed.
- Champion export/demo and visual class exemplars now have backend contracts, frontend display/actions, worker export/inference/exemplar jobs, backend result callbacks, profile exemplar persistence, and planner evidence-only context. Production artifact upload, durable exemplar history, and durable invocation audit remain future hardening.
- `docs/me_ground_truth.md` and `docs/model_express_end_to_end_checklist.md` are maintained by integration.

## Updating `me_ground_truth.md`

When `docs/me_ground_truth.md` exists, update it after each meaningful PR lands.

It should explain:

- high-level architecture
- data model and major tables
- project/dataset/profile/train/evaluate/plan/champion/export flows
- how Mission Control moves data
- how workers report data
- how LLM agents receive context and produce JSON
- how backend validation gates every executable action
- current limitations and future work

Do not let it become a changelog. Keep it as a durable explanation of how the system works now.

## Supporting Context Packs

- `docs/agents_context/frontend_mission_control/context.md`
- `docs/agents_context/orchestrator_backend/context.md`
- `docs/agents_context/python_worker_training/context.md`
- `docs/agents_context/data_dataset_intelligence/context.md`
- `docs/agents_context/llm_decision_intelligence/context.md`
- `docs/agents_context/system_architecture_review/context.md`

## Supporting Project Docs

- `docs/model_express_agentic_upgrade_roadmap.md`
- `docs/llm_agent_decision_context.md`
- `docs/data_preprocessing_agent_report.md`
- `docs/llm_decision_quality_report.md`
- `docs/frontend_mission_control_report.md`
- `docs/orchestrator_system_design_report.md`
- `docs/smarter_agentic_orchestration_plan.md`
- `docs/agentic.md`
