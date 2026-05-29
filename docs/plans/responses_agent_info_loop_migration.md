# Responses Information-Request Loop Implementation Plan

## Summary
No files were changed in Plan Mode. The implementation should start by creating `docs/plans/responses_agent_loop_migration_plan.md` with this PR-split plan.

Core migration:

```text
agent receives compact context
-> agent requests approved backend information
-> backend validates tool name, args, scope, and output caps
-> backend returns bounded answer
-> agent may request more information
-> agent emits final structured JSON
-> backend validation/storing/scheduling remains authoritative
```

Selected defaults:
- Migrate all LLM paths: Planner, Training Monitor, Visual Dataset Analysis.
- Use hybrid stored Responses state inside one active loop plus local backend memory for durable history.
- On plateau, prefer mixed-risk batches: safe/control + diagnostic + bold challenger.

## PR Split
### PR 0: Migration Doc
Create `docs/plans/responses_agent_loop_migration_plan.md`.

Include:
- architecture,
- information-request loop contract,
- tool catalogs,
- PR dependency graph,
- rollout steps,
- test plan,
- safety boundaries.

### PR 1: Shared Responses Runtime
Owner: orchestrator LLM runtime subagent.

Add Responses beside the current chat client.

Key work:
- Keep `local` and `openai_compatible` on `/chat/completions`.
- Use `/responses` only for `provider=openai` with `MODEL_EXPRESS_LLM_API_STYLE=responses`.
- Add config for API style, reasoning effort, plateau reasoning effort, stored responses, and max tool rounds.
- Add request/response types for response IDs, function calls, tool outputs, reasoning effort, and final JSON extraction.
- Add `llm.NewClient(config)` while preserving existing `JSONGenerator` behavior.

Tests:
- Chat path remains unchanged.
- Responses path posts to `/responses`.
- Final JSON extraction works.
- Function calls and `previous_response_id` are parsed.
- Unsupported providers ignore `responses` style and stay on chat.

### PR 2: Backend Tool Registry
Owner: planner/backend subagent.

Add the allowlisted backend question registry without changing final scheduling behavior.

Planner tools:
- `dataset_profile`
- `visual_summary`
- `memory`
- `scorecards`
- `ledger`
- `run_details`
- `per_class_detail`
- `mechanism_coverage`
- `model_catalog`
- `recent_planner_failures`
- `validate_candidate_experiments`

Rules:
- Tools are scoped to the active project/dataset/plan/job set.
- Unknown tools, malformed args, cross-scope IDs, and oversized outputs are rejected.
- Raw prompts, raw images, local paths, signed URIs, and unbounded profiles are never returned.
- `validate_candidate_experiments` is dry-run only and never writes rows or schedules jobs.

### PR 3: Experiment Planner Information Loop
Owner: planner/backend subagent.

Move planner from one-shot JSON to the real loop:

```text
planner_context_snapshot
-> information_requests
-> tool_results
-> optional candidate dry-run validation
-> final ExperimentPlanningRecommendation JSON
```

Final output still flows through:
- existing planner schema validation,
- candidate ranking,
- mechanism contract validation,
- dataset evidence validation,
- novelty checks,
- stop guards,
- decision creation,
- optional follow-up scheduling.

Plateau mode should request more detail before finalizing: prior failed mechanisms, best/worst runs, per-class failures, visual evidence, and validation-gated mechanisms.

### PR 4: Training Monitor Information Loop
Owner: training-monitor subagent.

Let the monitor request run-scoped details before writing memory.

Monitor tools:
- `get_epoch_metrics`
- `get_training_evaluation`
- `get_memory_records`
- `get_plan_config`
- `get_job_config`
- `get_objective_context`
- `validate_recommendation_draft`

Rules:
- Monitor remains run-scoped.
- It cannot propose experiments, jobs, commands, plans, or scheduling actions.
- Final output remains `TrainingEvaluationRecommendation`.

### PR 5: Visual Dataset Analysis Responses Loop
Owner: worker/visual-analysis subagent.

Move visual analysis to Responses behind config.

Visual tools:
- `get_dataset_metadata_summary`
- `get_sample_manifest`
- `get_allowed_operations`
- `validate_visual_analysis_draft`

Rules:
- Images stay only in the prepared bounded sample input.
- Tools never return local paths, storage URIs, image bytes, raw prompts, or mutation authority.
- Final output still passes existing evidence-only visual schema validation.
- Existing repair remains only for non-safety schema errors.

### PR 6: Mission Control Observability
Owner: frontend subagent.

Add an Agent Invocation Audit view in the Agents tab.

Show:
- API style,
- provider/model,
- reasoning effort,
- validation status/error,
- tool rounds,
- tool names,
- rejected tool calls,
- dry-run validation results,
- final decision linkage.

Hide:
- raw prompts,
- raw outputs,
- image URLs/base64,
- sample manifests,
- large nested tool payloads.

### PR 7: Integration Hardening
Owner: integration subagent.

Add fake-tested end-to-end coverage.

Tests must prove:
- approved information requests execute and feed back into the loop,
- unsupported tool calls are rejected and recorded,
- tool calls cannot schedule or mutate anything,
- final JSON still must pass backend validation,
- chat fallback still works,
- visual safety boundaries remain intact,
- UI build passes.

Commands:
- `go test ./...` from `services/orchestrator`
- `python -m pytest` from `services/worker`
- `npm run build` from `apps/mission-control`

## Safety Rules
LLMs never directly create plans, jobs, workers, champions, exports, inference runs, or dataset mutations. Tool calls are questions, not actions. Only final validated backend code may store decisions or schedule work.
