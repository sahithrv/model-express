# Responses Agent Loop Migration Plan

## Architecture
Model Express agents move from one-shot JSON generation to a bounded information-request loop:

```text
compact backend context
-> LLM asks approved backend information questions
-> backend validates tool name, arguments, scope, and output caps
-> backend returns bounded JSON answers
-> LLM may ask follow-up questions
-> LLM emits final structured JSON
-> backend validation, persistence, scheduling, exports, workers, inference, and dataset mutation remain authoritative
```

Responses mode is opt-in. `local` and `openai_compatible` providers keep the existing `/chat/completions` behavior. The OpenAI provider uses `/responses` only when `MODEL_EXPRESS_LLM_API_STYLE=responses`.

## Shared Runtime Contract
- Config fields: API style, reasoning effort, plateau reasoning effort, stored response state, and max tool rounds.
- The runtime exposes chat-compatible `GenerateJSON` plus a Responses loop interface for agents with approved tools.
- Each tool result is JSON-bounded and records whether the request was accepted or rejected.
- `previous_response_id` is used inside one active Responses loop when stored response state is enabled.
- Final JSON extraction accepts `output_text` or text output items and still returns raw JSON bytes to existing validators.

## Tool Catalogs
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

Training Monitor tools:
- `get_epoch_metrics`
- `get_training_evaluation`
- `get_memory_records`
- `get_plan_config`
- `get_job_config`
- `get_objective_context`
- `validate_recommendation_draft`

Visual tools:
- `get_dataset_metadata_summary`
- `get_sample_manifest`
- `get_allowed_operations`
- `validate_visual_analysis_draft`

## Safety Boundaries
- Tool calls are questions, never actions.
- LLMs never directly create plans, jobs, workers, champions, exports, inference runs, or dataset mutations.
- Unknown tools, malformed arguments, cross-scope IDs, and oversized outputs are rejected and recorded.
- Raw prompts, raw outputs, raw images, local paths, storage URIs, signed URLs, credentials, sample manifests, and unbounded payloads are not exposed in audit UI or backend tool answers.
- `validate_candidate_experiments` is dry-run only: it validates candidate payloads through existing backend gates but never writes rows or schedules jobs.
- Visual analysis remains evidence-only and cannot propose experiments or mutate datasets.

## PR Dependency Graph
1. Shared runtime support can land first and must preserve chat behavior.
2. Backend tool registries depend on the runtime tool loop types.
3. Planner and Training Monitor loops depend on the registry and runtime.
4. Visual analysis Responses mode is independent on the worker side but follows the same loop semantics.
5. Mission Control audit can read existing invocation fields plus sanitized loop metadata.
6. Integration hardening verifies the safety boundary end to end.

## Rollout
1. Default to chat mode everywhere.
2. Enable orchestrator Responses mode with `MODEL_EXPRESS_LLM_PROVIDER=openai` and `MODEL_EXPRESS_LLM_API_STYLE=responses`.
3. Tune `MODEL_EXPRESS_LLM_MAX_TOOL_ROUNDS` and reasoning effort per environment.
4. Enable worker visual Responses mode separately behind visual LLM config.
5. Use Agent Invocation Audit to confirm accepted/rejected tool calls and dry-run validation before enabling autonomous scheduling.

## Test Plan
- Runtime fake HTTP tests for chat fallback, `/responses` posting, final JSON extraction, function calls, `previous_response_id`, and provider gating.
- Planner tests for allowed tool execution, rejected tool calls, dry-run validation with no schedule/write side effects, backend validation retry compatibility, and final JSON validation.
- Training Monitor tests for run-scoped tools and rejection of experiment/scheduling authority.
- Worker visual tests for Responses mode, safe tool answers, no path/image leaks, and existing evidence-only schema validation.
- Mission Control build plus audit rendering checks that hide raw prompts, raw outputs, images/base64, sample manifests, and huge nested payloads.

