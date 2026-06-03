# LLM Efficiency Plan

## Goal

Make `gpt-5.4-mini` the normal operating model for agentic planning while keeping planner quality close to the current `gpt-5.4` behavior.

The practical target is that roughly $10 of LLM credits should last closer to a week of active development/testing instead of disappearing in a day. The main lever is shrinking Experiment Planner prompts without removing the decision-critical information that makes the agent useful.

## Current Baseline

Recent local Postgres inspection showed the largest `experiment_planner` invocations are approximately:

- Sent prompt: 78k-83k estimated input tokens.
- Stored audit context: 66k-70k estimated tokens.
- Output: usually 2k-9k estimated tokens.
- Valid planner calls observed: 44 / 47, about 94%.
- `visual_dataset_analysis` is much smaller, around 13k average input tokens, and is not the first optimization target.
- Memory embedding usage was also observed as a likely cost leak: one local run showed 393 text-embedding calls. The code has two spend paths: indexing source memory cards and embedding each retrieval query. Retrieval query embeddings can be spent even when retrieval is log-only or returns no useful cards, so embedding usefulness must be measured separately from planner prompt size.

Important interpretation:

- `input_messages` is the approximate prompt sent to the model.
- `input_context` is an audit copy and should not be counted as billed input unless it is embedded in `input_messages`.
- The real cost problem is still planner input size.
- The DB/audit duplication is a secondary storage/noise problem.
- The embedding cost problem is separate: embeddings may be useful for vector memory, but every automatic source-card index and every retrieval-query embedding should be attributable to an actual retrieved/used memory benefit.

## Targets

Near-term target:

- Planner average sent input <= 35k tokens.
- Planner high-end sent input <= 45k tokens.
- Planner valid-output rate >= 90% on mini.
- Embedding calls are measured by purpose and source, with retrieval-query calls marked useful only when they return cards that are injected or otherwise influence ranking.

Strong target:

- Planner average sent input <= 25k tokens.
- Planner high-end sent input <= 35k tokens.
- Planner valid-output rate remains within 5 percentage points of the current baseline.
- Embedding calls per active project day are reduced by at least 50% unless retrieval-hit evidence shows they are helping.

Aggressive target:

- Planner average sent input <= 18k tokens.
- Planner high-end sent input <= 25k tokens.
- Planner uses distilled memories first and raw memory only as fallback detail.
- Embeddings are reserved for distilled/high-value cards and cached retrieval queries; low-value raw cards use lexical fallback until promoted.

## Contract Preservation Requirement

Efficiency work must not make the Experiment Planner produce a different or weaker JSON contract.

The planner must continue returning the same backend-validated structure it returns today. Do not remove currently required top-level fields or required nested fields just to reduce output tokens. In particular, preserve the fields enforced by planner validation, including `summary`, `rationale`, `decision_type`, `confidence`, `planning_mode`, `hypothesis`, `deterministic_diagnosis_used`, `evidence_used`, `expected_failure_modes`, `dataset_preprocessing_rationale`, `changed_variables`, `success_criteria`, `stop_condition`, `deployment_tradeoff`, `rejected_options`, `why_can_beat_champion`, `candidate_hypotheses`, `proposal_mechanisms`, and the full `proposed_experiments` shape.

For `ADD_EXPERIMENTS`, preserve `candidate_hypotheses` with complete `experiment_config` objects. Each concrete experiment candidate still needs backend-valid `template`, `model`, `epochs`, `batch_size`, and `learning_rate` unless AutoML explicitly covers those parameters. Backend ranking may populate final `candidate_rankings`, `proposed_experiments`, and `proposal_mechanisms`, but the planner must still emit schema-valid candidates that can pass existing validation.

Allowed output optimization:

- Shorten free-text values with clear max-length guidance.
- Prefer concise evidence IDs and mechanism labels.
- Avoid repeated paragraphs.
- Keep arrays capped where backend validation already allows caps.
- Add backend-generated summaries outside the LLM response if needed.

Not allowed:

- Removing fields that currently prevent backend validation errors.
- Weakening backend validation to make mini look better.
- Changing the output schema without updating validation, tests, and replay fixtures.
- Replacing structured experiment fields with vague prose.

## PR 1: Exact Usage Capture

Owner: orchestrator LLM client and store.

Approximate byte-based token estimates are useful, but optimization should be driven by exact provider usage when available.

Implementation:

- Extend LLM response result types to carry optional usage metadata.
- Parse usage from Chat Completions and Responses responses when present.
- Capture at least:
  - input tokens
  - output tokens
  - total tokens
  - cached input tokens when provided
  - reasoning tokens when provided
  - request model
  - API style
  - tool round count
- Store usage on `agent_invocations.input_context.invocation_runtime.llm_usage` or a first-class JSON field if a migration is acceptable.
- Keep byte-based estimates as fallback when provider usage is missing.

Acceptance:

- Every new planner and monitor invocation shows exact usage when the provider returns it.
- Mission Control and SQL can compare exact tokens against approximate estimates.
- Missing usage does not fail the agent call.

## PR 2: Embedding Spend Audit And Gating

Owner: orchestrator memory, embeddings, API, and Mission Control.

Embedding spend must be accountable. The current code can spend embeddings in two ways:

- Source indexing through `MemoryIndexer.IndexCards`, including agent memory records, strategy scorecards, dataset profile cards, accepted visual-analysis cards, and preprocessing-hypothesis cards.
- Retrieval-query embeddings through memory retrieval for the Planner and Training Monitor.

The second path can cost money even when retrieval is log-only or returns no cards that reach the planner. That may be useful during rollout measurement, but it should not be invisible. Also, live indexing currently relies on the final DB upsert for de-duplication; that prevents duplicate rows, but it does not necessarily prevent duplicate provider calls before the upsert. Backfill is capped, but the current default cap is high enough that a single backfill can spend hundreds of calls.

Implementation:

- Capture embedding usage per call:
  - purpose: `source_index` or `retrieval_query`
  - source table/source id for indexing
  - retrieval purpose for query embeddings
  - project id, dataset id, plan id, job id when available
  - input text byte estimate
  - model
  - dimensions
  - exact usage/cost if provider returns it, otherwise approximate request count
  - retrieval result count
  - injected into planner/monitor: true/false
  - log-only: true/false
- Add a durable or diagnostic summary for embedding calls. A small table is preferred if cost reporting will live in Mission Control; otherwise start with structured execution events.
- Add retrieval-query cache keyed by:
  - model
  - dimensions
  - project/dataset scope
  - normalized query text hash
  - retrieval purpose
- Add source-card indexing guardrails:
  - do not embed pending/invalidated/unsafe cards
  - prefer distilled/high-confidence cards first
  - cap automatic indexing per project/day
  - cap backfill batches more conservatively than the current default when using paid embeddings
  - pre-check existing `(source_table, source_id, embedding_model)` before calling the provider during live indexing
  - do not re-embed unchanged source cards for the same model/dimensions/content hash
  - add an embedding text hash to detect unchanged source text even when surrounding metadata changes
  - batch source-card embeddings when the provider/API supports multiple inputs in one request
- Add retrieval-query guardrails:
  - if retrieval is log-only, optionally use lexical search only unless `MODEL_EXPRESS_MEMORY_LOG_ONLY_EMBEDDINGS=true`
  - skip query embedding when there are too few indexed cards to justify semantic retrieval
  - skip query embedding if a recent identical query was already embedded
  - downgrade to lexical retrieval when embedding budget is exhausted

Proposed flags:

```env
MODEL_EXPRESS_MEMORY_EMBEDDING_DAILY_BUDGET_USD=1.00
MODEL_EXPRESS_MEMORY_EMBEDDING_MAX_CALLS_PER_DAY=100
MODEL_EXPRESS_MEMORY_BACKFILL_LIMIT=50
MODEL_EXPRESS_MEMORY_RETRIEVAL_QUERY_CACHE_TTL_SECONDS=3600
MODEL_EXPRESS_MEMORY_LOG_ONLY_EMBEDDINGS=false
MODEL_EXPRESS_MEMORY_INDEX_DISTILLED_ONLY=false
```

Acceptance:

- A user can answer how many embedding calls were source indexing vs retrieval queries.
- A user can answer how many retrieval-query embedding calls returned cards that were actually injected into planner/monitor context.
- Re-running the same retrieval query in a short window does not call the embedding model again.
- Live indexing an already-embedded unchanged source does not call the embedding provider again.
- Backfill cannot silently spend hundreds of embedding calls without visible progress/cost telemetry.
- The planner can still use memory when embeddings are disabled or budget-exhausted through lexical/structured fallback.

Useful SQL after this PR:

```sql
SELECT
  date_trunc('day', created_at) AS day,
  metadata->>'purpose' AS purpose,
  metadata->>'embedding_model' AS model,
  COUNT(*) AS embedding_calls,
  COUNT(*) FILTER (WHERE (metadata->>'retrieved_count')::int > 0) AS retrieval_hits,
  COUNT(*) FILTER (WHERE metadata->>'injected' = 'true') AS injected_hits
FROM embedding_usage_events
GROUP BY day, purpose, model
ORDER BY day DESC, embedding_calls DESC;
```

Useful current-state SQL before this PR:

```sql
SELECT
  source_table,
  kind,
  embedding_model,
  COUNT(*) AS embedded_cards
FROM agent_memory_embeddings
GROUP BY source_table, kind, embedding_model
ORDER BY embedded_cards DESC;
```

```sql
SELECT
  payload->>'purpose' AS purpose,
  COUNT(*) AS retrieval_checks,
  SUM(COALESCE((payload->>'retrieved_count')::int, 0)) AS total_retrieved_cards,
  COUNT(*) FILTER (WHERE COALESCE((payload->>'retrieved_count')::int, 0) > 0) AS checks_with_hits
FROM execution_events
WHERE event_type = 'MEMORY_RETRIEVAL_LOGGED'
GROUP BY payload->>'purpose'
ORDER BY retrieval_checks DESC;
```

## PR 3: Section-Level Prompt Telemetry

Owner: orchestrator agents.

Before shrinking anything, record where the prompt weight comes from.

Implementation:

- Add per-section byte and approximate-token telemetry to planner prompt budget.
- Measure both static instructions and dynamic context sections.
- Suggested section names:
  - `static_instructions`
  - `output_schema`
  - `planner_context_snapshot_total`
  - `dataset_card`
  - `objective_context`
  - `champion_card`
  - `completed_experiment_log`
  - `project_trajectory_card`
  - `training_dynamics_card`
  - `per_class_error_card`
  - `deployment_card`
  - `mechanism_coverage_card`
  - `backend_gated_methods`
  - `label_quality_card`
  - `search_coverage`
  - `strategy_lessons`
  - `retrieved_memory`
  - `blocked_repeats`
  - `model_catalog`
  - `optimizer_feedback`
  - `validation_feedback`
- Store the telemetry under `planner_context_snapshot.prompt_budget.section_estimates`.

Acceptance:

- A SQL query can rank planner prompt sections by estimated token weight.
- The top three prompt offenders are visible after every planner call.
- Existing planner behavior is unchanged.

Useful SQL after this PR:

```sql
SELECT
  id,
  created_at,
  model,
  input_context #> '{planner_context_snapshot,prompt_budget,section_estimates}' AS section_estimates
FROM agent_invocations
WHERE agent_name = 'experiment_planner'
ORDER BY created_at DESC
LIMIT 10;
```

## PR 4: Static Prompt Compression

Owner: orchestrator agents.

The static planner prompt should be clear enough for mini but not essay-length.

Implementation:

- Split `experimentPlannerJSONRequest` into:
  - short role instruction
  - compact hard rules
  - compact output contract
  - dynamic context blob
- Remove repeated rule variants that backend validation already enforces.
- Replace long explanations with concise directives.
- Keep the rules that materially prevent bad plans:
  - backend validation owns execution
  - use only catalog models/templates
  - cite evidence from planner context
  - avoid duplicate mechanisms/signatures
  - treat live latency as a budget/tiebreaker
  - do not propose label mutation
  - do not make shallow epoch/lr-only changes
- Add max-length guidance for free-text fields.

Acceptance:

- Static prompt section size drops by at least 40%.
- Planner valid-output rate on existing tests remains unchanged.
- Mini can still produce schema-valid planner output in smoke tests.

## PR 5: Compact Planner Context Snapshot V2

Owner: orchestrator agents.

Create a `planner_context_snapshot_v2` that carries the same decision value with fewer tokens.

Implementation:

- Keep decision-critical cards:
  - project card
  - dataset card
  - objective context
  - champion card
  - project trajectory
  - failure diagnosis
  - training dynamics
  - per-class error card
  - deployment card
  - mechanism coverage
  - blocked repeats
  - strategy lessons
  - model catalog
- Replace verbose histories with delta summaries:
  - champion card
  - last N run deltas
  - best N positive deltas
  - worst N failed attempts
  - exhausted mechanisms
  - blocked repeats
  - best/worst lessons
- Do not send all prior plans, all run summaries, full evaluations, full signatures, or full memory payloads.
- Preserve IDs so the audit trail can point back to full DB rows.
- Use compact readable fields rather than cryptic one-letter keys.
- Add a rollout flag:
  - `MODEL_EXPRESS_PLANNER_CONTEXT_VERSION=v1|v2`
  - default to `v1` until replay checks pass.

Acceptance:

- Snapshot V2 contains all fields needed by backend validation and candidate ranking.
- Snapshot V2 dynamic context is at least 50% smaller than V1 on the current project.
- V1 can still be selected for rollback.

## PR 6: Model Catalog Compression

Owner: orchestrator planner/catalog.

The planner does not need verbose model descriptions on every call.

Implementation:

- Replace full model specs in the planner context with compact entries:

```json
[
  {"id":"yolo11n.pt","task":"detect","tier":"fast","default":true,"latency":"low"},
  {"id":"yolo11s.pt","task":"detect","tier":"balanced","latency":"medium"},
  {"id":"efficientnet_b0","task":"classify","tier":"fast","size":224}
]
```

- Keep only backend-valid fields the planner must choose from:
  - model id
  - task type
  - model kind/family
  - latency tier
  - quality tier
  - default image size when relevant
  - eligibility notes
- Move long rationale text into docs or backend-owned validation errors rather than every prompt.

Acceptance:

- Model catalog prompt section drops by at least 60%.
- Planner still chooses only backend-valid model IDs.
- YOLO/classification task separation remains explicit.

## PR 7: Distilled Memory First

Owner: memory layer and planner input builder.

Raw memory is valuable evidence, but it is too bulky for every planner call. The planner should receive ranked distilled lessons first.

Implementation:

- Add a distilled memory card shape, even if backed initially by existing `agent_memory_records`:

```json
{
  "id": "distilled_memory_123",
  "task_type": "object_detection",
  "applies_when": ["yolo", "small_objects", "live_inference"],
  "lesson": "Use YOLO nano/small first for live detection, then test medium only when latency remains within budget.",
  "evidence_ids": ["scorecard_1", "job_9", "memory_4"],
  "confidence": 0.82,
  "outcome": "improved_champion"
}
```

- Planner retrieval order:
  - top 3-8 distilled memories
  - top 3 failed/blocked lessons
  - raw memory cards only when no distilled memory exists
- Cap each distilled lesson to a small byte budget.
- De-duplicate memories by mechanism and applies-when signature.
- Keep source IDs for audit, but do not inline source payloads.

Acceptance:

- Retrieved memory section drops by at least 50% when memory is present.
- Planner still cites memory IDs in `evidence_used`.
- Raw memory fallback remains available.

## PR 8: Output Length Guardrails Without Contract Removal

Owner: orchestrator agents and validation.

Planner output cost is smaller than input cost, but long justification fields also encourage long reasoning. This PR must preserve the current planner output contract so backend validation does not regress.

Implementation:

- Identify free-text fields whose values can be shortened without removing the field:
  - long rationale paragraphs
  - repeated deployment tradeoff text
  - repeated risks/tradeoffs
  - verbose rejected options
- Add max length validation for high-volume text fields.
- Prefer compact arrays of evidence IDs and mechanism labels.
- Keep the existing fields needed for validation:
  - decision type
  - planning mode
  - hypothesis
  - evidence used
  - changed variables
  - proposed experiments
  - proposal mechanisms
  - expected effects
  - rejected options
  - stop condition
- Keep the current `proposed_experiments` structure intact, including backend-validated model/template/config fields.
- Do not remove existing required fields unless validation and replay fixtures are intentionally updated in the same PR.

Acceptance:

- Average planner output tokens drop by at least 25%.
- Validation still rejects shallow or unsupported plans.
- Existing backend planner validation tests still pass without weakening validation.
- Mission Control remains readable.

## PR 9: Audit Storage De-Duplication

Owner: store/API.

This does not reduce LLM billing directly, but it reduces DB bloat and makes logs easier to inspect.

Implementation:

- Store full `input_messages` because that is the sent prompt audit.
- Store a compact `input_context` audit summary instead of duplicating the full prompt context when it is already embedded in `input_messages`.
- Options:
  - store `planner_context_snapshot_hash`
  - store `prompt_context_summary`
  - store section estimates
  - store full context only behind debug flag
- Add a debug flag:
  - `MODEL_EXPRESS_STORE_FULL_AGENT_CONTEXT=true|false`
  - default can remain true until confidence is built.

Acceptance:

- Agent invocation rows remain audit-useful.
- DB storage for large planner invocations drops materially when compact audit mode is enabled.
- No loss of validation traceability.

## PR 10: Mini Quality Replay Gate

Owner: orchestrator agents and tests.

The goal is not only cheaper calls; mini must still make good decisions.

Implementation:

- Build a replay harness that can run selected historical planner inputs through mini.
- Compare:
  - JSON parse success
  - validation success
  - duplicate-signature rejection rate
  - backend candidate ranking score
  - mechanism diversity
  - task alignment
  - whether selected model is valid for classification vs YOLO
- Replay against:
  - current V1 prompt/context
  - compressed static prompt
  - V2 compact context
  - distilled-memory-first context
- Store replay result summaries as deterministic eval artifacts.

Acceptance:

- Mini validation success rate remains >= 90%.
- V2 compressed prompts do not increase invalid planner outputs by more than 5 percentage points.
- YOLO and image-classification scenarios both pass replay smoke tests.

## PR 11: Mission Control LLM Cost Panel

Owner: frontend and API.

Make token use visible so regressions are obvious.

Implementation:

- Add an endpoint or reuse agent invocation data to summarize:
  - calls by agent
  - calls by model
  - exact input/output tokens when available
  - approximate tokens fallback
  - cached tokens
  - estimated cost
  - valid vs invalid calls
  - largest prompt sections
  - embedding calls by source indexing vs retrieval query
  - retrieval checks, retrieval hits, and injected memory count
- Add a Mission Control panel showing:
  - today / 7-day / project lifetime cost estimate
  - top token-heavy invocations
  - section-level prompt offenders
  - model split between `gpt-5.4`, `gpt-5.4-mini`, visual model, embeddings
  - embedding spend and whether retrieval hits were actually used

Acceptance:

- A user can tell whether $10 will last one day or one week from the UI.
- The panel identifies which agent and section is causing prompt growth.
- The panel identifies whether embedding spend is indexing useful memory, query-searching useful memory, or producing no-hit/log-only retrieval checks.

## Rollout Strategy

1. Switch default LLM model to `gpt-5.4-mini`.
2. Add exact usage capture, embedding spend audit, and section telemetry.
3. Temporarily cap paid embedding calls while collecting usefulness telemetry.
4. Run a few planner calls without compression to establish mini baseline.
5. Compress static prompt first.
6. Enable context V2 behind flag.
7. Add distilled memory first.
8. Enable output length guardrails without removing contract fields.
9. Enable compact audit storage.
10. Use replay and live runs to compare quality.
11. Make V2 default only after mini passes quality gates.

## Environment Flags

Proposed flags:

```env
MODEL_EXPRESS_LLM_MODEL=gpt-5.4-mini
MODEL_EXPRESS_VISUAL_LLM_MODEL=gpt-5.4-mini
MODEL_EXPRESS_LLM_API_STYLE=responses
MODEL_EXPRESS_LLM_STORED_RESPONSES=true
MODEL_EXPRESS_PLANNER_CONTEXT_VERSION=v2
MODEL_EXPRESS_PLANNER_STATIC_PROMPT_VERSION=compact_v1
MODEL_EXPRESS_PLANNER_MAX_SENT_TOKENS=35000
MODEL_EXPRESS_MEMORY_DISTILLED_FIRST=true
MODEL_EXPRESS_STORE_FULL_AGENT_CONTEXT=false
MODEL_EXPRESS_MEMORY_EMBEDDING_DAILY_BUDGET_USD=1.00
MODEL_EXPRESS_MEMORY_EMBEDDING_MAX_CALLS_PER_DAY=100
MODEL_EXPRESS_MEMORY_BACKFILL_LIMIT=50
MODEL_EXPRESS_MEMORY_RETRIEVAL_QUERY_CACHE_TTL_SECONDS=3600
MODEL_EXPRESS_MEMORY_LOG_ONLY_EMBEDDINGS=false
```

## SQL Tracking Queries

Largest actual sent planner prompts:

```sql
SELECT
  id,
  created_at,
  model,
  validation_status,
  CEIL(octet_length(input_messages::text) / 4.0)::int AS approx_sent_input_tokens,
  CEIL(octet_length(input_context::text) / 4.0)::int AS approx_audit_context_tokens,
  CEIL(octet_length(raw_output) / 4.0)::int AS approx_output_tokens
FROM agent_invocations
WHERE agent_name = 'experiment_planner'
ORDER BY approx_sent_input_tokens DESC
LIMIT 20;
```

Planner cost trend by day:

```sql
WITH sized AS (
  SELECT
    date_trunc('day', created_at) AS day,
    agent_name,
    model,
    validation_status,
    CEIL(octet_length(input_messages::text) / 4.0)::int AS approx_sent_input_tokens,
    CEIL(octet_length(raw_output) / 4.0)::int AS approx_output_tokens
  FROM agent_invocations
)
SELECT
  day,
  agent_name,
  model,
  COUNT(*) AS calls,
  ROUND(AVG(approx_sent_input_tokens)) AS avg_input_tokens,
  MAX(approx_sent_input_tokens) AS max_input_tokens,
  ROUND(AVG(approx_output_tokens)) AS avg_output_tokens,
  COUNT(*) FILTER (WHERE validation_status = 'valid') AS valid_calls,
  COUNT(*) FILTER (WHERE validation_status <> 'valid') AS non_valid_calls
FROM sized
GROUP BY day, agent_name, model
ORDER BY day DESC, avg_input_tokens DESC;
```

Current embedding inventory:

```sql
SELECT
  source_table,
  kind,
  embedding_model,
  COUNT(*) AS embedded_cards
FROM agent_memory_embeddings
GROUP BY source_table, kind, embedding_model
ORDER BY embedded_cards DESC;
```

Current retrieval-hit usefulness:

```sql
SELECT
  payload->>'purpose' AS purpose,
  COUNT(*) AS retrieval_checks,
  SUM(COALESCE((payload->>'retrieved_count')::int, 0)) AS total_retrieved_cards,
  COUNT(*) FILTER (WHERE COALESCE((payload->>'retrieved_count')::int, 0) > 0) AS checks_with_hits,
  ROUND(AVG(COALESCE((payload->>'retrieved_count')::int, 0)), 2) AS avg_cards_per_check
FROM execution_events
WHERE event_type = 'MEMORY_RETRIEVAL_LOGGED'
GROUP BY payload->>'purpose'
ORDER BY retrieval_checks DESC;
```

## Subagent Execution Prompt

Use this prompt to execute the plan with multiple subagents:

```text
You are Codex working in the Model Express repo. Implement docs/plans/llm_efficiency_plan.md end to end using multiple subagents. Use gpt-5.4-mini by default unless a task truly needs a stronger model.

Critical contract rule:
Do not change the Experiment Planner's required JSON output contract in a way that causes backend validation errors. LLM efficiency changes may compress prompt/context text, but must not remove, rename, or omit the Experiment Planner JSON output contract. Preserve the current ExperimentPlanningRecommendation shape, especially summary, rationale, decision_type, confidence, candidate_hypotheses with complete experiment_config, proposed_experiments, proposal_mechanisms, planning_mode, deterministic_diagnosis_used, evidence_used, hypothesis, expected_failure_modes, dataset_preprocessing_rationale, success_criteria, stop_condition, deployment_tradeoff, changed_variables, rejected_options, why_can_beat_champion, and mechanism metadata. Backend ranking may populate final candidate_rankings, proposed_experiments, and proposal_mechanisms, but the planner must still emit schema-valid ADD_EXPERIMENTS candidates that pass existing validation. Output optimization may shorten field values and cap repeated prose, but must not remove required fields or weaken backend validation.

Execution approach:
1. Start by reading docs/plans/llm_efficiency_plan.md, docs/me_ground_truth.md, and the relevant planner, memory, embedding, API, store, and Mission Control files.
2. Spawn subagents with disjoint ownership:
   - Subagent A: exact LLM usage capture in the LLM client, trace structs, invocation persistence, and tests.
   - Subagent B: embedding spend audit/gating/cache in memory/embeddings/API/store, including source-index vs retrieval-query usefulness telemetry.
   - Subagent C: planner section-level prompt telemetry, static prompt compression, and compact planner context V2 behind flags.
   - Subagent D: distilled-memory-first retrieval/context shaping and model catalog compression.
   - Subagent E: Mission Control cost/prompt/embedding telemetry panel and types.
   - Subagent F: replay/verification harness proving mini keeps schema validity and YOLO/classification planning quality.
3. Tell every subagent they are not alone in the codebase, must not revert unrelated edits, and must keep write scopes disjoint.
4. Implement in small, reviewable slices behind rollout flags. Keep V1 behavior available until replay checks pass.
5. Add tests for every slice. Do not weaken existing planner validation tests.
6. Run focused tests after each slice, then full orchestrator and frontend builds/tests before final handoff.
7. Update docs/me_ground_truth.md with the final repo status.

Success criteria:
- gpt-5.4-mini can be used as the default planner model.
- Exact LLM usage is stored when provider usage is available.
- Embedding usage is accountable by source-index vs retrieval-query purpose, with cache/gates to prevent no-hit/log-only waste.
- Planner section estimates identify top prompt offenders.
- Planner average sent input target is <= 35k tokens and high-end target <= 45k tokens on comparable project state.
- Existing planner output validation remains intact; no new backend validation errors are introduced.
- Distilled/high-value memory is preferred over bulky raw memory.
- Mission Control can show LLM token/cost and embedding usefulness/cost.
```


## Final Acceptance Criteria

The efficiency project is done when:

- `gpt-5.4-mini` is the default planner model.
- Exact usage is recorded when provider usage is available.
- Planner section-level prompt telemetry is visible.
- Embedding calls are visible by source-index vs retrieval-query purpose.
- Retrieval-query embedding usefulness is visible through retrieval hit and injected-memory telemetry.
- Average planner sent input is <= 35k tokens.
- High-end planner sent input is <= 45k tokens.
- Planner valid-output rate remains >= 90%.
- YOLO object-detection and image-classification planner scenarios both pass replay checks.
- Distilled memory is preferred over raw memory for planner context.
- Mission Control can show daily and weekly LLM cost estimates.
- $10 weekly-budget mode is realistic under normal active usage, measured through token and embedding spend telemetry.

## Non-Goals

- Do not remove memory entirely.
- Do not weaken backend validation to make mini look better.
- Do not remove required planner output fields to save output tokens.
- Do not let LLMs schedule jobs directly.
- Do not rely only on prompt wording for safety.
- Do not optimize visual dataset analysis before planner prompt bloat is under control.
