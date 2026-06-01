# Vector Retrieval Plan

Status: implemented in this branch behind rollout flags.

Implementation note: PR 1 through PR 9 have been wired into the orchestrator, store, planner, training monitor, candidate ranking, replay evals, and Mission Control. Retrieval remains opt-in with `MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED`; prompt influence stays disabled by default through `MODEL_EXPRESS_MEMORY_RETRIEVAL_LOG_ONLY=true`, and cross-project retrieval stays disabled unless `MODEL_EXPRESS_MEMORY_CROSS_PROJECT_ENABLED=true`.

Filename note: this file intentionally keeps the requested spelling, `vector_retirval_plan.md`.

## Goal

Add semantic/vector retrieval to Model Express so the LLM agents can retrieve high-value prior lessons from similar runs, datasets, objectives, mechanisms, and outcomes without stuffing raw history into prompts.

The target behavior is:

- The Planner sees compact lessons from similar successful and failed strategies, not only the most recent same-dataset records.
- Candidate ranking can score memory similarity deterministically, even when the LLM did not cite the right memory ID.
- The Training Monitor can compare a run against similar prior dynamics, model families, and objective tradeoffs.
- Dataset and visual-analysis evidence can retrieve preprocessing lessons from similar dataset fingerprints.
- Prompt context remains capped, source-linked, and backend-validated.

This is not a general-purpose chat memory system. It is a decision-memory system for image-classification experiment planning.

## Current Baseline

The project already has the right foundation:

- `agent_memory_records` store distilled memories.
- `agent_invocations` store full audit traces.
- `strategy_scorecards` store mechanism-level outcome learning.
- Planner context is compacted into `planner_context_snapshot`.
- Candidate ranking already has a `memory_similarity` score component.
- Rejected options and blocked repeats already exist.

The main current limitation is retrieval. Memory lookup is mostly project/dataset scoped and sorted by recency. The system remembers what happened recently, not necessarily what is most relevant.

Important current files:

- `services/orchestrator/internal/memory/model.go`
- `services/orchestrator/internal/store/store.go`
- `services/orchestrator/internal/store/memory.go`
- `services/orchestrator/internal/store/postgres.go`
- `services/orchestrator/internal/store/migrations/001_init.sql`
- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/candidate_ranking.go`
- `services/orchestrator/internal/agents/training_monitor_llm.go`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/strategies/model.go`

## Product Principle

Vector retrieval should retrieve small memory cards, not raw records.

Do embed:

- Strategy outcome lessons.
- Rejected options.
- Run-dynamics summaries.
- Dataset fingerprints.
- Visual/preprocessing hypotheses.
- Champion tradeoff lessons.

Do not embed as prompt-facing memory:

- Full prompts.
- Raw LLM outputs.
- Full invocation contexts.
- Full epoch arrays.
- Full manifests or image-level records.
- Unbounded JSON payloads.

Full source records should remain available through IDs for audit, replay, and future fine-tuning, but retrieval should return compact decision cards.

## High-ROI Memory Types

### Strategy Outcome Memory

Source:

- `strategy_scorecards`
- `planning_outcome` memory records

Fields to capture:

- mechanism
- intervention
- diagnosis triggers
- evidence used
- dataset traits
- objective profile
- proposed changes
- expected delta
- actual delta
- outcome
- cost and runtime
- lesson
- source decision, source plan, follow-up plan

Why it matters:

This is the highest-value memory because it answers: "When the system tried this kind of strategy in this kind of situation, did it actually improve the champion?"

### Rejected Option Memory

Source:

- planner `rejected_options`
- backend validation rejection memory
- blocked repeats
- project trajectory blocked mechanisms

Fields to capture:

- option
- reason
- evidence
- applies_when
- mechanism
- related failure modes
- rejected experiment shape

Why it matters:

This lets the system skip shallow repeats, invalid proposals, and already-exhausted mechanisms.

### Run-Dynamics Memory

Source:

- `training_evaluation`
- `training_run_summaries`
- `training_run_evaluations`
- epoch metric summaries

Fields to capture:

- model
- model family
- mechanism inferred from the run config
- training dynamics
- overfit, underfit, plateau, instability
- per-class weakness
- objective fit
- rank score
- cost and runtime
- tags

Why it matters:

This gives the Training Monitor and Planner better local analogies, such as "this model family repeatedly plateaued on imbalanced fine-grained datasets."

### Dataset Fingerprint Memory

Source:

- dataset profile
- normalized metadata safe summary
- visual-analysis record
- visual exemplar context

Fields to capture:

- task type
- class count bucket
- image count bucket
- imbalance ratio bucket
- dimension stats bucket
- dataset traits
- metadata capabilities such as bbox/attributes/keypoints
- visual traits
- classes to watch
- preprocessing hypotheses

Why it matters:

This enables cross-project retrieval. A new dataset can benefit from prior lessons on similar dataset shapes before burning runs to rediscover the same strategy.

### Champion Tradeoff Memory

Source:

- `project_champions`
- champion selection decisions
- deployment profiles
- export/demo metadata

Fields to capture:

- selected model
- target metric
- objective profile
- quality, latency, cost, runtime
- selection reason
- what failed to beat it

Why it matters:

This improves stop/select/champion-challenge decisions.

## Retrieval Formula

Use hybrid retrieval. Pure vector similarity is not enough because this system has strong structured metadata.

Suggested final score:

```text
final_score =
  0.34 * semantic_similarity
+ 0.16 * dataset_similarity
+ 0.14 * objective_similarity
+ 0.14 * mechanism_similarity
+ 0.12 * outcome_utility
+ 0.06 * actionability
+ 0.04 * recency
- stale_schema_penalty
- validation_penalty
- redundancy_penalty
```

Outcome utility should be asymmetric:

```text
improved_champion: strong positive
minor_improvement: moderate positive if cost/runtime were acceptable
no_improvement: useful negative
failed: useful negative if failure cause matches the query
pending: low value
invalidated: exclude by default
```

Retrieval should always return:

- memory ID or scorecard ID
- source table and source ID
- reason for retrieval
- compact lesson
- outcome status
- mechanism/intervention
- relevant dataset/objective traits
- source links for audit

## Prompt Budget Rules

Planner:

- Default retrieved memory cards: 8 to 12 total.
- Successful strategy cards: max 3.
- Failed strategy cards: max 3.
- Rejected/blocked cards: max 4.
- Dataset/preprocessing analogy cards: max 2.

Training Monitor:

- Default retrieved memory cards: 5 to 8 total.
- Prefer same model family, same dataset, same dynamics pattern.

Candidate ranking:

- Can retrieve more backend-only evidence because it does not all enter prompt context.
- Keep ranking explanations compact and source-linked.

## Proposed Data Model

Add a separate embedding table instead of overloading `agent_memory_records`.

```text
agent_memory_embeddings
- id
- source_table
- source_id
- project_id
- dataset_id
- kind
- scope
- embedding_model
- embedding_dimensions
- embedding vector
- embedding_text
- summary_card jsonb
- metadata jsonb
- quality_score
- outcome_score
- created_at
- updated_at
```

Useful metadata keys:

```text
task_type
agent_name
memory_kind
mechanism
intervention
planning_mode
outcome
actual_delta
expected_delta
cost_usd
runtime_seconds
model
model_family
objective
dataset_traits
diagnosis_triggers
validation_status
source_plan_id
followup_plan_id
job_id
```

For Postgres, prefer pgvector because the rest of the control plane already lives in Postgres. The current local compose file uses `postgres:16`, so the migration PR should either switch the image to a pgvector-capable image or document the required extension setup.

## Subagents To Spin Up

### System Architecture Review Agent

Use for:

- Confirming the overall slice order.
- Reviewing data ownership, storage boundaries, and rollout risk.

Agent context:

- `docs/agents_context/system_architecture_review/context.md`

Helpful docs:

- `docs/me_ground_truth.md`
- `docs/orchestrator_system_design_report.md`
- `docs/agentic.md`
- `docs/plans/model_express_agentic_upgrade_roadmap.md`

### Orchestrator Backend Agent

Use for:

- Memory contracts.
- Store interfaces.
- Postgres migrations.
- pgvector schema.
- Embedding indexing service.

Agent context:

- `docs/agents_context/orchestrator_backend/context.md`

Helpful docs:

- `docs/me_ground_truth.md`
- `docs/plans/llm_agent_decision_context.md`
- `docs/orchestrator_system_design_report.md`
- `docs/model_express_pr_completion_report.md`

### LLM Decision Intelligence Agent

Use for:

- Planner context integration.
- Training Monitor integration.
- Candidate ranking use of retrieved memories.
- Prompt budget and compact memory cards.

Agent context:

- `docs/agents_context/llm_decision_intelligence/context.md`

Helpful docs:

- `docs/llm_decision_quality_report.md`
- `docs/plans/llm_agent_decision_context.md`
- `docs/model_express_pr_completion_report.md`
- `docs/me_ground_truth.md`

### Data Dataset Intelligence Agent

Use for:

- Dataset fingerprint cards.
- Visual-analysis and metadata memory cards.
- Preprocessing-hypothesis retrieval.

Agent context:

- `docs/agents_context/data_dataset_intelligence/context.md`

Helpful docs:

- `docs/data_preprocessing_agent_report.md`
- `docs/plans/datatset_llm_plan.md`
- `docs/me_ground_truth.md`
- `docs/agents_context/python_worker_training/context.md`

### Frontend Mission Control Agent

Use for:

- Retrieved-memory UI.
- Ranking explanation UI.
- Memory source links.
- Defensive rendering of new payload fields.

Agent context:

- `docs/agents_context/frontend_mission_control/context.md`

Helpful docs:

- `docs/frontend_mission_control_report.md`
- `docs/me_ground_truth.md`
- `docs/llm_decision_quality_report.md`

### Integration Coordinator Agent

Use for:

- Cross-PR sequencing.
- E2E acceptance.
- Replay/eval scenarios.
- Rollout flags and operational checks.

Agent context:

- `docs/agents_context/integration_coordinator/context.md`

Helpful docs:

- `docs/model_express_end_to_end_checklist.md`
- `docs/me_ground_truth.md`
- `docs/orchestrator_system_design_report.md`

## PR Split

The PRs below are designed to minimize file overlap. Some PRs can run in parallel after PR 1 establishes stable contracts.

## PR 1: Vector Memory Contracts And Card Builders

Owner subagent:

- Orchestrator Backend Agent
- LLM Decision Intelligence Agent for review only

Goal:

Define the shared memory retrieval contracts and canonical compact card builders without adding pgvector or changing Planner behavior.

Primary files:

- `services/orchestrator/internal/memory/model.go`
- New: `services/orchestrator/internal/memory/retrieval.go`
- New: `services/orchestrator/internal/memory/cards.go`
- New tests under `services/orchestrator/internal/memory`

Possible types:

```go
type EmbeddableMemoryCard struct {
    SourceTable string
    SourceID string
    ProjectID string
    DatasetID string
    Kind string
    Scope string
    Text string
    SummaryCard map[string]any
    Metadata map[string]any
    QualityScore float64
    OutcomeScore float64
}

type MemoryRetrievalQuery struct {
    ProjectID string
    DatasetID string
    AgentName string
    Purpose string
    Text string
    Kinds []string
    Mechanisms []string
    DatasetTraits []string
    Objective string
    Limit int
}

type MemoryRetrievalResult struct {
    SourceTable string
    SourceID string
    Kind string
    Score float64
    SemanticScore float64
    StructuredScore float64
    RetrievalReason string
    SummaryCard map[string]any
    Metadata map[string]any
}
```

Implementation notes:

- Add builders for `AgentMemoryRecord` and `StrategyScorecard`.
- Keep card text deterministic and compact.
- Include source IDs, mechanism, outcome, and lesson.
- Do not call embedding APIs.
- Do not add DB migrations.
- Do not touch Planner or Training Monitor call sites.

Tests:

- Strategy scorecard card includes mechanism/intervention/outcome/actual delta.
- Planning outcome card includes proposed models and lesson.
- Training evaluation card excludes raw payload dumps.
- Card text is deterministic for the same input.

Acceptance:

- A later PR can index cards without reinterpreting raw payloads.
- Existing tests pass.

Avoid touching:

- `postgres.go`
- planner prompt files
- frontend files

## PR 2: Pgvector Schema And Store Retrieval Interface

Owner subagent:

- Orchestrator Backend Agent

Goal:

Add durable storage for embeddings and a store interface for vector search. This PR should not call an embedding provider and should not change agent behavior.

Primary files:

- `infra/compose.yaml`
- `services/orchestrator/internal/store/store.go`
- `services/orchestrator/internal/store/postgres.go`
- `services/orchestrator/internal/store/memory.go`
- New migration: `services/orchestrator/internal/store/migrations/005_vector_memory.sql`
- Store tests

Schema:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS agent_memory_embeddings (
  id text PRIMARY KEY,
  source_table text NOT NULL,
  source_id text NOT NULL,
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  dataset_id text NOT NULL DEFAULT '',
  kind text NOT NULL,
  scope text NOT NULL DEFAULT '',
  embedding_model text NOT NULL,
  embedding_dimensions integer NOT NULL,
  embedding vector(1536) NOT NULL,
  embedding_text text NOT NULL,
  summary_card jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  quality_score double precision NOT NULL DEFAULT 0,
  outcome_score double precision NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (source_table, source_id, embedding_model)
);
```

The final dimensions should match the selected embedding model. If dimensions need to be configurable, include `embedding_dimensions` and use a documented default.

Store interface additions:

```go
UpsertMemoryEmbedding(record memory.MemoryEmbeddingRecord) (memory.MemoryEmbeddingRecord, error)
SearchMemoryEmbeddings(query memory.MemoryRetrievalQuery) ([]memory.MemoryRetrievalResult, error)
ListUnembeddedMemorySources(projectID string, limit int) ([]memory.EmbeddableMemoryCard, error)
```

Implementation notes:

- Postgres implementation performs vector distance plus structured filters.
- In-memory implementation can use deterministic lexical scoring as a test fallback.
- Add indexes for project, dataset, kind, source, metadata fields where useful.
- Consider partial indexes for `outcome` and `mechanism`.

Tests:

- Migration idempotency.
- Upsert is idempotent by source/model.
- Search respects project and kind filters.
- In-memory store returns stable ordering.

Acceptance:

- Vector storage is available behind store interfaces.
- No LLM agent behavior changes.

Avoid touching:

- `experiment_planner_llm.go`
- `training_monitor_llm.go`
- frontend files

## PR 3: Embedding Client, Indexer, And Backfill

Owner subagent:

- Orchestrator Backend Agent
- Integration Coordinator Agent for rollout review

Goal:

Create embeddings for compact memory cards and backfill existing useful memories. This PR should not change Planner prompts yet.

Primary files:

- New: `services/orchestrator/internal/embeddings/model.go`
- New: `services/orchestrator/internal/embeddings/client.go`
- New: `services/orchestrator/internal/memory/indexer.go`
- `services/orchestrator/internal/llm/model.go`
- `services/orchestrator/internal/api/handlers.go` only if adding an admin/backfill endpoint
- Tests for embedding client and indexer

Environment:

```text
MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED=false
MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED=false
MODEL_EXPRESS_MEMORY_EMBEDDING_PROVIDER=openai
MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL=<embedding-model>
MODEL_EXPRESS_MEMORY_EMBEDDING_BASE_URL=<optional>
MODEL_EXPRESS_MEMORY_EMBEDDING_API_KEY=<optional>
MODEL_EXPRESS_MEMORY_EMBEDDING_DIMENSIONS=1536
MODEL_EXPRESS_MEMORY_BACKFILL_LIMIT=500
```

Implementation notes:

- Keep embedding config separate from chat/Responses LLM config.
- Support `openai`, `openai_compatible`, and `local` styles if practical.
- If disabled or missing config, indexing should no-op safely.
- Index these sources first:
  - strategy scorecards
  - planning outcome memory
  - planning feedback rejected options
  - training evaluation memory
- Store deterministic `embedding_text` and `summary_card`.
- Do not index invalidated/pending records unless explicitly allowed.

Tests:

- Fake embedding client returns stable vectors.
- Indexer skips unsupported/malformed memory.
- Backfill is idempotent.
- Missing embedding config does not break normal orchestrator flow.

Acceptance:

- Existing memories can be embedded.
- New memory writes can enqueue or trigger indexing safely.
- Retrieval is still not injected into Planner prompts in this PR.

Avoid touching:

- Planner prompt/schema tests except shared imports if unavoidable.
- Mission Control UI.

## PR 4: Planner Retrieval Context

Owner subagent:

- LLM Decision Intelligence Agent

Goal:

Inject retrieved compact memory cards into Planner input and `planner_context_snapshot` with strict caps.

Primary files:

- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/planner_information_tools.go`
- `services/orchestrator/internal/api/handlers.go`
- Planner tests

New Planner input field:

```go
RetrievedMemory []memory.MemoryRetrievalResult
```

Snapshot addition:

```json
"retrieved_memory": {
  "successful_strategy_cards": [],
  "failed_strategy_cards": [],
  "blocked_or_rejected_cards": [],
  "dataset_preprocessing_cards": [],
  "caps": {},
  "retrieval_enabled": true
}
```

Implementation notes:

- Build a retrieval query from:
  - dataset card
  - objective context
  - deterministic diagnosis
  - source-plan result summary
  - mechanism coverage
  - current champion card
- Keep existing recency memory path as fallback.
- Do not expose raw embedding text in prompt.
- Keep retrieved cards source-linked.
- Add prompt-budget telemetry for retrieved memory count and approximate bytes.

Tests:

- Planner snapshot includes retrieved cards when provided.
- Planner snapshot remains bounded.
- Raw memory payloads and embedding text are excluded.
- Retrieval disabled path matches current behavior.

Acceptance:

- Planner receives semantically relevant compact lessons.
- No scheduling or validation rules are loosened.

Avoid touching:

- `candidate_ranking.go` beyond type compilation if possible.
- Training Monitor files.
- Postgres migration files.

## PR 5: Backend Candidate Ranking Uses Retrieved Memory

Owner subagent:

- LLM Decision Intelligence Agent

Goal:

Use retrieved memory in deterministic candidate ranking so memory similarity works even when the LLM does not cite memory IDs.

Primary files:

- `services/orchestrator/internal/agents/candidate_ranking.go`
- `services/orchestrator/internal/agents/experiment_planner_llm.go` only for shared types if needed
- Candidate ranking tests

Implementation notes:

- Extend `candidateMemoryScore`.
- Match retrieved cards against candidate mechanism, intervention, model family, proposed changes, diagnosis triggers, and objective.
- Positive memories should add a bounded bonus.
- Failed/no-improvement memories should apply a bounded penalty.
- Rejected/blocked memories can reject a candidate if they match strongly.
- Record `score_components["retrieved_memory"]`.
- Include reason strings such as:
  - `similar retrieved successful strategy`
  - `similar retrieved failed mechanism`
  - `blocked by retrieved rejected option`

Tests:

- Similar improved scorecard increases ranking.
- Similar failed scorecard decreases ranking.
- Similar rejected option can block a candidate.
- Unrelated high-similarity text does not override structured mismatch.
- Ranking remains deterministic.

Acceptance:

- Candidate rankings become outcome-aware.
- Backend ranking can use more retrieved evidence than the prompt includes.

Avoid touching:

- Store layer.
- Training Monitor.
- Frontend.

## PR 6: Training Monitor Retrieval

Owner subagent:

- LLM Decision Intelligence Agent

Goal:

Give the Training Monitor relevant prior run-dynamics memory rather than only recent same-dataset memory.

Primary files:

- `services/orchestrator/internal/agents/training_monitor_llm.go`
- `services/orchestrator/internal/agents/training_monitor_information_tools.go`
- `services/orchestrator/internal/api/handlers.go`
- Training Monitor tests

Retrieval query should include:

- model
- model family
- current run metrics
- train/val gap
- plateau/instability signals
- per-class weakness
- objective context
- dataset traits

Prompt context addition:

```json
"retrieved_run_memory": [
  {
    "source_id": "",
    "kind": "training_evaluation",
    "summary": "",
    "model_family": "",
    "dynamics": "",
    "retrieval_reason": ""
  }
]
```

Implementation notes:

- Keep `trainingMonitorMaxMemoryRecords` as the hard cap.
- Prefer same project/dataset, but allow cross-project only when retrieval is explicitly enabled.
- Do not include raw payloads.
- Fall back to existing `compactMemoryRecords`.

Tests:

- Retrieved run memory appears in compact context.
- Raw epoch history remains excluded.
- Disabled retrieval preserves current behavior.

Acceptance:

- Training Monitor can compare a run against similar previous dynamics.
- Prompt size remains stable.

Avoid touching:

- Planner files.
- Candidate ranking.
- Migrations.

## PR 7: Dataset And Preprocessing Retrieval Cards

Owner subagent:

- Data Dataset Intelligence Agent

Goal:

Create and index dataset/preprocessing memory cards so new datasets can retrieve lessons from similar dataset fingerprints.

Primary files:

- `services/orchestrator/internal/datasets/model.go`
- `services/orchestrator/internal/datasets/metadata/service.go`
- `services/orchestrator/internal/api/handlers.go` for source collection if needed
- `services/orchestrator/internal/memory/cards.go`
- Dataset/visual-analysis tests

Card sources:

- dataset profile
- active metadata safe summary
- accepted visual analysis
- preprocessing hypotheses
- label-quality audit summaries, if already stored in profile metadata

Implementation notes:

- Create dataset fingerprint text from stable buckets:
  - class count bucket
  - image count bucket
  - imbalance bucket
  - dimension pattern
  - metadata capability flags
  - visual traits
  - preprocessing hypotheses
- Do not embed image URIs or raw examples.
- Keep visual evidence advisory only.
- Retrieval should inform Planner evidence; backend validation still decides whether preprocessing is executable.

Tests:

- Dataset card builder excludes raw URIs and manifest rows.
- Bbox capability appears as structured metadata.
- Similar dataset card can be indexed without an LLM invocation.

Acceptance:

- Planner can retrieve preprocessing/dataset analogies.
- No worker execution behavior changes.

Avoid touching:

- Python worker training code unless a missing source field must be produced.
- Candidate ranking.
- Frontend.

## PR 8: Mission Control Retrieval Visibility

Owner subagent:

- Frontend Mission Control Agent

Goal:

Expose retrieved memory and ranking memory reasons in Mission Control without making the UI noisy.

Primary files:

- `apps/mission-control/src/types.ts`
- `apps/mission-control/src/App.tsx`
- `apps/mission-control/src/styles.css`
- API response typing as needed

UI concepts:

- Retrieved memory section on agent decision detail.
- Candidate ranking reasons show retrieved-memory hits.
- Source links to memory records, scorecards, plans, jobs, or decisions when available.
- Badge outcome:
  - improved
  - minor
  - no improvement
  - failed
  - rejected

Implementation notes:

- Render defensively when retrieval is disabled or fields are absent.
- Do not add a large raw JSON dump.
- Keep display compact: source, outcome, mechanism, lesson, retrieval reason.

Tests/build:

- `npm run build`
- Add component-level defensive rendering tests if the app has an existing pattern.

Acceptance:

- Operators can see why memory influenced a recommendation.
- Missing retrieval fields do not break project detail views.

Avoid touching:

- Orchestrator store.
- Planner algorithms.

## PR 9: Retrieval Evals, Replay, And Rollout Guardrails

Owner subagent:

- Integration Coordinator Agent
- System Architecture Review Agent for final review

Goal:

Verify retrieval improves decision quality without increasing prompt size or causing stale bad memories to dominate.

Primary files:

- `services/orchestrator/internal/agents/evals/replay.go`
- `services/orchestrator/internal/agents/evals/replay_test.go`
- `services/orchestrator/internal/agents/evals/scorers.go`
- `docs/llm_decision_quality_report.md`
- `docs/me_ground_truth.md`
- This plan file if final status updates are needed

Eval scenarios:

- Similar successful strategy should be retrieved and cited.
- Similar failed mechanism should reduce ranking.
- Rejected option should block shallow repeat.
- Cross-project similar dataset should help preprocessing proposal.
- Unrelated high-similarity memory should not beat structured filters.
- Retrieval disabled should preserve baseline behavior.

Metrics:

- retrieved card count
- prompt byte increase
- retrieval hit source mix
- selected candidate memory score
- rejected candidate memory penalty
- no-improvement repeat rate
- invalid proposal rate

Rollout flags:

```text
MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED=false
MODEL_EXPRESS_MEMORY_CROSS_PROJECT_ENABLED=false
MODEL_EXPRESS_MEMORY_RETRIEVAL_MAX_CARDS=10
MODEL_EXPRESS_MEMORY_RETRIEVAL_MIN_SCORE=0.55
MODEL_EXPRESS_MEMORY_RETRIEVAL_LOG_ONLY=true
```

Acceptance:

- Retrieval can run in log-only mode.
- Retrieval can be disabled without schema rollback.
- Replay tests prove that retrieval changes ranking for the right reasons.
- Docs describe operating and debugging retrieval.

Avoid touching:

- Core schema unless fixing a bug from prior PRs.
- Frontend layout beyond doc-linked fields.

## Parallelization Map

Wave 0:

- PR 1 starts first and defines stable contracts.

Wave 1 after PR 1:

- PR 2 can implement storage.
- PR 3 can implement embedding client/indexer against PR 1 contracts and merge after PR 2.
- PR 7 can implement dataset/preprocessing card builders.
- PR 8 can start UI mocks/types from the PR 1 result shape, then wire real fields later.

Wave 2 after PR 2 and PR 3:

- PR 4 adds Planner retrieval context.
- PR 6 adds Training Monitor retrieval.

Wave 3 after PR 4:

- PR 5 adds candidate ranking retrieved-memory scoring.

Wave 4 after PR 4, PR 5, PR 6:

- PR 8 completes UI wiring if it was started earlier.
- PR 9 adds evals, rollout, and final docs.

## Non-Overlap Rules

- Store/migration work belongs to PR 2.
- Embedding provider/indexing work belongs to PR 3.
- Planner prompt/context work belongs to PR 4.
- Candidate ranking changes belong to PR 5.
- Training Monitor prompt/context work belongs to PR 6.
- Dataset/visual memory source work belongs to PR 7.
- Mission Control display work belongs to PR 8.
- Replay/eval and rollout docs belong to PR 9.

If a PR needs a type from another area, prefer adding the type to PR 1 contracts rather than editing another agent's implementation file.

## Risks And Mitigations

Risk: memory bloat.

Mitigation: index compact cards only, cap retrieval, keep raw records source-linked but prompt-excluded.

Risk: stale or bad memories dominate.

Mitigation: outcome scores, validation filters, schema versioning, recency only as a small bonus.

Risk: vector similarity retrieves semantically similar but operationally invalid memories.

Mitigation: hybrid scoring with hard filters for task type, backend-valid mechanisms, objective, and outcome.

Risk: cross-project retrieval leaks irrelevant context.

Mitigation: keep cross-project retrieval behind a flag and require minimum structured similarity.

Risk: pgvector complicates local setup.

Mitigation: isolate storage in PR 2, keep in-memory lexical fallback for tests/dev, document compose changes.

Risk: prompt context grows again.

Mitigation: strict card caps and prompt-budget telemetry in Planner and Training Monitor.

## Definition Of Done

The final product is done when:

- Strategy scorecards, planning outcomes, rejected options, run evaluations, and dataset fingerprints can be indexed.
- Planner retrieves compact relevant memories by semantic and structured similarity.
- Training Monitor retrieves compact relevant run-dynamics memories.
- Candidate ranking uses retrieved memories deterministically.
- Mission Control can show memory sources and retrieval reasons.
- Retrieval can run in log-only mode, full mode, or disabled mode.
- Replay/eval tests prove retrieval helps avoid repeats and promote proven mechanisms.
- Prompt size remains bounded and auditable.
