# LLM Decision Intelligence Agent Context

## Mission And Scope

The LLM Decision Intelligence Agent improves Model Express agent reasoning for image-classification experimentation. Its scope is planner and monitor prompt/schema quality, decision explainability, outcome learning, and compact context design for future LLM calls.

The agent should optimize how LLMs decide, explain, remember, and compare experiment strategies. It should not take over backend validation, worker execution, dataset mutation, or Mission Control UI ownership.

## Core Safety Principle

LLMs propose structured JSON only. The backend parses the JSON into typed Go structs, validates schema and policy, rejects unsupported or duplicate work, stores durable traces/decisions, and schedules follow-up plans only through deterministic guardrails.

Even in autonomous mode, the LLM must not directly create jobs, start workers, mutate datasets, choose arbitrary files, bypass model catalogs, or execute shell/backend actions.

## Current Agent Roles

`training_monitor` evaluates one completed or failed training job. It receives the plan, job config, run summary, rich evaluation when available, epoch metrics, objective context, and related memory. It outputs a JSON recommendation with quality summary, training dynamics, cost/risk interpretation, findings, tags, rank score, and an action such as `RANK_MODELS`, `PRUNE_RUN`, `STOP_PROJECT`, or `CHANGE_PREPROCESSING`. It does not propose experiment batches. Valid output becomes `training_evaluation` memory.

`experiment_planner` reviews a completed plan as a batch. It can recommend `ADD_EXPERIMENTS`, `SELECT_CHAMPION`, `STOP_PROJECT`, or `WAIT`. For `ADD_EXPERIMENTS`, it proposes complete experiment configs or candidate hypotheses that backend ranking can convert into final experiments. Valid output is stored as an agent invocation, usually an agent decision, and `planning_feedback` memory. Follow-up plan outcomes later become `planning_outcome` memory and strategy scorecards.

## Planner Intelligence Surface

Planner prompt context is now a compact `planner_context_snapshot`, not a raw dump of every project table. The backend still loads full Postgres history for validation and ranking, but the LLM sees a distilled decision brief: project card, dataset card, source-plan card, objective context, champion card, completed experiment ledger, failure diagnosis, search coverage, strategy lessons, blocked repeats, visual evidence, model catalog, retry validation feedback, stop/continue pressure, and prompt-budget metadata.

The snapshot deliberately omits raw `dataset.profile`, raw plan/job/evaluation lists, full epoch metric history, and unbounded memory payloads. Those fields are either summarized into cards or kept backend-only. This keeps prompt size stable as a project accumulates follow-up rounds while preserving the evidence needed for senior-data-scientist-style decisions.

Deterministic diagnosis is computed before the LLM call in `diagnosis.go`. It exposes scores for overfitting, underfitting, plateau, instability, class imbalance, minority-class failure, cost efficiency, latency penalty, and improvement stagnation, plus recommended failure modes and concise evidence strings. Planner outputs should cite these signals rather than inventing diagnosis.

Backend stop criteria can override an LLM `ADD_EXPERIMENTS` recommendation and persist `SELECT_CHAMPION`. This happens after repeated no-improvement follow-up rounds and now also when the current champion is within the minimum meaningful improvement threshold of a bounded metric ceiling, such as accuracy or macro-F1 near 1.0. The LLM should treat near-ceiling performance as a strong stop/select signal, not as an invitation to spend credits chasing impossible gains.

Supported planning modes are `explore`, `exploit`, `champion_challenge`, `preprocessing_ablation`, `class_imbalance_ablation`, and `stop_or_select`. Mode-specific backend rules reject shallow or mismatched plans, such as explore batches with too few model families or class-imbalance ablations that do not target balancing/per-class metrics.

Candidate hypotheses are the preferred search shape. The planner can return 6-12 candidate hypotheses with evidence, proposed changes, expected metric impact, tradeoffs, risk, cost, novelty, memory links, and complete experiment configs. If `proposed_experiments` is empty, backend ranking selects 1-5 experiments.

Candidate ranking scores expected gain, novelty, cost, risk, deployment fit, redundancy, diagnosis alignment, and memory similarity. Rankings include `score_components`, selected/rejected flags, rejection reasons, and experiment signatures. Duplicates, tiny-only changes, weak high-cost candidates, poor objective fit, and diagnosis-mismatched ideas should be rejected or penalized. Preprocessing, resolution, sampling, crop/bbox, normalization, and augmentation-policy changes count as meaningful mechanisms when they are evidence-backed.

If a planner output passes JSON/schema validation but fails backend proposal validation, the backend may retry the planner once with `planner_validation_feedback` in context. That feedback includes the deterministic rejection reason, rejected models/experiments, and instructions to return corrected JSON only. The corrected response is still accepted only through backend validation.

Rejected options are first-class planner output. Each item should include `option`, `reason`, `evidence`, and `applies_when`. Prior rejected options become `blocked_repeats` inside the compact snapshot so future planner calls avoid known-bad patterns.

Strategy scorecards are structured outcome memory separate from raw agent memory. They track strategy type, planning mode, dataset traits, objective profile, proposed changes, expected delta, actual delta, confidence before/after, cost, runtime, outcome, lesson, and tags. Future planner prompts receive these as compact `strategy_lessons`; they should prefer `improved_champion` lessons and avoid similar failed/no-improvement lessons.

The snapshot is reasoning context only; backend schema validation, model catalog checks, duplicate detection, same-mechanism filtering, final plan creation, and scheduling gates remain unchanged.

## Recent Work And Known Gaps

Recent work added richer dataset profiles, preprocessing-aware planned experiment fields, backend validation, worker-visible preprocessing config, deterministic diagnosis, planning modes, candidate ranking, rejected options, strategy scorecards, and Mission Control visibility for agent reasoning.

Preprocessing prompt update: backend and worker contracts now include `resolution_strategy`, `preprocessing`, `augmentation_policy`, `sampling_strategy`, class balancing, optimizer, scheduler, and fine-tuning fields. The planner prompt/examples now explicitly show the supported preprocessing contract and JSON shape.

Planner context compaction: recent planning-context work replaces raw LLM prompt dumps with `planner_context_snapshot_v1`. The goal is smaller planner calls and cleaner review/audit surfaces while retaining the distilled evidence used to choose or reject planner actions.

Visual exemplars: the current slice supports optional evidence-only visual evidence inside `planner_context_snapshot.visual_evidence`. Backend caps exemplars from `datasets.profile.visual_exemplars` and passes class summary metadata, caps, budgets, warnings, and audit details, not executable authority. The LLM must still return JSON only and backend validation stays unchanged.

Semantic memory: not implemented. Current retrieval is mostly project/dataset scoped. Future retrieval should find similar datasets, objectives, imbalance profiles, model families, and strategy outcomes without bloating prompts.

Score components UI: candidate rankings now expose `score_components`, and Mission Control renders score components/rejection status defensively. Deeper links among decisions, invocations, follow-up plans, and outcomes remain future work.

## Key Files To Inspect

- `services/orchestrator/internal/agents/experiment_planner_llm.go`
- `services/orchestrator/internal/agents/experiment_planner_llm_test.go`
- `services/orchestrator/internal/agents/training_monitor_llm.go`
- `services/orchestrator/internal/agents/training_monitor_llm_test.go`
- `services/orchestrator/internal/agents/diagnosis.go`
- `services/orchestrator/internal/agents/candidate_ranking.go`
- `services/orchestrator/internal/agents/objective.go`
- `services/orchestrator/internal/agents/reviewer.go`
- `services/orchestrator/internal/plans/model.go`
- `services/orchestrator/internal/decisions/model.go`
- `services/orchestrator/internal/memory/model.go`
- `services/orchestrator/internal/strategies/model.go`
- `services/orchestrator/internal/api/handlers.go`
- `services/orchestrator/internal/api/handlers_test.go`
- `services/orchestrator/internal/datasets/model.go`
- `services/worker/worker/datasets/profiler.py`
- `services/worker/worker/training/local.py`
- `services/worker/worker/training/modal_app.py`
- `apps/mission-control/src/App.tsx`
- `apps/mission-control/src/types.ts`

Key docs:

- `docs/llm_agent_decision_context.md`
- `docs/llm_decision_quality_report.md`
- `docs/data_preprocessing_agent_report.md`
- `docs/model_express_agentic_upgrade_roadmap.md`
- `docs/frontend_mission_control_report.md`
- `docs/orchestrator_system_design_report.md`
- `docs/agentic.md`

Useful tests:

- From `services/orchestrator`: `go test ./internal/agents`
- Broader backend check: `go test ./...`
- Frontend build only if UI files change: `npm run build`

## Safe Boundaries With Other Agents

Backend/data agents own schema enforcement, dataset profiling, preprocessing validation, worker config, execution scheduling, job lifecycle, idempotency, and storage. LLM Decision Intelligence may propose prompt/schema/context improvements, but should not loosen backend validation or create unsupported execution paths.

Data/preprocessing agents own visual exemplar generation, bbox/crop execution, dataset normalization, split-file handling, annotation parsing, and advanced augmentation support. LLM planner changes may consume those outputs once validated and budgeted.

Frontend agents own Mission Control layout, typed display fields, component tests, and UX for rankings/scorecards/rejections. LLM payload fields should remain backend-compatible and defensively displayable.

Worker agents own actual transforms, training loops, artifact export, Modal/local behavior, and inference runtime. LLM plans must use supported catalog/config values only.

## Future Task Checklist

- Keep planner prompt/examples aligned with `resolution_strategy`, `preprocessing`, `augmentation_policy`, and `sampling_strategy` allowed values.
- Keep planner tests covering preprocessing-aware outputs, diagnosis evidence, planning modes, rejected options, candidate ranking behavior, visual exemplar evidence-only context, caps/audit details, and JSON-only/backend-validation wording.
- Add durable invocation audit fields for whether exemplars were used.
- Add semantic retrieval for similar dataset/objective/strategy memory.
- Keep prompt context compact by capping memories, metrics, exemplars, and scorecards, and by using compact planner context snapshots instead of raw prompt dumps.
- Add UI support for candidate `score_components`, rejected options, strategy scorecards, and decision/outcome links.
- Add normalized validation/rejection events so failed LLM outputs are visible even without durable decisions.
- Preserve the safety contract: LLM JSON proposal first, backend validation and execution always.
