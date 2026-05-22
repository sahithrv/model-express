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

Planner input should stay compact and evidence-rich. Current context includes project goal, dataset profile, source plan, dataset planning insights, objective context, deterministic diagnosis, supported model catalog, current champion, baseline champion, run deltas, no-improvement rounds, stop signals, strategy memory, scorecards, plan jobs/summaries/evaluations/metrics, prior plans/jobs/evaluations, prior memory, existing experiment signatures, and follow-up limits.

Deterministic diagnosis is computed before the LLM call in `diagnosis.go`. It exposes scores for overfitting, underfitting, plateau, instability, class imbalance, minority-class failure, cost efficiency, latency penalty, and improvement stagnation, plus recommended failure modes and concise evidence strings. Planner outputs should cite these signals rather than inventing diagnosis.

Supported planning modes are `explore`, `exploit`, `champion_challenge`, `preprocessing_ablation`, `class_imbalance_ablation`, and `stop_or_select`. Mode-specific backend rules reject shallow or mismatched plans, such as explore batches with too few model families or class-imbalance ablations that do not target balancing/per-class metrics.

Candidate hypotheses are the preferred search shape. The planner can return 6-12 candidate hypotheses with evidence, proposed changes, expected metric impact, tradeoffs, risk, cost, novelty, memory links, and complete experiment configs. If `proposed_experiments` is empty, backend ranking selects 1-5 experiments.

Candidate ranking scores expected gain, novelty, cost, risk, deployment fit, redundancy, diagnosis alignment, and memory similarity. Rankings include `score_components`, selected/rejected flags, rejection reasons, and experiment signatures. Duplicates, tiny-only changes, weak high-cost candidates, poor objective fit, and diagnosis-mismatched ideas should be rejected or penalized. Preprocessing, resolution, sampling, crop/bbox, normalization, and augmentation-policy changes count as meaningful mechanisms when they are evidence-backed.

Rejected options are first-class planner output. Each item should include `option`, `reason`, `evidence`, and `applies_when`. Prior rejected options become `rejected_strategy_memory` so future planner calls avoid known-bad patterns.

Strategy scorecards are structured outcome memory separate from raw agent memory. They track strategy type, planning mode, dataset traits, objective profile, proposed changes, expected delta, actual delta, confidence before/after, cost, runtime, outcome, lesson, and tags. Future planner prompts should prefer `improved_champion` patterns and avoid similar failed/no-improvement strategies.

## Recent Work And Known Gaps

Recent work added richer dataset profiles, preprocessing-aware planned experiment fields, backend validation, worker-visible preprocessing config, deterministic diagnosis, planning modes, candidate ranking, rejected options, strategy scorecards, and Mission Control visibility for agent reasoning.

Preprocessing prompt update: backend and worker contracts now include `resolution_strategy`, `preprocessing`, `augmentation_policy`, `sampling_strategy`, class balancing, optimizer, scheduler, and fine-tuning fields. The planner prompt/examples now explicitly show the supported preprocessing contract and JSON shape.

Visual exemplars: the current slice supports optional evidence-only `visual_exemplar_context` in planner input. Backend caps exemplars from `datasets.profile.visual_exemplars` and passes class summary metadata, caps, budgets, warnings, and audit details, not executable authority. The LLM must still return JSON only and backend validation stays unchanged.

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
- Keep prompt context compact by capping memories, metrics, exemplars, and scorecards.
- Add UI support for candidate `score_components`, rejected options, strategy scorecards, and decision/outcome links.
- Add normalized validation/rejection events so failed LLM outputs are visible even without durable decisions.
- Preserve the safety contract: LLM JSON proposal first, backend validation and execution always.
