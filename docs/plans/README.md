# Reliability And Agent Quality Roadmap

Status: Proposed
Date: 2026-07-09

## Purpose

This directory breaks four reliability and agent-quality problems into small, dependency-ordered pull requests:

1. [Experiment Execution Fidelity](./01-experiment-execution-fidelity.md)
2. [Candidate Ranking Correctness](./02-candidate-ranking-correctness.md)
3. [Progress Reporting And Activity Efficiency](./03-progress-reporting-and-activity-efficiency.md)
4. [Empirical Agent Quality Calibration](./04-agent-quality-calibration.md)

The plans deliberately establish trustworthy execution data before using historical outcomes to calibrate the agent. A planner cannot learn reliably from an experiment if the stored proposal does not prove what the worker actually ran.

## Final Outcome

When all four tracks are complete, ModelExpress should be able to answer these questions from durable, structured data:

1. What experiment did the planner request?
2. What configuration did the selected runner actually apply?
3. Why did the backend choose this candidate over the alternatives?
4. What stage is the project in right now, and how long has it been there?
5. How accurate have this planner version's predictions and decisions been historically?
6. Can a new prompt, scorer, or retrieval strategy be promoted without reducing quality?

## PR Sizing Rules

Each PR in these plans should follow these boundaries:

- Make one primary behavioral change.
- Prefer one subsystem; cross-language PRs are reserved for contract plumbing that cannot be safely split.
- Add or update tests in the same PR as the behavior.
- Keep database changes additive and backward compatible.
- Keep readers tolerant of the previous payload version during rollout.
- Avoid combining schema changes, a worker behavior change, and a major UI redesign in one PR.
- Include a rollback switch when a material policy change alters scheduling or production traffic. A localized correctness fix may instead use explicit before/after replay evidence and a normal code-revert plan.

Sizing labels used in the plans:

- **S**: localized behavior and tests, normally one subsystem.
- **M**: a narrow contract change crossing two subsystems.
- No PR is intentionally sized as **L**. If implementation grows beyond a focused review, split it at the contract boundary described in that PR.

## Cross-Track Delivery Order

The tracks can overlap, but their release order matters.

| Wave | Pull requests | Reason |
| --- | --- | --- |
| 1 | Ranking PR 1; Fidelity PR 1; Activity PR 1; Calibration PR 1 | Fix the known selector defect and establish contracts and baselines without broad behavior changes. |
| 2 | Ranking PR 2; Fidelity PRs 2-5; Activity PRs 2-7; Calibration PRs 2-5 | Make selection auditable, persist accepted specs/receipt plumbing, create durable progress primitives, and make evaluation faithful. |
| 3 | Ranking PR 3; Fidelity PR 6 then Activity PR 8; Fidelity PR 7 then Activity PR 9; Activity PRs 10-11; Calibration PR 6 | Serialize overlapping worker edits, establish classifier/YOLO truth and progress, add live-state reads, and persist candidate provenance. |
| 4 | Fidelity PRs 8-10; Activity PRs 12-13; Calibration PRs 7-8 | Establish realized identity/evidence/export parity, cut the UI over incrementally, then link and report trustworthy outcomes. |
| 5 | Fidelity PRs 11-12; Activity PR 14; Calibration PRs 9-11 | Enforce fidelity, measure the activity rollout, shadow ranker v2, require structural gates, and make strict validation default. |
| 6 | Calibration PR 12; Activity PR 15 after its compatibility window | Gather controlled v2 outcomes, promote by cohort, and retire legacy activity only after observation. |

Ranking PR 1 and Activity PRs 1-7 do not need to wait for worker fidelity. Calibration PRs 7-12 require Fidelity PRs 8-9 either directly or transitively because outcome learning must use realized identity and eligibility. Within Wave 4, Calibration PR 7 follows Fidelity PRs 8-9; those items are not unrestricted parallel work.

Fidelity PRs 6-7 and Activity PRs 8-9 touch the same classifier/YOLO worker files. They are logically parallel tracks but must be implemented in the serialized/rebase order shown in Wave 3.

## Migration Coordination

Several tracks add migrations and may be developed concurrently. Planning documents intentionally do not reserve numeric filenames. Each migration-bearing PR must rebase immediately before merge, take the next available migration number, and pass a check that filenames are unique and strictly ordered.

## Shared Release Gates

Every track should meet these gates before its final rollout PR is merged:

1. Existing stored records and payloads still deserialize.
2. New fields are additive until all readers have migrated.
3. Deterministic canned/structural replay fixtures produce stable output. Repeated live evaluations are analyzed statistically and are not expected to be byte-identical.
4. Logs and activity payloads contain no raw prompts, secrets, storage credentials, or large unbounded JSON.
5. Classification and object-detection smoke scenarios both pass where the change is task-sensitive.
6. Feature flags have an explicit default, owner, rollback procedure, and deletion condition.
7. Documentation identifies which metrics prove that the change is an improvement.

## Suggested Branch And PR Naming

Use the normal `codex/` branch prefix when Codex creates a branch. Suggested PR title prefixes:

- `fidelity:` for experiment execution fidelity
- `ranking:` for candidate selection correctness
- `activity:` for progress reporting
- `calibration:` for agent evaluation and rollout

## Out Of Scope

These plans do not add video tasks, new model families, or Roasty integration. They create the reliable execution, observability, and evaluation foundation that those features should build upon.
