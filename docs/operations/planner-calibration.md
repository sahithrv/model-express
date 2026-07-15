# Planner Calibration Commands

The planner calibration tooling has two deliberately separate paths.

## Deterministic rubric evaluation

Run the complete checked-in corpus without credentials or network access:

```bash
cd services/orchestrator
go run ./cmd/planner-eval
```

This emits one bounded JSONL artifact for 15 scenarios and 15 targeted failing
mutations. The corpus covers imbalance/minority failure (classification and
detection), overfit, underfit, plateau, exhausted architecture shopping,
latency/cost constraints, duplicates/no-ops, label-audit wait behavior, and
successful/failed/rejected memory. Every supported decision type is present.
Scenarios are sorted by fixture name and include a review explanation and
coverage tags alongside parse status, backend schedulability, first-pass/eventual
validity, decision/mechanism/evidence/task/stop checks, safety, response bytes,
and mutation detection. Candidate ranker scores are never quality labels.

The deterministic path explicitly uses relaxed backend finalization so shell
configuration cannot change the checked artifact. It does not make live LLM
calls.

## Checked baseline and review procedure

Verify the corpus against the checked baseline:

```bash
cd services/orchestrator
go run ./cmd/planner-eval -check-baseline
```

The comparison reports changes per scenario with the scenario's explanation.
It applies three independent tolerance classes:

- **Critical:** decision, backend schedulability, safety, scenario removal, and
  expected mutation detection. The checked tolerance is zero regressions.
- **Quality:** correctness-check regressions and corpus pass rate. The checked
  pass-rate floor is 100%.
- **Efficiency:** response-byte growth. The checked allowances are 256 bytes per
  scenario and 2,048 bytes across the corpus. Regressions are emitted as
  warnings by the required gate because response bytes are only a deterministic
  token/cost proxy. They fail only when `-gate-efficiency` is explicitly set.

The required orchestrator CI workflow runs the checked comparison without live
provider flags or credentials. Critical safety/validation and quality tolerance
regressions fail the job. To exercise a hard efficiency gate explicitly, run:

```bash
cd services/orchestrator
go run ./cmd/planner-eval -check-baseline -gate-efficiency
```

To update the baseline, first inspect the normal artifact and the failed
comparison, explain every changed scenario in review, then run:

```bash
cd services/orchestrator
go run ./cmd/planner-eval \
  -update-baseline internal/agents/evals/baselines/planner_rubric_v1.json
go run ./cmd/planner-eval -check-baseline
go test ./internal/agents/evals ./cmd/planner-eval
```

Do not update the baseline only to make a regression green. New scenarios are
reported as critical review items until the checked baseline is intentionally
replaced.

## Planner validation rollout modes

`MODEL_EXPRESS_PLANNER_VALIDATION_MODE` is the source of truth for planner
validation configuration. New and cloud configurations default to `strict`;
missing or invalid settings also resolve to strict:

- `relaxed` preserves the compatibility behavior as a one-release rollback.
- `shadow_strict` returns the relaxed recommendation but executes the same
  strict checks used by `strict`, persisting typed would-block verdicts.
- `strict` blocks and retries on those strict findings.

The old `MODEL_EXPRESS_STRICT_PLANNER_VALIDATION` boolean is read only as a
compatibility alias when the mode variable is absent. Setting the mode to
`relaxed` is the one-switch rollback. Every relaxed resolution emits a
`planner_validation_relaxed_rollback` diagnostic with a monotonically
increasing process-local count and policy version. Remove the relaxed and
legacy compatibility paths after one release; do not silently make them the
default again.

The strict-default rollout policy requires at least 100 observations, at least
98% eventual validity, no more than 20% retries, and no more than 5%
unsupported first-pass proposals. One backend-validation retry is allowed per
attempt group, so a correctable first-pass LLM violation is not itself an
unsafe schedule. Unsafe schedules and post-validation escapes must both remain
exactly zero; either is an immediate pause/rollback signal.

Planner invocations persist `strict_validation_verdict` and
`validation_outcome`. Findings have stable codes, categories, and stages for
missing evidence, mechanism mismatches, invalid task/model proposals,
proposal-time no-ops, and other strict-contract failures. Outcomes record
first-pass status, eventual status, and retry outcome while the existing
attempt group/index/reason fields preserve attribution to the exact planner
variant.

Proposal-time no-op validation compares accepted-spec hashes, so unsupported
or default-equivalent request differences cannot disguise the same executable
configuration. Execution-time requested-versus-realized mismatches remain in
the execution-fidelity reports and are not inferred by planner validation.

## Guarded planner-policy rollout

Planner-policy experiments are disabled in every checked configuration. When
enabled, each project receives a deterministic SHA-256 cohort bucket under the
versioned `planner_project_cohort_v1` rule. The bucket is stable across process
restarts and nested across the guarded `5 -> 25 -> 50 -> 100` stages. Every
planner invocation and candidate provenance row persists the cohort ID, active
policy ID, and full assignment, so later outcome comparisons use the policy
that actually selected and executed the candidate.

The retained rollout controls are:

- `MODEL_EXPRESS_PLANNER_ROLLOUT_ENABLED`
- `MODEL_EXPRESS_PLANNER_ROLLOUT_POLICY_ID`
- `MODEL_EXPRESS_PLANNER_ROLLOUT_LAST_APPROVED_POLICY_ID`
- `MODEL_EXPRESS_PLANNER_ROLLOUT_STAGE_PERCENT`
- `MODEL_EXPRESS_PLANNER_ROLLOUT_STATE` (`active`, `paused`, or
  `manual_review`)
- `MODEL_EXPRESS_PLANNER_ROLLOUT_DIMENSIONS` (`ranker`, `prompt`, `context`, or
  `retrieval`)
- `MODEL_EXPRESS_PLANNER_ROLLOUT_FACTORIAL`
- `MODEL_EXPRESS_PLANNER_ROLLOUT_ROLLBACK`
- `MODEL_EXPRESS_PLANNER_ROLLOUT_RANKER_VERSION`,
  `MODEL_EXPRESS_PLANNER_ROLLOUT_PROMPT_VERSION`,
  `MODEL_EXPRESS_PLANNER_ROLLOUT_CONTEXT_VERSION`, and
  `MODEL_EXPRESS_PLANNER_ROLLOUT_RETRIEVAL_VARIANT`

Exactly one dimension may differ from the last approved policy. Multiple
dimensions are rejected unless `MODEL_EXPRESS_PLANNER_ROLLOUT_FACTORIAL=true`
explicitly declares a factorial experiment. Paused and manual-review policies
assign every project to the last approved policy. Setting
`MODEL_EXPRESS_PLANNER_ROLLOUT_ROLLBACK=true` is the one-switch rollback and
also assigns every project to that last approved policy without removing any
rollout flag.

Promotion requires at least 50, 100, 200, and 400 treatment/control samples at
the 5%, 25%, 50%, and 100% checkpoints, respectively, with the same minimum of
observed executed outcomes. Safety and task-compatibility regressions must be
zero. First-pass validity may be no more than 2 percentage points inferior,
eventual validity no more than 0.5 points inferior, and observed outcomes no
more than 1 point inferior. Cost and latency increases are each capped at 10%,
failure rate at 5%, and retry rate at 20%. Insufficient samples hold the stage;
bounded non-critical regressions pause it; any safety, task-compatibility, or
unexecuted-counterfactual outcome claim requires manual review. At 100%, the
policy is approved only after the same checks pass.

Ranker shadow artifacts remain comparisons, not outcomes. An outcome is
attributed only to a candidate selected and actually executed by its persisted
policy. Candidates selected solely by an inactive comparison policy are
unexecuted counterfactuals and receive no outcome label.

## Candidate provenance and forecast contract

Migration `021_candidate_provenance.sql` creates one immutable decision-time
row for every candidate in an accepted `ADD_EXPERIMENTS` recommendation. Rows
are written before manual and autonomous scheduling paths diverge and contain
the invocation, decision, exact planner variant, zero-based candidate and
selected-experiment indexes, requested and accepted execution hashes, task,
mechanism, base ranker score, selection state/reasons, and the frozen candidate
forecast. Unselected and ranker-rejected candidates start with
`outcome_status=unknown`; selection is not treated as an observed outcome.

The forecast is sourced only from `candidate.expected_metric_impact` and keeps
`expected_delta_vs_champion` as a separate recommendation-level field. Its
contract freezes the target, direction, score basis/version, baseline job and
score, units, and finite delta range at decision time. The backend fills this
contract for compatibility with responses produced before the prompt update
and rejects supplied contracts that disagree with the frozen context.

Decision and candidate insertion are atomic in PostgreSQL and the memory
store. The existing-decision path also performs an idempotent ensure so a
decision-only partial write from a non-transactional boundary can be repaired
without duplicate rows. Follow-up plan, experiment, job, and realized-effective
hash columns intentionally remain null until outcome finalization. Invalid
pre-acceptance attempts remain only in the invocation audit.

The rollout is additive and has no scheduling flag. Existing historical
decisions are not inferred or backfilled because they may lack a trustworthy
frozen forecast or accepted-spec identity; provenance begins with decisions
that declare `candidate_provenance_v1`.

## Opt-in paired provider evaluation

Live generation is read-only and requires all of the following:

1. `-live` on the command line.
2. `MODEL_EXPRESS_PLANNER_EVAL_LIVE=true` in the environment.
3. Positive provider-call and request-byte budgets.
4. Normal `MODEL_EXPRESS_LLM_*` provider/model credentials and settings.

Example for one fixture and the three production request variants:

```bash
cd services/orchestrator
MODEL_EXPRESS_PLANNER_EVAL_LIVE=true \
go run ./cmd/planner-eval \
  -live \
  -fixtures internal/agents/evals/testdata/classification_smoke.json \
  -repeats 1 \
  -max-attempts 1 \
  -max-provider-calls 15 \
  -max-request-bytes 500000
```

With no `-fixtures` argument, live evaluation keeps the four starter scenarios;
the expanded corpus is deterministic by default and must be opted into for live
runs with explicit fixture paths. The provider-call budget reserves the configured maximum tool rounds before an
invocation, so it is a hard upper bound even if a tool loop or failed request
does not return complete usage. Each result records the actual production-built
initial request size/hash, an immutable finalizer-input hash, provider usage,
locally measured latency, tool traces, validation/retry status, and parsed
output. Cost is emitted only when a matching versioned pricing snapshot is
configured.

The command imports no store or API package and has no mutation capability for
projects, plans, jobs, memory, decisions, or champions. Output lines are also
bounded by `-max-jsonl-bytes`.
