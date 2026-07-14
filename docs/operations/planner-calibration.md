# Planner Calibration Commands

The planner calibration tooling has two deliberately separate paths.

## Deterministic rubric evaluation

Run the checked-in starter scenarios without credentials or network access:

```bash
cd services/orchestrator
go run ./cmd/planner-eval
```

This emits one bounded JSONL artifact. Scenarios are sorted by fixture name, and
the result records parse status, backend schedulability, first-pass/eventual
validity, decision/mechanism/evidence/task/stop checks, and safety checks. It
does not use candidate ranker scores as quality labels.

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

The provider-call budget reserves the configured maximum tool rounds before an
invocation, so it is a hard upper bound even if a tool loop or failed request
does not return complete usage. Each result records the actual production-built
initial request size/hash, an immutable finalizer-input hash, provider usage,
locally measured latency, tool traces, validation/retry status, and parsed
output. Cost is emitted only when a matching versioned pricing snapshot is
configured.

The command imports no store or API package and has no mutation capability for
projects, plans, jobs, memory, decisions, or champions. Output lines are also
bounded by `-max-jsonl-bytes`.
