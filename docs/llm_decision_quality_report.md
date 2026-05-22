# LLM Decision Quality Report

## Scope

This report covers the Model Express LLM decision wrapper for image-classification experimentation. It focuses on planner quality, strategy memory, deterministic diagnosis, candidate ranking, and prompt/schema upgrades. It does not change preprocessing validation or frontend behavior.

## Current Weaknesses

The prior planner shape was valid but too close to prompt polish. It asked for better experiments, but it did not always force a search process before choosing a batch. That made shallow proposals possible: more epochs, small learning-rate nudges, repeated model variants, or a vague "try a stronger model" recommendation.

The strongest gaps were:

- Diagnosis signals were implicit in prose instead of computed before the LLM call.
- Planning mode was not a first-class control variable.
- The LLM could jump straight to final experiments instead of generating and comparing alternatives.
- Rejected options were not consistently remembered as reusable negative guidance.
- Outcome memory existed, but it needed a strategy-level scorecard that future planner calls could read.
- Ranking was not transparent enough to explain why one candidate beat another.
- Preprocessing recommendations could appear without dataset evidence; validation ownership for those fields should remain with PR 1.

## Implemented Wrapper Architecture

The improved wrapper is:

```text
completed plan results
-> deterministic diagnosis
-> compact prompt context with objective, model catalog, memory, scorecards
-> LLM generates 6-12 candidate hypotheses or final experiments
-> deterministic candidate ranking selects 1-5 experiments
-> backend validates schema, planning mode, diversity, novelty, and experiment shape
-> decision payload stores evidence, rejected options, rankings, diagnosis, and scorecards
-> follow-up outcome updates planning memory and strategy scorecards
```

The LLM still cannot create jobs, mutate datasets, create workers, or bypass backend validation.

## Implemented Changes

### Deterministic Diagnosis

`services/orchestrator/internal/agents/diagnosis.go` computes:

- `overfitting_score`
- `underfitting_score`
- `plateau_score`
- `instability_score`
- `class_imbalance_score`
- `minority_class_failure_score`
- `cost_efficiency_score`
- `latency_penalty`
- `improvement_stagnation_score`

It also emits `recommended_failure_modes`, `deterministic_diagnosis_used`, and concise evidence strings. The planner prompt receives this before the LLM call.

### Planning Modes

The planner now recognizes:

- `explore`
- `exploit`
- `champion_challenge`
- `preprocessing_ablation`
- `class_imbalance_ablation`
- `stop_or_select`

Validation adds mode-specific rules. For example, `explore` needs at least two model families, `class_imbalance_ablation` must target class balancing/per-class metrics, and `stop_or_select` cannot propose weak low-delta follow-ups.

### Candidate Hypotheses

The planner schema supports `candidate_hypotheses` with:

- evidence used
- diagnosis and planning mode
- proposed changes
- expected metric impact
- expected tradeoffs and failure modes
- risk, cost, novelty
- similar success/failure memory ids
- complete experiment config

If `proposed_experiments` is empty, the backend ranks candidates and selects final experiments.

### Candidate Ranking

`services/orchestrator/internal/agents/candidate_ranking.go` ranks candidates over:

- expected gain
- novelty
- cost
- risk
- deployment fit
- memory similarity
- redundancy
- diagnosis alignment

This pass added auditable `score_components` to each `CandidateRanking`, plus explicit deployment-fit and redundancy components. Candidate ranking now explains not only the final score, but the contribution of each scoring family.

Example ranking fragment:

```json
{
  "candidate_index": 1,
  "score": 0.842,
  "selected": true,
  "score_components": {
    "base": 0.45,
    "expected_gain": 0.18,
    "novelty": 0.128,
    "cost": 0,
    "risk": -0.06,
    "deployment_fit": 0.06,
    "redundancy": 0,
    "diagnosis_alignment": 0.18,
    "memory_similarity": 0.05
  },
  "reasons": [
    "candidate fits low-latency live objective",
    "aligned with class imbalance/minority failure diagnosis",
    "similar successful strategy scorecard",
    "selected by deterministic backend ranking"
  ]
}
```

### Rejected Options Memory

The planner output includes `rejected_options` with:

- option
- reason
- evidence
- applies_when

The backend extracts applicable rejected options from prior planning feedback/outcome memory and provides them as `rejected_strategy_memory`.

### Strategy Scorecards

Strategy scorecards are now a structured outcome layer separate from raw memory. They track:

- strategy type and planning mode
- dataset traits
- objective profile
- proposed changes
- expected delta
- actual delta
- confidence before/after
- cost and runtime
- outcome and lesson

This lets future planner calls distinguish "valid JSON" from "actually improved the champion."

## Prompt Upgrades

The planner prompt now requires:

- evidence citations
- deterministic diagnosis used
- planning mode
- why each batch can beat the champion
- expected failure modes
- stop condition
- deployment tradeoff
- rejected options
- preprocessing rationale tied to dataset evidence
- model catalog compliance
- candidate hypotheses for backend ranking

Bad prompt behavior:

```text
Run MobileNet again with two more epochs and a slightly lower learning rate.
```

Good prompt behavior:

```text
Minority-class recall is the failure mode, so run a class-imbalance ablation:
weighted loss on a compact champion challenger, class-balanced sampling on the
current family, and a higher-resolution EfficientNet control. Stop if macro-F1
or minority recall fails to improve by the meaningful threshold.
```

## Improved Planner Input Example

```json
{
  "deterministic_diagnosis": {
    "overfitting_score": 0.71,
    "plateau_score": 0.62,
    "class_imbalance_score": 0.6,
    "minority_class_failure_score": 0.76,
    "latency_penalty": 0.5,
    "recommended_failure_modes": ["overfitting", "class_imbalance", "minority_class_failure", "latency_penalty"]
  },
  "objective_context": {
    "primary_objective": "low_latency_live_service",
    "metric_preferences": ["macro_f1", "accuracy", "latency_ms"]
  },
  "rejected_strategy_memory": [
    {
      "option": "same MobileNet with more epochs",
      "reason": "prior runs plateaued",
      "evidence": "plateau_score remained high",
      "applies_when": ["plateau"]
    }
  ]
}
```

## Improved Planner Output Example

```json
{
  "decision_type": "ADD_EXPERIMENTS",
  "planning_mode": "class_imbalance_ablation",
  "deterministic_diagnosis_used": ["minority_class_failure_score=0.760", "latency_penalty=0.500"],
  "evidence_used": ["worst per-class recall is 0.32", "current champion is fast but weak on minority recall"],
  "hypothesis": "Weighted loss or balanced sampling can improve minority recall without abandoning low-latency models.",
  "expected_failure_modes": ["majority precision may fall", "balanced sampling may overfit rare classes"],
  "changed_variables": ["class_balancing", "augmentation", "model_family"],
  "success_criteria": "Improve macro-F1 by 0.02 or minority recall by 0.05 without material latency regression.",
  "stop_condition": "Select the current champion if neither candidate improves minority recall.",
  "rejected_options": [
    {
      "option": "more epochs only",
      "reason": "does not address plateau or minority failure",
      "evidence": "plateau_score and minority_class_failure_score are high",
      "applies_when": ["plateau", "minority_class_failure"]
    }
  ],
  "candidate_hypotheses": []
}
```

## PR Boundary

PR 3/4 should own:

- planner schema and prompt fields
- deterministic diagnosis
- candidate hypothesis generation and ranking
- rejected option memory
- strategy scorecards and planner outcome learning
- tests for planner validation and ranking behavior
- this report

PR 1 should own:

- preprocessing schema
- dataset artifact/profile validation
- bounding-box/crop/metadata validation
- any backend enforcement for shared preprocessing fields

Frontend display of these fields should be a later UI PR unless the owning frontend agent is explicitly assigned.

## Follow-Up Work

- Add semantic retrieval for strategy memory across similar datasets.
- Add richer dataset profile fields through PR 1, then expose them to planner context.
- Add holdout/test-split awareness to reduce validation-set overfitting.
- Add candidate-level expected latency/model-size estimates from the model catalog.
- Add calibration of diagnosis thresholds from observed project outcomes.
- Teach training monitor to emit structured run-level diagnosis that can be aggregated directly.
- Add UI inspection for `score_components`, rejected options, and strategy scorecards.

## Verification

Focused agent tests passed:

```text
go test ./internal/agents
ok model-express/services/orchestrator/internal/agents
```
