# Frontend Mission Control Report

## Changed Screens And Components

- Added a demo-oriented experiment timeline near the top of Mission Control. It synthesizes project creation, dataset upload/profile, plan creation, job launch/completion, training monitor memory, planner decision, backend gate status, follow-up scheduling, and champion selection from existing project detail responses.
- Added a dataset intelligence panel using the current dataset `profile` payload. It shows class distribution, image/class counts, imbalance ratio, size range, corrupt image count, artifact detection status, recommended metrics, preprocessing suggestions, and warnings.
- Expanded agent decision visibility with reasoning cards for summary, evidence, diagnosis, hypothesis, proposed experiments, rejected options, tradeoffs, risks, and confidence when those fields exist in the decision payload.
- Added backend gate and rejection visibility for rejected planner options and candidate-ranking failures such as duplicate signatures, unsupported options, minor changes, cost, objective-fit, planning-mode, and low-score rejection reasons.
- Added a champion comparison table across completed training summaries with accuracy, macro-F1, runtime, estimated cost, latency, model size, and objective-fit score when evaluation payloads are available.
- Improved responsive behavior for Mission Control panels, tables, timeline cards, settings, reasoning cards, and narrow screens.

## Backend Data Still Needed

- First-class dataset profile API fields would make the dataset intelligence panel more complete: `class_distribution`, `image_dimension_stats`, `metadata_summary`, `artifact_summary`, `recommended_metrics`, and `recommended_preprocessing`.
- Mission Control can show `agent_decisions.payload` reasoning, but a compact latest agent-invocation endpoint would make validation failures visible even when no durable decision is created.
- Champion comparison would benefit from normalized model deployment fields on every evaluation: `estimated_latency_ms`, `model_size_mb`, `parameter_count`, and an objective-fit score with a consistent key.
- Rejection visibility is currently inferred from `rejected_options`, `candidate_rankings`, and optional `validation_error`. A normalized rejection event list would support clearer filtering and audit trails.
- Timeline accuracy would improve if execution events covered dataset upload/profile completion, monitor evaluation, planner validation accepted/rejected, and champion persistence with stable event types.

## Remaining UX Issues

- The app still presents many controls in one long scroll. A future pass should add section navigation or pinned tabs for Overview, Agents, Runs, Data, and Operations.
- Manual job creation remains JSON-first. It is useful for debugging, but demos would benefit from a safer form-based queue builder.
- Agent memory and strategy scorecards are helpful but still dense; they should eventually link back to the source decision, follow-up plan, and outcome.
- Dataset metadata detection is defensive because the current profiler mostly reports folder/image stats. Metadata-rich datasets will need richer profiling before the UI can explain them well.

## Demo Walkthrough Notes

1. Create a project and upload an image-folder dataset.
2. Watch the timeline move from dataset uploaded to profiled, then inspect Dataset Intelligence for imbalance, image sizes, warnings, and suggested metrics.
3. Execute the generated experiment plan and use Training Run Summary plus Champion Comparison to explain quality, runtime, cost, latency, and fit.
4. Run the experiment reviewer. Use Agent Decisions to show what evidence the planner used, what hypothesis it proposed, which options it rejected, and what the backend accepted or blocked.
5. If the decision is `ADD_EXPERIMENTS`, schedule the follow-up and show the timeline linking the accepted decision to a new plan.
6. Once a champion exists, use Selected Champion and Champion Comparison to explain why it won beyond raw accuracy.
