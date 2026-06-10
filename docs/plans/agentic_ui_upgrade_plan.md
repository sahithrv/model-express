# Agentic UI Upgrade Plan

Status: Draft  
Date: 2026-06-09  
Owner: Product Design / Frontend  
Target demo window: 1 week

## Executive Summary

Model Express should stop presenting itself as an ML operations console and start presenting itself as an autonomous ML engineer completing a mission.

The current UI has the right technical capabilities, but it exposes them in the wrong order. It foregrounds datasets, plans, metadata imports, agent logs, worker timelines, automation settings, and telemetry. That makes the product feel like a machine that requires supervision.

The redesigned UI should answer four questions in the first 10 seconds:

1. What mission is the AI engineer working on?
2. What is it doing right now?
3. Why did it make the latest decision?
4. What result has it achieved?

The north star is:

> Model Express is an autonomous ML engineer that trains, evaluates, refines, and exports a model for you.

Not:

> Model Express is an AutoML dashboard with agent telemetry.

## Reference Research

These references were inspected through the Browser MCP workflow and public product pages on 2026-06-09.

- [OpenAI Codex](https://openai.com/codex/): strongest reference for task-centric agent work, current activity visibility, parallel work sessions, and hiding internal orchestration behind simple verbs like reading, editing, testing, and committing.
- [Claude Code](https://claude.com/product/claude-code) and [Claude Code docs](https://code.claude.com/docs/en/overview): strongest reference for a minimal default surface, technical detail behind expansion, local context chips, and user control.
- [Devin](https://devin.ai/) and [Cognition's Devin launch post](https://cognition.ai/blog/introducing-devin): strongest reference for a session feed plus artifact/result preview, multi-step progress, PR/result cards, elapsed work time, and "teammate" positioning.
- [Manus](https://manus.im/): strongest reference for prompt-first, mission-first positioning. The public homepage is extremely sparse: one question, one task input, and task categories. The useful lesson is "less structure, more intelligence."
- [Linear](https://linear.app/): secondary reference for human and agent collaboration in one activity timeline, crisp status labels, and low-noise product navigation.
- [Vercel](https://vercel.com/): secondary reference for restrained visual hierarchy, outcome-oriented status language, and product confidence without dashboard clutter.

## Current UX Critique

This critique is based on the current local Mission Control render, source structure in `apps/mission-control/src/App.tsx`, and the active CSS/navigation model in `apps/mission-control/src/styles.css`.

### What Works

- The product already has valuable technical depth: dataset profiling, visual analysis, experiment plans, evaluations, memory, agent decisions, worker state, export, and local demo inference.
- A first pass of `MissionDigest` and `MissionOverview` already exists. This is the right direction and should be evolved rather than discarded.
- Tabs already hide some detail by section, which means the redesign can be achieved with a focused IA/component pass instead of a full app rewrite.
- The existing system can already create the raw ingredients needed for a strong agentic UI: observations, decisions, run outcomes, rejected options, champion selection, and export status.

### What Fails

The current UI still says "control plane" before it says "AI engineer."

Observed first-viewport language includes:

- "Agentic vision training control plane"
- "ORCHESTRATOR OFFLINE"
- "Control Plane"
- "Live LLM activity"
- "No planner, memory, validation, worker, or training activity is active"
- Tabs for `Overview`, `Data`, `Experiments`, `Agents`, `Operations`, `Export`

This language exposes implementation structure instead of user value. A recruiter or non-technical user should not need to know what an orchestrator, planner, memory retrieval, validation event, or worker handoff is.

### Specific Issues

1. The first screen is not mission-led.
   - It starts with system state and app infrastructure.
   - It should start with the mission: "Train Drone Detector" or "Build a helmet classifier."

2. The active work is not human-readable enough.
   - "Live LLM activity" is a system label.
   - The user needs "What Model Express is doing" or "AI engineer is reviewing failed detections."

3. Reasoning is mixed with telemetry.
   - Agent decisions, memory retrieval, validation, scorecards, and invocations compete with one another.
   - Reasoning should be summarized as Observation, Reasoning, Decision, Expected Outcome.
   - Telemetry should be hidden in Developer View.

4. Navigation reflects subsystems, not user questions.
   - `Data`, `Experiments`, `Agents`, and `Operations` are implementation areas.
   - The user wants `Mission`, `Activity`, `Results`, and `Export`.

5. Too many panels have equal visual weight.
   - Dataset Intelligence, Visual Dataset Analysis, Experiment Timeline, Automation Settings, Champion Export, Training Run Summary, Champion Comparison, Live Agent Activity, Agent Decisions, Decision Quality, Telemetry, Invocation Audit, Memory Retrieval, and Workers all look like peers.
   - They are not peers. Mission status, current thinking, timeline, and current champion should dominate.

6. The UI overuses internal nouns.
   - "metadata import"
   - "agent invocations"
   - "memory retrieval probe"
   - "automation timeline"
   - "strategy scorecards"
   - These should exist, but only behind technical expansion or Developer View.

7. The no-project state undersells the product.
   - It says to create or choose a project.
   - It should preview the core promise: "Give Model Express a dataset and a goal. It will train, compare, refine, and export the best model."

8. Result presentation is too metric-first.
   - A champion model needs an explanation: why it won, what improved, what tradeoffs remain, and whether export is ready.

9. The current dark-green control-plane visual language feels operational.
   - It is competent, but it reinforces "ML Ops dashboard."
   - The redesign should feel calmer and more like a work session: lighter hierarchy, stronger whitespace, fewer bordered panels, clearer current task affordance.

## Product Positioning

### One-Sentence Product Frame

Model Express is an autonomous ML engineer that turns a dataset and goal into a trained, evaluated, export-ready model.

### Interface Frame

Every screen should imply:

- A mission exists.
- An AI engineer is working.
- The AI engineer learns from evidence.
- The AI engineer makes decisions.
- The AI engineer produces a usable result.

### Copy Rules

Use human work verbs:

- Reviewing dataset
- Finding failure patterns
- Comparing experiments
- Testing a refinement
- Selecting champion
- Preparing export

Avoid internal system verbs:

- planner_started
- memory_retrieval_logged
- agent_outcome_recorded
- worker_requirement_created
- invocation_validated

Translate internals:

| Internal language | User-facing language |
| --- | --- |
| planner_started | AI is planning the next experiment |
| memory_retrieval_logged | AI reviewed previous experiments |
| agent_outcome_recorded | AI recorded what it learned |
| validation_rejected | Proposed experiment was blocked |
| worker_requirement_created | Training capacity requested |
| champion_selected | Best model selected |
| telemetry summary | Developer usage details |

## New Information Architecture

### Top-Level Navigation

Primary navigation should be:

1. Mission
2. Activity
3. Results
4. Export

Optional:

5. Developer View

`Developer View` should not appear as a peer tab by default. It should be accessed through a top-right toggle, command menu, or "View technical details" action.

### Tier 1: Always Visible On Mission

These dominate the Mission screen.

1. Mission Card
   - Goal
   - Status
   - Progress
   - Best result
   - Estimated time remaining
   - Current next action or blocker

2. Current AI Thinking
   - Observation
   - Reasoning
   - Decision
   - Expected outcome

3. Progress Timeline
   - Dataset profiled
   - Initial experiments
   - Evaluation complete
   - Refinement round
   - Champion selection
   - Export

4. Results Snapshot
   - Current champion
   - Primary metric
   - Secondary confidence metric
   - Export readiness
   - "Why this is best"

### Tier 2: Collapsed By Default

These are available from Mission, Activity, or Results, but collapsed until requested.

- Dataset details
- Experiment details
- Agent memories
- Evaluation reports
- Run comparisons
- Strategy scorecards
- Visual analysis
- Failure examples
- Cost/runtime details

### Tier 3: Developer View

These never appear by default.

- Raw agent invocations
- Memory retrieval logs
- Validation events
- Worker logs
- Prompt traces
- Execution metadata
- SSE/polling status
- Token usage telemetry
- Raw payloads and IDs

## Navigation Redesign

### Current Navigation

- Overview
- Data
- Experiments
- Agents
- Operations
- Export

This navigation requires the user to understand how the system is built.

### Proposed Navigation

- Mission
- Activity
- Results
- Export

This navigation matches the user's mental model.

### Mapping Existing Content

| Current section | New location | Default state |
| --- | --- | --- |
| Overview | Mission | Always visible |
| Dataset Intelligence | Mission detail drawer + Developer View | Collapsed |
| Metadata Import | Developer View | Hidden |
| Visual Dataset Analysis | Activity evidence + Results diagnostics | Collapsed |
| Training Run Summary | Results | Visible after runs exist |
| Champion Comparison | Results | Visible after 2+ completed runs |
| Experiment Plans | Activity and Developer View | Summary visible, raw plan collapsed |
| Agent Logs | Activity | Summarized |
| Operations Timeline | Developer View | Hidden |
| Automation Settings | Mission settings drawer | Collapsed |
| Workers | Developer View | Hidden |
| Export | Export | Visible |

### Navigation Copy

Use:

- Mission
- Activity
- Results
- Export

Avoid:

- Agents
- Operations
- Control Plane
- Telemetry
- Invocation Audit

## Homepage Wireframe

Desktop target: 1280 x 720 first viewport.

```text
+----------------------------------------------------------------------------+
| Model Express                         AI engine: Working        Export ready |
+----------------------------------------------------------------------------+
| Missions           | Train Drone Detector                         Running   |
|--------------------|--------------------------------------------------------|
| > Drone Detector   | Goal                                                   |
|   Helmet Classifier| Detect small drones in aerial images and export ONNX.  |
|   Defect Finder    |                                                        |
|                    | Progress: 6 / 12 experiments     Best: 0.78 mAP50-95  |
| New Mission        | ETA: 18 min                     Current: Refinement    |
|                    |                                                        |
|                    | What the AI is doing                                   |
|                    | Observation                                            |
|                    | Small drones are causing most misses.                  |
|                    | Reasoning                                              |
|                    | The current model struggles with distant objects.      |
|                    | Decision                                               |
|                    | Testing a larger image size with YOLO11S.              |
|                    | Expected outcome                                       |
|                    | Improve small-object recall without large latency cost.|
|                    |                                                        |
|                    | [Dataset] [Initial Runs] [Evaluation] [Refinement]     |
|                    |    done       done          done         active        |
|                    |                                                        |
|                    | Current champion                         Why it leads  |
|                    | YOLO11S  0.78 mAP50-95  Export ready     Best recall   |
+----------------------------------------------------------------------------+
```

### First Viewport Rules

- The mission name must be the largest semantic label.
- The current AI thinking block must be visible without scrolling.
- Progress timeline must be visible without scrolling.
- Best result must be visible without scrolling.
- No raw IDs, storage paths, worker names, JSON, or prompt traces.
- Only one primary action.

## Mission Page Redesign

### Primary Layout

Use a three-zone layout:

1. Left sidebar
   - Mission list
   - New Mission
   - Optional filters: Running, Complete, Blocked

2. Main mission column
   - Mission Card
   - Current AI Thinking
   - Progress Timeline
   - Key Activity

3. Right inspector
   - Current champion/result
   - Next action
   - Export readiness
   - Collapsible evidence

### Mission Card

Example:

```text
Train Drone Detector
Running experiments

Goal
Detect small drones in aerial imagery and export a production-ready ONNX model.

Progress
6 / 12 experiments completed

Best result
0.78 mAP50-95

Estimated time remaining
18 minutes
```

### Current AI Thinking

This is the most important component.

It should feel alive, but it must not reveal hidden chain-of-thought or raw prompts.

Example:

```text
What the AI is doing

Observation
Small drones are causing most missed detections.

Reasoning
The model is performing well on large objects but loses detail at distance.

Decision
Launch a refinement experiment with larger image size.

Expected outcome
Improve small-object recall while keeping export latency acceptable.
```

### Current AI Thinking States

| State | User-facing text |
| --- | --- |
| No project | Waiting for a mission |
| Dataset selected | Reviewing dataset structure |
| Profiling | Profiling images and labels |
| Planning | Planning first experiments |
| Training | Waiting for experiments to finish |
| Reviewing | Comparing results and failure patterns |
| Refining | Testing a targeted improvement |
| Selecting champion | Choosing the best model |
| Exporting | Preparing export package |
| Blocked | Needs your input |

## Activity Feed Redesign

### Goal

The Activity page should feel like watching an AI coworker work.

It should not feel like reading logs.

### Feed Structure

Each event becomes one of these cards:

- Mission update
- Observation
- Decision
- Experiment started
- Experiment finished
- Result improved
- Blocker
- User action
- Export prepared

### Activity Card Anatomy

```text
[12:41 PM] Decision
Test larger image size

Observation
Small-object recall is lagging behind large-object precision.

Why this matters
The target use case depends on distant drone detection.

Action
Started YOLO11S at 960px.

[View technical details]
```

### Technical Detail Expansion

Collapsed by default:

```text
Technical details
- Source: planner decision
- Experiment signature: hidden until expanded
- Validation: accepted
- Memory evidence: 3 related prior runs
- Raw payload: Developer View only
```

### Feed Filtering

Default filters:

- All
- Decisions
- Experiments
- Results
- Blockers

Developer filters only in Developer View:

- Invocations
- Memory
- Validation
- Workers
- Telemetry

### Activity Feed Copy Rules

Bad:

```text
memory_retrieval_logged
Retrieved 8 records from agent_memory where source_table=training_run_evaluations.
```

Good:

```text
AI reviewed previous experiments
It found that larger image sizes helped small-object recall in similar datasets.
```

Bad:

```text
planner.validation_rejected
```

Good:

```text
Proposed experiment was blocked
The change was too similar to a completed run, so Model Express skipped it.
```

## Timeline Redesign

### Goal

The timeline should show mission progress, not system events.

### Desktop Timeline

```text
[done] Dataset Profiled
       4,820 images, 3 classes, small-object risk detected

[done] Initial Experiments
       YOLO11N and YOLO11S completed

[done] Evaluation Complete
       Best run reached 0.76 mAP50-95

[active] Refinement Round
         Testing larger image size for small-object recall

[waiting] Champion Selection
          Starts after refinement results

[waiting] Export
          ONNX package prepared after champion selection
```

### Timeline Principles

- Stages are mission-level, not event-level.
- Each stage has one sentence explaining the user value.
- Each stage can expand to show technical evidence.
- Active stage should be visually prominent.
- Completed stages should show what was learned, not just that something finished.

### Recommended Stage Model

1. Mission Created
2. Dataset Reviewed
3. Initial Plan Created
4. Initial Experiments
5. Evaluation
6. Refinement
7. Champion Selection
8. Export
9. Demo Validation

### Stage Detail Drawer

Example for `Refinement`:

```text
Refinement Round

What happened
Model Express found that small drones account for most misses.

What it is doing
Testing larger image resolution with the current best architecture.

Success criteria
Recall improves without unacceptable latency increase.

Technical details
[collapsed]
```

## Results Page Redesign

### Goal

Results should explain the achieved outcome and why the AI chose it.

### Results First View

```text
Current Champion
YOLO11S

Primary result
0.78 mAP50-95

Why it won
Best balance of small-object recall, latency, and export size.

Compared with baseline
+0.11 mAP50-95
+0.08 recall
+14 ms latency

Export
ONNX ready
```

### Results Sections

1. Champion Summary
   - Model name
   - Primary metric
   - Improvement over baseline
   - Export status
   - "Why it won"

2. Learning Summary
   - What the AI learned from the dataset
   - What failed
   - What improved
   - Remaining risks

3. Comparison
   - Top 3 candidates only by default
   - Full table collapsed

4. Failure Analysis
   - Most common misses
   - Examples if available
   - Recommended next mission if needed

5. Export Readiness
   - Format
   - Validation status
   - Test image status
   - Known limitations

### Champion Card Copy

Bad:

```text
Champion Comparison
mAP50_95: 0.781
objective_fit: 0.82
```

Good:

```text
Current champion: YOLO11S
It is the best model because it improves small-drone recall while staying within the export latency target.
```

## Export Page Redesign

Export should feel like the handoff from the AI engineer.

### Export Page Layout

```text
Export Package
Ready

Includes
- ONNX model
- Preprocessing contract
- Label map
- Model card
- Test images

Validation
Passed local inference test

Use this when
You need drone detection in an edge or API deployment.

Known limitations
Small drones under 12px remain difficult.
```

### Export Page Actions

Primary:

- Download package

Secondary:

- Run demo image
- View model card
- Compare champion
- Open technical manifest

Raw manifest belongs behind "Open technical manifest", not in the default export page.

## Mobile And Responsive Considerations

### Mobile Principle

Mobile should be a mission status and review surface, not a dense operations console.

### Mobile Navigation

Use bottom tabs:

- Mission
- Activity
- Results
- Export

Developer View should be hidden behind settings or command menu.

### Mobile Mission Layout

Order:

1. Mission status
2. Current AI Thinking
3. Progress timeline
4. Current champion
5. Next action
6. Recent activity
7. Collapsed details

### Mobile Activity Feed

- One column.
- Cards should use short titles and one-line summaries.
- Technical expansion opens a full-screen sheet.
- Avoid side-by-side metrics.
- Timeline becomes vertical.

### Responsive Breakpoints

- `>= 1180px`: three-zone layout.
- `760px - 1179px`: two-zone layout, right inspector moves below mission card.
- `< 760px`: single-column, bottom navigation, sticky current status.

### Text Fitting Rules

- Long model names wrap.
- Metric cards use stable min heights.
- Buttons allow two-line labels.
- No horizontal scroll on the Mission page.
- Technical tables can horizontally scroll inside Developer View only.

## Developer View Proposal

### Purpose

Developer View preserves auditability without overwhelming the default product.

It is for:

- Engineers debugging orchestration
- Demo backup when something fails
- Backend validation inspection
- Prompt and telemetry review
- Worker and queue diagnostics

### Access

Options:

1. Top-right `Developer View` toggle.
2. Keyboard shortcut.
3. Command menu item.

Do not show it as a primary tab for normal users.

### Developer View IA

Developer View tabs:

- Invocations
- Memory
- Validation
- Workers
- Telemetry
- Raw Events
- Settings

### Developer View Guardrails

- It can show raw IDs, payloads, storage URIs, and logs.
- It should clearly label sensitive/debug content.
- It should not leak into Mission, Activity, Results, or Export.
- It should have "Back to Mission" as the obvious escape hatch.

## Visual Direction

### Desired Feel

- Autonomous coworker
- Focused work session
- Calm but alive
- Evidence-backed decisions
- Recruiter-readable in 10 seconds

### Avoid

- Dense dashboard grids
- Equal-weight panels
- Raw logs by default
- ML platform visual language
- One-note dark green control-plane palette
- Decorative charts that do not answer a user question

### Recommended Style

- Neutral app shell with restrained color.
- One accent color for active AI work.
- Status tones for done, active, blocked, warning.
- Fewer bordered cards.
- More conversational activity rows.
- Small, clear icons from `lucide-react`.
- 8px radius maximum unless current design system requires otherwise.

### Naming Changes

| Current | Proposed |
| --- | --- |
| Control Plane | Mission |
| Live LLM activity | What the AI is doing |
| Agent Decisions | Decisions |
| Agent Memory | What it learned |
| Operations Timeline | Technical timeline |
| Automation Settings | Autonomy settings |
| Champion Comparison | Why this model won |
| Dataset Intelligence | Dataset review |
| Visual Dataset Analysis | Failure examples |
| Mission Control Telemetry | Usage telemetry |

## Step-By-Step Implementation Plan

This is scoped for a 1-week demo.

### Day 1: IA And Language Pass

Files:

- `apps/mission-control/src/App.tsx`
- `apps/mission-control/src/styles.css`

Tasks:

1. Rename top-level tabs to `Mission`, `Activity`, `Results`, `Export`.
2. Move current `Data`, `Experiments`, `Agents`, and `Operations` content under collapsible sections or Developer View.
3. Replace "Control Plane", "Orchestrator", and "Live LLM activity" copy in default views.
4. Define the mission stage model.
5. Define event translation rules from raw event types to user-facing activity cards.

Acceptance:

- First viewport no longer says "control plane", "orchestrator", "planner", "memory retrieval", "worker", or "invocation" unless Developer View is enabled.

### Day 2: Mission Page

Tasks:

1. Evolve existing `MissionDigest` into `MissionBrief`.
2. Add Mission Card with goal, status, progress, best result, and ETA.
3. Add Current AI Thinking panel with Observation, Reasoning, Decision, Expected Outcome.
4. Add mission-stage timeline.
5. Add compact champion/result snapshot.

Acceptance:

- A recruiter can understand the product from Mission page alone.
- The current AI action is visible without scrolling.

### Day 3: Activity Feed

Tasks:

1. Build an `ActivityFeed` that groups raw events into readable cards.
2. Add card types for Observation, Decision, Experiment Started, Experiment Finished, Result Improved, Blocker, Export.
3. Add collapsed technical details.
4. Move raw event payloads into Developer View.

Acceptance:

- Activity reads like an AI engineer's work journal, not logs.
- Every card has a plain-language title and one-sentence summary.

### Day 4: Results Page

Tasks:

1. Build Champion Summary.
2. Add "Why it won" explanation from existing evaluation and decision data.
3. Show top 3 model candidates by default.
4. Collapse full comparison table.
5. Add learning summary and remaining risks.

Acceptance:

- Results explain the decision, not just the metric.

### Day 5: Export And Developer View

Tasks:

1. Simplify Export page around package readiness and demo validation.
2. Move manifest, raw model profile, and technical artifacts behind expansion.
3. Add Developer View toggle.
4. Move raw invocations, memory logs, validation events, worker logs, telemetry, and raw execution metadata into Developer View.

Acceptance:

- Default app is non-technical.
- Debuggability is preserved.

### Day 6: Responsive And Visual Polish

Tasks:

1. Desktop three-zone layout.
2. Tablet two-zone layout.
3. Mobile one-column layout with bottom nav.
4. Confirm no text overlap.
5. Reduce bordered-card density.
6. Tune visual hierarchy and status colors.

Acceptance:

- Mission first viewport is readable at 1280 x 720.
- Mobile Mission page is usable without horizontal scroll.

### Day 7: Demo Script And QA

Tasks:

1. Prepare demo state examples:
   - No mission
   - Dataset review
   - Experiments running
   - Refinement decision
   - Champion selected
   - Export ready
   - Blocked state
2. Verify transitions between Mission, Activity, Results, Export.
3. Verify Developer View can reveal raw details.
4. Run frontend build.
5. Create a short demo walkthrough.

Acceptance:

- Demo tells one clear story: "I gave Model Express a goal and dataset; it worked like an AI ML engineer and delivered a model."

## React/Electron Component Hierarchy

### Proposed Structure

Keep implementation incremental. Existing `App.tsx` is large, so extract only when the new structure is clear.

```text
App
  AppShell
    AppChrome
    MissionSidebar
      MissionList
      NewMissionButton
      MissionFilters
    Workspace
      TopStatusBar
      MainNavigation
      MissionRoute
      ActivityRoute
      ResultsRoute
      ExportRoute
      DeveloperRoute
```

### Mission Components

```text
MissionRoute
  MissionHeader
  MissionCard
  CurrentAIThinking
    ThinkingRow: Observation
    ThinkingRow: Reasoning
    ThinkingRow: Decision
    ThinkingRow: ExpectedOutcome
  MissionStageTimeline
    MissionStageItem
    MissionStageDrawer
  CurrentChampionSnapshot
  NextActionPanel
  CollapsibleEvidence
```

### Activity Components

```text
ActivityRoute
  ActivityFilterBar
  ActivityFeed
    ActivityCard
      ActivityCardHeader
      ActivityCardSummary
      ActivityCardEvidence
      TechnicalDetailsDisclosure
  ActivityComposerOrPrompt
```

### Results Components

```text
ResultsRoute
  ChampionSummary
  WhyItWon
  LearningSummary
  CandidateComparisonCompact
  CandidateComparisonFullDisclosure
  FailureAnalysis
  ResultArtifacts
```

### Export Components

```text
ExportRoute
  ExportPackageSummary
  ExportValidationStatus
  DemoInferencePanel
  ModelCardPreview
  ExportArtifactList
  TechnicalManifestDisclosure
```

### Developer Components

```text
DeveloperRoute
  DeveloperNav
  InvocationAudit
  MemoryRetrievalLogs
  ValidationEvents
  WorkerDiagnostics
  RawExecutionEvents
  TelemetrySummary
  RawPayloadViewer
```

### Data Derivation Layer

Create derived view models instead of passing raw backend data directly into UI components.

```text
buildMissionBrief(...)
buildCurrentThinking(...)
buildMissionStages(...)
buildActivityFeed(...)
buildResultsSummary(...)
buildDeveloperDiagnostics(...)
```

The UI should render these view models:

- `MissionBrief`
- `AIThinking`
- `MissionStage[]`
- `ActivityCardModel[]`
- `ResultsSummary`
- `ExportSummary`
- `DeveloperDiagnostics`

This keeps product language in one layer and reduces raw backend leakage.

## Data And Event Translation

### MissionBrief

Suggested fields:

```text
MissionBrief
  id
  title
  goal
  statusLabel
  progressLabel
  completedExperiments
  totalExperiments
  bestMetricLabel
  bestMetricValue
  etaLabel
  primaryAction
  blocker
```

### AIThinking

Suggested fields:

```text
AIThinking
  state
  observation
  reasoning
  decision
  expectedOutcome
  confidenceLabel
  updatedAt
  sourceDecisionId
```

Important: `reasoning` here means a short user-facing rationale derived from durable decision fields. It must not expose hidden chain-of-thought or raw prompts.

### ActivityCardModel

Suggested fields:

```text
ActivityCardModel
  id
  type
  title
  summary
  timestamp
  status
  evidenceSummary
  resultSummary
  technicalSource
  developerPayloadRef
```

## Demo Storyboard

### Scene 1: Mission Starts

User sees:

- "Train Drone Detector"
- "Reviewing dataset"
- Goal sentence
- AI thinking says it is checking labels and image sizes

Message:

> Model Express starts like an AI engineer: it reads the assignment and inspects the dataset before deciding what to do.

### Scene 2: AI Learns

User sees:

- Observation: small drones are hard
- Reasoning: distant objects need higher resolution
- Decision: test YOLO11S at larger image size

Message:

> It does not just launch random trials. It explains the evidence behind the next experiment.

### Scene 3: Progress

User sees:

- Timeline with Dataset, Initial Runs, Evaluation, Refinement
- Activity feed with decisions and experiment completions

Message:

> The user can see progress without reading logs.

### Scene 4: Result

User sees:

- Current champion
- Metric improvement
- Why it won
- Export ready

Message:

> The end product is not a dashboard. It is a trained model with an explanation and handoff package.

### Scene 5: Technical Depth

User opens Developer View.

Message:

> The technical audit trail still exists, but it does not dominate the default product.

## Acceptance Criteria

The redesign is successful when:

1. The first viewport clearly communicates the mission, current AI work, progress, and best result.
2. A non-technical recruiter can describe the product after 10 seconds.
3. The default UI does not expose raw orchestration, worker, memory, invocation, or telemetry labels.
4. Technical users can still inspect raw details in Developer View.
5. The Activity page reads like an AI work journal.
6. The Results page explains why the champion won.
7. The Export page feels like a handoff package.
8. Mobile shows mission status, current thinking, progress, and results without horizontal scroll.
9. Existing capabilities are preserved.
10. `npm run build` succeeds in `apps/mission-control`.

## Non-Goals

- Do not redesign Model Express as an ML dashboard.
- Do not copy MLflow, Weights & Biases, Databricks, Kubeflow, or TensorBoard.
- Do not remove technical auditability.
- Do not expose hidden chain-of-thought.
- Do not require a backend rewrite for the 1-week demo.
- Do not add decorative marketing pages.

## Risks And Mitigations

### Risk: Hiding details makes expert users slower

Mitigation:

- Preserve Developer View.
- Add one-click "View technical details" disclosures.
- Keep stable anchors for deep sections.

### Risk: Frontend-derived summaries drift from backend semantics

Mitigation:

- Use a dedicated translation layer.
- Keep raw event source references in Developer View.
- Later add a backend `mission-summary` endpoint if needed.

### Risk: The AI appears to claim more certainty than it has

Mitigation:

- Include "Expected outcome" and "Remaining risk."
- Use conservative wording: "testing", "likely", "based on completed runs."

### Risk: Demo data is incomplete

Mitigation:

- Build graceful empty, active, blocked, and completed states.
- Use existing fields when present.
- Use neutral fallback copy when evidence is unavailable.

## Immediate Next Steps

1. Approve the navigation change: `Mission`, `Activity`, `Results`, `Export`, hidden `Developer View`.
2. Approve the default Mission layout and Current AI Thinking model.
3. Implement Day 1 and Day 2 first, because they deliver the biggest demo impact.
4. Move raw technical panels into Developer View after the Mission page works.
5. Use the demo storyboard to test whether the UI reads as an autonomous AI engineer.
