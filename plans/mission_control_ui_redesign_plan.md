# Mission Control UI Redesign Plan

Status: Draft
Date: 2026-06-03

## Purpose

Redesign the Mission Control default project screen so it behaves more like Linear, Codex, and GitHub Actions: calm by default, status-forward, and rich only when the operator drills in.

The homepage should answer only three questions:

1. What is happening?
2. Is it healthy?
3. What should I do next?

Every other piece of information must remain accessible, but it should move into focused drill-down views instead of being visible by default.

## Product Principles

1. Status before detail.
   - The default view should summarize the current project state in plain operational language.
   - Prefer "2 training jobs running, best macro-F1 0.812" over "Jobs 12, Plans 3, Workers 2".

2. One primary next action.
   - The Overview should present at most one primary action and up to three secondary actions.
   - Actions should explain why they matter, not just expose buttons.

3. Drill down by intent.
   - Data questions go to Data.
   - Experiment and run questions go to Experiments.
   - Agent reasoning and audit questions go to Agents.
   - Worker, settings, event, and queue questions go to Operations.
   - Champion export, demo, and feedback questions go to Export.

4. Failures stay visible.
   - Warnings, blocked work, unhealthy workers, offline orchestrator state, failed jobs, backend rejections, and unavailable runtimes must be summarized on Overview.
   - The detailed payloads, tables, logs, and reasoning move behind links.

5. No lost capability.
   - This is an information architecture and presentation change first.
   - Existing controls, audit views, manual queue tools, dataset intelligence, decisions, telemetry, and export/demo surfaces should still exist.

## Current UI Baseline

Mission Control already has these tabs in `apps/mission-control/src/App.tsx`:

- `Overview`
- `Data`
- `Experiments`
- `Agents`
- `Operations`
- `Export`

Inactive tab content is already hidden by `data-active-tab` CSS in `apps/mission-control/src/styles.css`, so the redesign does not need a full navigation rewrite.

The current Overview still exposes too much:

- Five generic metric cards: Datasets, Jobs, Plans, Workers, Queued.
- A full experiment lifecycle timeline.
- A full agent activity panel with many events.
- A detailed selected champion panel when a champion exists.

Those are useful details, but they do not all belong in the first viewport.

## Default Screen Contract

The Overview should contain exactly these zones:

1. Project Status Header
   - Project name, concise state label, orchestrator health.
   - Example states: `Setting up dataset`, `Profiling dataset`, `Plan ready`, `Training`, `Review needed`, `Champion selected`, `Blocked`.

2. What Is Happening
   - One prominent sentence plus 2-4 supporting facts.
   - This replaces the generic summary metric grid.
   - Examples:
     - "Dataset profiling is running. Mission Control is waiting for class counts before planning experiments."
     - "3 experiments are training. Best reported macro-F1 is 0.812."
     - "The planner proposed 2 follow-up experiments, but auto execution is off."

3. Is It Healthy
   - A compact health strip with no more than 4 items.
   - Health items should be status chips with short labels:
     - Orchestrator
     - Dataset
     - Workers
     - Runs
     - Agents, only if agent activity is relevant
   - Each item can link to its drill-down tab.

4. What Should I Do Next
   - One primary action.
   - Up to three secondary actions.
   - Actions should include a reason and a target tab when they need detail.

5. Live LLM Activity
   - A compact Codex/Cursor-style activity strip that shows what the agentic system is doing right now.
   - This should make the UI feel alive while long-running planner, memory, validation, or worker handoff steps are in progress.
   - Example statuses:
     - `Thinking`
     - `Retrieving memories`
     - `Reading dataset profile`
     - `Ranking candidates`
     - `Validating plan`
     - `Scheduling workers`
     - `Waiting for training results`
     - `Writing decision`
   - The strip should show high-level work labels only. It must not expose raw prompts, hidden reasoning, full tool payloads, secrets, storage paths, or large JSON.

6. Recent Signals
   - A compact feed of up to 5 important events.
   - This should not be the full timeline or full event log.
   - Include only state-changing events, failures, blocks, successes, and the latest active work.

7. Outcome Summary
   - Only shown when a champion exists.
   - Keep it compact: champion model, primary metric, objective-fit/latency/cost if available, and one "Open Export" or "Compare Runs" action.
   - Full confusion previews, deployment details, prediction history, and test bench remain in Export or Experiments.

## Default Screen Content Budget

The Overview must not show these by default:

- Raw project, job, dataset, worker, decision, or plan IDs.
- Storage URIs.
- JSON configs.
- Full experiment lists.
- Full metric charts.
- Full class distribution.
- Full agent decision reasoning.
- Raw prompts, raw LLM outputs, hidden chain-of-thought, or full tool payloads.
- Candidate ranking details.
- Full execution-event logs.
- Tables longer than 5 rows.
- More than one primary call to action.

Counts are allowed only when they answer health or next action. For example:

- Good: "2 queued jobs, 0 active workers" because it explains a problem.
- Less useful: "Jobs: 24" because it is inventory, not status.

## Proposed Information Architecture

### Overview

Answers the three homepage questions. Uses derived summaries and links into detail tabs.

Default cards:

- `MissionStatusPanel`
- `HealthStrip`
- `NextActionQueue`
- `LiveAgentActivity`
- `RecentSignals`
- `ChampionOutcomeSummary`, conditional

### Data

Owns dataset questions:

- Dataset Intelligence
- Visual Dataset Analysis
- Metadata import status
- Dataset table
- Class distribution
- Preprocessing and metric recommendations
- Dataset warnings and profile details

### Experiments

Owns plan, job, run, and comparison questions:

- Experiment Plan
- Training Run Summary
- Champion Comparison
- Recent Jobs
- Run Metrics
- Selected run evaluation details

### Agents

Owns reasoning and audit questions:

- Detailed live-agent activity timeline
- Agent Decisions
- Decision Quality
- Mission Control Telemetry
- Agent Invocation Audit
- Memory Retrieval Probe
- Agent Memory
- Backend gate and rejection detail

### Operations

Owns controls and infrastructure questions:

- Automation Settings
- Automation Timeline
- Worker Requirements
- Workers
- Manual Job Queue
- SSE/polling state
- Worker startup/supervision detail

### Export

Owns champion use and validation questions:

- Champion Export / Demo
- Export status
- Demo images
- Local/worker prediction status
- Prediction history
- Champion feedback

## Derived UI State

Introduce a compact derived digest in `apps/mission-control/src/App.tsx` before large rendering changes. This keeps the Overview simple and makes the logic testable by inspection.

Suggested shape:

```ts
type MissionDigest = {
  state:
    | "empty"
    | "dataset_needed"
    | "profiling"
    | "plan_ready"
    | "training"
    | "review_needed"
    | "follow_up_ready"
    | "champion_selected"
    | "blocked";
  headline: string;
  detail: string;
  facts: Array<{ label: string; value: string; tone?: "ok" | "warning" | "bad" | "info" }>;
  health: MissionHealthItem[];
  nextActions: MissionNextAction[];
  liveActivity: MissionLiveActivity;
  recentSignals: MissionSignal[];
  champion?: MissionChampionSummary;
};

type MissionLiveActivity = {
  status: "idle" | "active" | "waiting" | "blocked" | "failed" | "succeeded";
  label: string;
  detail: string;
  streamState: ActivityStreamState;
  steps: Array<{
    id: string;
    label: string;
    status: "active" | "waiting" | "succeeded" | "failed" | "blocked";
    timestamp?: string;
    targetTab?: ProjectTabKey;
    targetId?: string;
  }>;
};

type MissionNextAction = {
  id: string;
  label: string;
  reason: string;
  priority: "primary" | "secondary";
  disabled?: boolean;
  run?: () => void;
  targetTab?: ProjectTabKey;
  targetId?: string;
};
```

Build it from existing data:

- `health`
- `selectedProject`
- `detail.datasets`
- `detail.jobs`
- `detail.plans`
- `detail.runSummaries`
- `detail.runEvaluations`
- `detail.champion`
- `detail.decisions`
- `detail.workerRequirements`
- `detail.executionEvents`
- `detail.agentInvocations`
- `detail.agentMemory`
- `activityStreamState`
- `automationSettings`

Do not add backend semantics in the frontend. The digest should summarize known backend state and point to existing API-backed actions.

## Live Agent Activity Stream

Add a visible live-activity affordance to Overview so the UI does not look stuck while LLM and orchestration work is happening.

This should feel similar to Codex, Cursor, and GitHub Actions:

- A current activity row with a spinner or pulse when work is active.
- A short verb phrase, not a paragraph.
- A compact step list of the latest 3-5 agent/system actions.
- A link to open the full Agents or Operations drill-down.

Example Overview display:

```text
Thinking
Planner is comparing finished runs and checking whether follow-up experiments are worth scheduling.

Retrieving memories        done
Ranking candidates         active
Validating plan            waiting
```

Suggested event labels:

- `Thinking`
- `Reading project goal`
- `Reading dataset profile`
- `Retrieving memories`
- `Reviewing completed runs`
- `Ranking candidates`
- `Checking backend gates`
- `Writing decision`
- `Scheduling follow-up`
- `Starting workers`
- `Waiting for training`
- `Selecting champion`

Source data, in priority order:

1. Existing activity stream events and `visibleActivityEvents`.
2. `detail.agentInvocations`, when an invocation is active or recently completed.
3. `detail.executionEvents`, especially planner, memory, validation, worker, and champion events.
4. `detail.decisions` and `detail.agentMemory`, as fallback evidence when live events are not available.
5. Polling/SSE connection state, so the UI can say `Listening`, `Reconnecting`, or `Polling`.

Privacy and trust constraints:

- Show process breadcrumbs, not private reasoning.
- Never display raw prompts, raw completions, tool arguments, base64, storage URIs, local paths, secrets, or large JSON.
- It is acceptable to show `Thinking` as a status label, but do not reveal hidden chain-of-thought.
- Use sanitized summaries already available through existing helpers such as `activitySafeDisplayText`.

Failure behavior:

- If an LLM call fails, show a concise failure row and link to Agents.
- If backend validation rejects a proposal, show `Validation blocked` and link to the rejection detail.
- If the activity stream is reconnecting, show that Mission Control is still polling.
- If no live event exists, show `Idle` or the latest meaningful completed step.

## State Priority Rules

Use deterministic priority so the Overview does not compete with itself.

1. Orchestrator offline or request failure.
   - State: `blocked`
   - Primary action: Refresh or verify orchestrator URL.

2. No selected project.
   - State: `empty`
   - Primary action: New Project.

3. Project has no dataset.
   - State: `dataset_needed`
   - Primary action: Upload dataset.

4. Dataset exists but is not profiled.
   - State: `profiling`
   - Primary action: Open Data, or rerun visual/profile worker if stalled.

5. Profiled dataset has no plan.
   - State: `plan_ready` or `review_needed`, depending on available controls.
   - Primary action: Open Experiments.

6. Latest plan exists and no jobs have been launched.
   - State: `plan_ready`
   - Primary action: Execute Plan when safe and available.

7. Jobs are queued/running/assigned.
   - State: `training`
   - Primary action: Open Runs.
   - If queued jobs exist and workers are missing, make worker health warning visible.

8. Terminal runs exist and no latest decision exists.
   - State: `review_needed`
   - Primary action: Review Experiments.

9. Latest decision is `ADD_EXPERIMENTS` and no follow-up plan exists.
   - State: `follow_up_ready`
   - Primary action: Schedule Follow-up.

10. Champion exists.
   - State: `champion_selected`
   - Primary action: Open Export or Compare Runs.
   - Warnings can still override to `blocked` if active failures exist.

## Drill-Down Links

Overview actions should be able to navigate without duplicating content.

Implementation options:

1. Add a helper:

```ts
function openProjectTab(tab: ProjectTabKey, targetId?: string) {
  setActiveProjectTab(tab);
  if (targetId) {
    window.requestAnimationFrame(() => {
      document.getElementById(targetId)?.scrollIntoView({ block: "start" });
    });
  }
}
```

2. Use existing panel IDs where available:
   - `data`
   - `runs`
   - `agents`
   - `operations`
   - `export-demo`
   - `plans`

3. Add missing IDs for important drill-down anchors:
   - `agent-activity`
   - `automation-timeline`
   - `workers`
   - `recent-jobs`
   - `run-metrics`
   - `champion-comparison`

## Component Plan

Keep this initially in `App.tsx` unless the file becomes harder to work with. A later cleanup can extract components into `apps/mission-control/src/components`.

### New Components

- `MissionOverview`
  - Receives `digest`, handlers, loading state, and navigation callback.

- `MissionStatusPanel`
  - Renders the headline, detail, and compact facts.

- `MissionHealthStrip`
  - Renders health chips with tone and drill-down links.

- `MissionNextActions`
  - Renders one primary action and secondary actions.

- `LiveAgentActivity`
  - Renders the current high-level LLM/orchestration activity, recent sanitized steps, and stream status.
  - Uses active animations only while the status is genuinely active, waiting, connecting, or reconnecting.
  - Links to Agents or Operations for the full audit trail.

- `MissionSignals`
  - Renders up to 5 important state changes.

- `ChampionOutcomeSummary`
  - Renders compact champion outcome and links to Export or Experiments.

### Components To Shrink Or Move

- `MetricCard`
  - Keep for drill-down sections if useful.
  - Remove generic metric-card grid from Overview.

- `Experiment Timeline`
  - Move out of default Overview.
  - Keep a compact Recent Signals list on Overview.
  - Full lifecycle timeline can live in Operations or Experiments.

- `AgentActivityPanel`
  - Keep current detailed activity in Agents or Operations.
  - Overview should use `LiveAgentActivity` plus a condensed current-state signal from `agentActivityCurrentState`.
  - The detailed panel should remain the place for metadata, event history, and audit context.

- `Selected Champion`
  - Replace the detailed Overview panel with `ChampionOutcomeSummary`.
  - Keep the detailed comparison and export/demo experiences in Experiments and Export.

## Visual Direction

The UI should feel like an operator cockpit, not a reporting dashboard.

Use:

- Dense but calm rows.
- Small status chips.
- Plain language.
- Clear severity colors.
- Icon buttons where the action is familiar.
- One highlighted primary action.

Avoid:

- Large metric-card grids on Overview.
- Multiple full-width panels fighting for attention.
- Long explanatory text.
- Decorative visuals.
- Repeated headings that restate the tab name.

The current dark green palette can remain, but the Overview should rely less on many bordered cards and more on compact rows grouped by intent.

## Implementation Phases

### Phase 1: Build The Digest

Files:

- `apps/mission-control/src/App.tsx`

Tasks:

1. Add `MissionDigest`, `MissionHealthItem`, `MissionLiveActivity`, `MissionNextAction`, and `MissionSignal` types.
2. Add `buildMissionDigest(...)`.
3. Add `buildMissionLiveActivity(...)` to derive high-level steps from live events, invocations, decisions, memory, and stream state.
4. Reuse existing helpers:
   - `normalizedStatus`
   - `jobStatusCounts`
   - `summarizeTrainingRuns`
   - `agentActivityCurrentState`
   - `activitySafeDisplayText`
   - `decisionRejections`
   - `backendGateSummary`
   - `formatMaybeMetric`
   - `formatCurrency`
   - `formatSeconds`
5. Add deterministic priority rules.
6. Keep the existing Overview unchanged until the digest is ready.

Acceptance:

- Digest can describe empty, profiling, plan-ready, training, review-needed, blocked, and champion-selected states.
- Digest includes a live activity state that can render active, waiting, idle, blocked, failed, and succeeded agent/system work.
- No new backend calls.
- No orchestration behavior changes.

### Phase 2: Replace The Overview

Files:

- `apps/mission-control/src/App.tsx`
- `apps/mission-control/src/styles.css`

Tasks:

1. Replace the `summary-grid` on Overview with `MissionOverview`.
2. Replace the full `Experiment Timeline` Overview panel with compact `RecentSignals`.
3. Move or retag the full `Experiment Timeline` to a drill-down tab if it remains useful.
4. Add `LiveAgentActivity` near the top of Overview so long-running LLM/planner work visibly progresses.
5. Replace the detailed `Agent Activity` Overview panel with current-state summary plus "Open Agents" or "Open Operations".
6. Replace the detailed `Selected Champion` Overview panel with compact champion outcome.
7. Ensure the first viewport at 1120x720 answers all three questions without scrolling.

Acceptance:

- Overview contains no raw IDs, storage URIs, JSON, full tables, full charts, or full agent payloads.
- Overview has one primary action.
- Overview shows concise live LLM/orchestration activity when an agent, validation, memory retrieval, or worker handoff is active.
- Every removed detail has a visible drill-down path.

### Phase 3: Organize Drill-Down Tabs

Files:

- `apps/mission-control/src/App.tsx`
- `apps/mission-control/src/styles.css`

Tasks:

1. Add or refine anchor IDs for major panels.
2. Make Overview actions navigate to target tabs and anchors.
3. Confirm Data contains all dataset intelligence and visual analysis.
4. Confirm Experiments contains plan, runs, jobs, metrics, and comparison.
5. Confirm Agents contains all reasoning, telemetry, audit, memory, and rejection detail.
6. Confirm Operations contains settings, workers, worker requirements, timeline, and manual queue.
7. Confirm Export contains the champion export/demo/test bench and feedback flow.

Acceptance:

- Any fact summarized on Overview can be expanded in one click.
- Existing operator controls are still available.
- Manual debug surfaces remain outside the default view.

### Phase 4: Copy And Empty States

Files:

- `apps/mission-control/src/App.tsx`

Tasks:

1. Rewrite Overview copy into short operational sentences.
2. Add state-specific empty messages:
   - No project selected.
   - Dataset upload needed.
   - Profiling in progress.
   - Plan waiting.
   - Runs in progress.
   - Review needed.
   - Champion ready.
3. Make next-action reasons concrete.
4. Keep technical labels in drill-downs where they are useful.

Acceptance:

- Overview can be understood without knowing the backend model.
- Drill-downs preserve technical precision.

### Phase 5: Responsive And Accessibility Pass

Files:

- `apps/mission-control/src/styles.css`

Tasks:

1. Define stable dimensions for status rows, health chips, and action rows.
2. Ensure action text wraps cleanly.
3. Verify no Overview text overlaps on the app minimum size and common desktop sizes.
4. Preserve keyboard access for tabs and action buttons.
5. Make health tones readable without relying only on color.

Acceptance:

- No horizontal overflow from Overview content.
- The primary action is reachable by keyboard.
- Health and warning states remain understandable with icons/text.

### Phase 6: Verification

Commands:

```powershell
cd apps/mission-control
npm run build
```

Manual smoke scenarios:

1. No project selected.
2. Project created, no dataset.
3. Dataset uploaded, profiling not complete.
4. Profiled dataset, no plan.
5. Plan ready, auto execution off.
6. Jobs queued with no active workers.
7. Jobs running.
8. Failed job or backend rejection.
9. Completed runs, review needed.
10. Follow-up proposed.
11. Champion selected.
12. Orchestrator offline or SSE fallback.
13. LLM review active: Overview shows `Thinking`, `Reviewing completed runs`, or equivalent active status.
14. Memory retrieval active or recently completed: Overview shows `Retrieving memories` without raw payloads.
15. Backend validation active or blocked: Overview shows `Validating plan` or `Validation blocked` with a drill-down link.

Visual checks:

- First viewport answers the three homepage questions.
- Live LLM activity makes long-running agent work feel active rather than stuck.
- Details are available through tabs and anchors.
- Overview has no dense tables, raw payloads, or long lists.
- The page remains readable at the current app minimum size.

## Suggested PR Breakdown

### PR 1: Mission Digest

Add derived digest types, `buildMissionDigest`, and `buildMissionLiveActivity`, but keep UI mostly unchanged.

Why:

- Low visual risk.
- Lets us validate state priority rules before changing layout.

### PR 2: Overview Redesign

Replace Overview with the new status, health, next-action, signal, and champion-summary components.

Why:

- Delivers the core user-facing improvement.
- Keeps drill-down content intact.

### PR 3: Drill-Down Navigation

Add anchor IDs, one-click Overview links, and tab organization cleanup.

Why:

- Ensures removed default information is still easy to find.

### PR 4: Polish And Regression Pass

Tune copy, CSS, responsive behavior, build verification, and manual smoke scenarios.

Why:

- Makes the redesign feel intentional rather than just hidden.

## Acceptance Criteria

The redesign is complete when:

1. The selected-project Overview answers "What is happening?", "Is it healthy?", and "What should I do next?" in the first viewport.
2. The Overview has one primary action and no more than three secondary actions.
3. Generic inventory metrics are not shown by default unless they explain health or action.
4. Raw IDs, storage URIs, JSON, full tables, full charts, long timelines, and detailed agent reasoning are hidden from the default screen.
5. Live LLM/orchestration activity is visible on Overview during active work, using concise labels such as `Thinking`, `Retrieving memories`, `Ranking candidates`, `Validating plan`, or `Waiting for training`.
6. Live activity never exposes raw prompts, raw completions, hidden chain-of-thought, secrets, paths, storage URIs, base64, large JSON, or full tool payloads.
7. Every hidden detail remains accessible in a relevant drill-down tab.
8. Failed, blocked, unhealthy, and rejected states remain visible on Overview as summaries.
9. Existing controls still work: create project, execute plan, review experiments, schedule follow-up, automation settings, manual queue, worker views, export/demo, and champion feedback.
10. `npm run build` succeeds in `apps/mission-control`.

## Non-Goals

- Do not change orchestrator semantics.
- Do not remove auditability.
- Do not introduce a new backend overview endpoint unless the existing client-side digest proves insufficient.
- Do not redesign the new-project upload flow beyond adding clearer next-action paths.
- Do not delete the manual JSON queue in this pass.
- Do not reduce polling or SSE behavior as part of the first UI pass.

## Risks

1. Hiding too much can make expert users feel slower.
   - Mitigation: one-click drill-down links and stable tab anchors.

2. Derived frontend summaries can drift from backend semantics.
   - Mitigation: summarize only existing backend states and keep actions routed through existing APIs.

3. A single primary action can be ambiguous in edge cases.
   - Mitigation: deterministic priority rules and secondary actions for common alternatives.

4. Dense drill-down tabs can remain overwhelming.
   - Mitigation: this plan starts with the default screen, then organizes anchors and panel order. A later pass can split large tabs into subviews if needed.

## Future Follow-Up

If this redesign works, consider a later backend endpoint such as `GET /projects/:id/mission-summary` that returns a normalized status digest. That would reduce defensive frontend derivation and make Mission Control, CLI, and possible web views share the same operational summary.
