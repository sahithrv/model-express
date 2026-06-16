# Final UI Changed

Date: June 16, 2026

This is primarily the visual-only frontend polish plan for small Codex-5.3-Spark PRs. One exception is PR UI-0, a tiny frontend display-state bug fix for the mission board showing "Refinement" after a champion has already been selected and exported. It intentionally excludes export download behavior, project history fetch semantics, backend status changes, and any new API contracts. Those belong in `docs/plans/final_v1_updates_for_release.md`.

## Rules For Every UI PR

- Keep each PR small enough for one Spark pass.
- Prefer one file, usually `apps/mission-control/src/styles.css`.
- Do not touch backend, worker, store, migrations, or Electron IPC.
- Do not add dependencies.
- Do not alter polling, export, run, or state-machine behavior.
- Exception: PR UI-0 may adjust frontend-only mission stage derivation so completed champion/export state is represented honestly.
- Do not introduce new data requirements.
- Use existing React structure, existing CSS, and existing `lucide-react` icons.
- Run `npm run build` from `apps/mission-control`.
- Verify at 1320x860 and a narrow desktop width around 1120x720.

## Design Brief

- Product context: Mission Control is a dense local desktop operator UI for dataset import, experiment runs, agent decisions, champion selection, export, and demo testing.
- User goal: understand the current mission state quickly, then act on the next useful control without hunting.
- Current UI problem: the app has strong product-specific structure, but some dense areas read visually similar, long export/status text can compete with actions, old-project/export recovery is not visually emphasized enough, and the mission board can still show "Refinement" as the current step after champion selection and export are done.
- Desired feel: quiet command center, technical, trustworthy, demo-ready, with sharper hierarchy and less same-weight card noise.
- Stack constraints: Vite, React, TypeScript, Electron, plain CSS, `lucide-react`, no UI framework.

## Visual Directions

### 1. Safe Polished Version

- Layout approach: keep current shell and tabs, tighten spacing, improve hierarchy, strengthen active states.
- Navigation style: clearer active mission row and workflow tabs.
- Component style: restrained borders, better text truncation, stronger status/action separation.
- Color/spacing/typography ideas: keep the dark green base, add tiny amber/cyan accents for warning/info states only.
- Motion/interaction ideas: subtle hover/focus polish only.
- Why this fits: lowest risk and best for a release polish pass.
- Risks: can feel incremental if each PR is too timid.

### 2. Distinctive Version

- Layout approach: make mission state feel more like a cockpit with stronger section bands and denser telemetry.
- Navigation style: add visual "mission rail" treatment to the sidebar and tabs.
- Component style: stronger contrast and more pronounced status badges.
- Color/spacing/typography ideas: more cyan/amber status language, less all-green sameness.
- Motion/interaction ideas: selected row transitions and chart hover polish.
- Why this fits: more memorable screenshots.
- Risks: higher chance of visual churn and unintended overlap.

### 3. Experimental Version

- Layout approach: split mission command into a more theatrical live-ops dashboard.
- Navigation style: bolder tab rail and larger current-state module.
- Component style: heavier telemetry, stronger visual separators.
- Color/spacing/typography ideas: expanded status palette.
- Motion/interaction ideas: animated status indicators.
- Why this fits: demo screenshots could look distinctive.
- Risks: not appropriate for a small v1 release polish pass.

## Recommended Direction

Use the Safe Polished Version with elements of the Distinctive Version and Experimental Version if the added implementation cost is not too large. It fits the existing codebase, limits behavior risk, and gives Spark-sized PRs clear boundaries.

## PR UI-0: Mission Flow Completion State Fix

Spark size: small

Files:

- `apps/mission-control/src/features/mission/projectDetailModel.tsx`
- `apps/mission-control/src/features/mission/ProjectRoutePanels.tsx` only if `currentMissionStage` needs a defensive fallback

Goal:

- When a champion model has been selected and the export package is ready, the mission board should not keep saying the project is under "Refinement". It should make the completed handoff state obvious.

Observed bug:

- The screenshot state has "Champion Selected", "Export package prepared", Export marked done, and Demo Validation marked done, but the top board still says "Now: Refinement" and the Refinement tile is "WAITING".
- `buildMissionStages` marks Refinement as done only when a follow-up plan exists.
- If the AI chose a champion without creating a follow-up refinement plan, Refinement remains waiting even though later stages are done.
- `currentMissionStage` then picks the first waiting stage, so an obsolete optional step becomes the current project state.

Allowed changes:

- Update `buildMissionStages` so the workflow is monotonic:
  - If `detail.champion` exists, Refinement should no longer be waiting for this release path.
  - If `exportReady` or demo validation is true, Refinement, Champion Selection, and Export must not appear behind the current state.
  - Keep Refinement active or blocked only for a real `ADD_EXPERIMENTS` decision that still needs a follow-up plan and has not selected a champion.
- Use direct, user-facing completion copy such as:
  - `Champion selected; no further refinement is required for this handoff.`
  - `Export package ready for use.`
  - `Mission complete` or `Ready for use` when every stage is done.
- If needed, adjust `currentMissionStage` so waiting stages that appear before the latest completed stage cannot become the "Now" state.
- Keep the change frontend-only. Do not change backend statuses, export records, job state names, or polling.

Do not:

- Reorder the nine mission stages.
- Add a tenth stage.
- Change export readiness logic beyond reusing existing `readyONNXExport(exportDemo.exports)`.
- Implement Save ZIP in this PR.
- Hide real blockers. If a follow-up refinement decision is actually blocking champion selection, it should still show as blocked/active.

Acceptance:

- With champion selected and ready export present, the Refinement tile is not `WAITING`.
- With champion selected, ready export present, and demo validation succeeded, the header does not say "Refinement"; it communicates that the handoff/demo path is done.
- The progress counter reaches `9/9 steps complete` when all nine visible stages are complete.
- A project that genuinely still needs follow-up experiments can still show Refinement as active or blocked.
- `npm run build` passes from `apps/mission-control`.

Suggested test:

- Add or update a small model test for `buildMissionStages`:
  - champion selected with ready export marks Refinement done,
  - champion selected with ready export and demo prediction returns no waiting stage before the latest done stage,
  - ADD_EXPERIMENTS without champion still keeps Refinement active/waiting.

## PR UI-1: Sidebar Mission List Polish

Spark size: small

Files:

- `apps/mission-control/src/styles.css`

Goal:

- Make old and active missions easier to scan without changing project loading behavior.

Allowed changes:

- Improve `.project-list`, `.project`, `.project.active`, `.project-copy`, `.project-state`, and `.project-status-dot`.
- Tighten row height and gap consistency.
- Improve truncation for long project names/goals.
- Make active row contrast clear without adding a new color theme.
- Keep status pill width stable so rows do not jitter.

Do not:

- Change `refreshProjects`.
- Change project selection logic.
- Add search/filter.
- Add new API calls.

Acceptance:

- Long project names truncate cleanly.
- Active mission is obvious within 2 seconds.
- Rows still fit at 1120px app width.

## PR UI-2: Topbar And Engine Chip Hierarchy

Spark size: small

Files:

- `apps/mission-control/src/styles.css`

Goal:

- Make the selected mission title, state pill, and engine health easier to read in screenshots and during a demo.

Allowed changes:

- Adjust `.topbar`, `.topbar-copy`, `.topbar-kicker`, `.mission-state-pill`, `.engine-chip`, and `.topbar-actions`.
- Keep title text from colliding with action buttons.
- Improve disabled/offline engine contrast.
- Add a small responsive wrap rule for narrower desktop widths.

Do not:

- Change topbar content.
- Change health polling.
- Change actions.

Acceptance:

- Long mission goal text does not overlap actions.
- Engine ready/offline state is readable.
- The state pill is visible but not louder than the title.

## PR UI-3: Workflow Tabs Rhythm And Focus States

Spark size: small

Files:

- `apps/mission-control/src/styles.css`

Goal:

- Make section tabs feel like a controlled workflow rather than a row of same-weight cards.

Allowed changes:

- Refine `.section-tabs`, `.section-tabs button`, `.workflow-tab-index`, `.workflow-tab-main`, `.workflow-tab-icon`, and `.workflow-tab-dot`.
- Stabilize tab heights.
- Improve hover and focus-visible states.
- Make done/active/blocked states easier to distinguish using existing status classes.

Do not:

- Change `workflowTabs.tsx`.
- Reorder tabs.
- Change tab state calculations.

Acceptance:

- Keyboard focus is visible.
- Active tab is clearly selected.
- Text does not wrap awkwardly at 1120px.

## PR UI-4: Export/Demo Panel Visual Polish

Spark size: small

Files:

- `apps/mission-control/src/styles.css`

Goal:

- Make export status, export records, portable bundle metadata, and demo controls clearer without implementing download behavior.

Allowed changes:

- Refine `.export-demo-panel`, `.export-demo-grid`, `.export-block`, `.export-status-line`, `.export-record`, `.export-record-list`, and `.portable-bundle-record`.
- Make portable bundle rows visually distinct but still restrained.
- Improve wrapping/truncation for long `file://` and `s3://` URIs.
- Keep the Request ONNX button visually secondary to a future Save ZIP button.

Do not:

- Add the Save ZIP button in this UI-only PR.
- Change `buildChampionExportDemo`.
- Change export status logic.

Acceptance:

- Long artifact URIs do not overflow.
- Failed/pending/ready export rows are visually distinct.
- Portable bundle is easy to spot when present.

## PR UI-5: Activity And Results Card Density Pass

Spark size: small

Files:

- `apps/mission-control/src/styles.css`

Goal:

- Reduce same-weight card noise in the mission activity and results areas.

Allowed changes:

- Refine `.activity-card-list`, `.activity-card`, `.activity-card-icon`, `.activity-card-body`, `.activity-card-facts`, `.results-primary-metric`, `.candidate-list`, and related empty states.
- Improve vertical rhythm and text hierarchy.
- Make failed/blocker cards readable without making the whole dashboard feel alarmed.

Do not:

- Change activity event generation.
- Change fallback activity logic.
- Change candidate ranking display data.

Acceptance:

- Activity cards scan cleanly.
- Failure cards are noticeable but not visually overwhelming.
- Empty states still look intentional.

## PR UI-6: Metric Chart And Run Panel Polish

Spark size: small

Files:

- `apps/mission-control/src/styles.css`

Goal:

- Make run metric charts and metric tabs more readable during a demo.

Allowed changes:

- Refine `.metric-area`, `.metric-toolbar`, `.metric-tabs`, `.metric-tab`, `.metric-inline-stats`, `.chart-wrap`, `.chart-stat`, `.metric-chart`, `.chart-label`, and `.chart-tooltip`.
- Improve tab selected/hover/focus states.
- Improve chart empty state and tooltip readability.
- Keep chart dimensions stable.

Do not:

- Change metric calculations.
- Change selected metric behavior.
- Change SVG chart generation.

Acceptance:

- Metric tabs do not shift when selected.
- Tooltip text fits.
- Empty chart state is readable.

## PR UI-7: Empty States And First-Run Visual Finish

Spark size: small

Files:

- `apps/mission-control/src/styles.css`

Goal:

- Make first-run/no-data areas feel finished enough for a demo video.

Allowed changes:

- Refine `.mission-empty-state`, `.mission-empty-hero`, `.mission-empty-copy`, `.mission-empty-panel`, `.activity-empty-state`, `.results-empty-state`, and `.export-waiting-state`.
- Reduce decorative heaviness if it competes with controls.
- Improve alignment, spacing, and responsive behavior.

Do not:

- Change empty-state copy.
- Add feature explanations.
- Add images or generated assets.

Acceptance:

- No empty-state text overlaps at 1120x720.
- Empty screens point attention to existing controls.
- The first screen still feels like the product, not a landing page.

## PR UI-8: Final CSS Overflow And Contrast Audit

Spark size: small

Files:

- `apps/mission-control/src/styles.css`

Goal:

- Catch visual regressions from the prior small PRs.

Allowed changes:

- Add or refine responsive rules near existing media queries.
- Fix text overflow in buttons, badges, project rows, export rows, activity cards, and topbar actions.
- Adjust contrast only where text is hard to read.

Do not:

- Re-theme the app.
- Touch TypeScript unless a class name is demonstrably wrong.

Acceptance:

- No visible text overlap at 1320x860 or 1120x720.
- Buttons keep stable dimensions.
- `npm run build` passes.

## Final Visual Verification

After the UI PRs land:

- Run `npm run build` from `apps/mission-control`.
- Launch Mission Control.
- Check:
  - no project state,
  - active project with running jobs,
  - failed job state,
  - selected champion with export records,
  - export waiting state,
  - narrow desktop width.

Keep screenshots for the release/demo notes if the UI looks stable.
