# Agents Context

This folder contains compact context packs for future Model Express subagents. Use these files to reduce prompt bloat: instead of restating the entire repo state, point each subagent at the relevant context file plus any task-specific docs.

## How To Use

For a future multi-agent task:

1. Start with `docs/model_express_agentic_upgrade_roadmap.md`.
2. Give each subagent its matching `docs/agents_context/.../context.md`.
3. Add only the supporting docs needed for that task.
4. Assign clear write scopes and PR boundaries.
5. Have the Integration Coordinator check schema/API/worker/frontend/LLM mismatches before finalizing.

## Context Packs

- Frontend / Mission Control Agent:
  - `docs/agents_context/frontend_mission_control/context.md`
- Orchestrator / Backend Agent:
  - `docs/agents_context/orchestrator_backend/context.md`
- Python Worker / Training Agent:
  - `docs/agents_context/python_worker_training/context.md`
- Data / Dataset Intelligence Agent:
  - `docs/agents_context/data_dataset_intelligence/context.md`
- LLM Decision Intelligence Agent:
  - `docs/agents_context/llm_decision_intelligence/context.md`
- System Architecture Review Agent:
  - `docs/agents_context/system_architecture_review/context.md`
- Integration Coordinator Agent:
  - `docs/agents_context/integration_coordinator/context.md`

## Update Policy

Update the relevant context pack when a PR changes that agent's durable responsibilities, contracts, or known gaps.

Keep each file concise. These are prompt seeds, not full implementation reports. Detailed history belongs in the reports and roadmap under `docs/`.
