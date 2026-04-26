# Distributed Agentic Vision Training Platform for Gaming AI

## Project Summary

This project is a distributed, agentic computer vision training platform designed to produce strong image-based classifiers for a Gaming AI application.

The system should help turn labeled gameplay screenshots into usable vision models by:

- profiling gaming image datasets
- planning model experiments with specialized agents
- dispatching training jobs across remote GPU workers
- tracking runs, metrics, and artifacts with MLflow
- pruning bad experiments early using mid-training statistics
- adapting future experiments based on shared results
- selecting the best model
- exporting optimized models for low-latency local inference in the Gaming AI app

---

## 🔥 Critical Architecture Clarification

This system uses **externally provisioned GPU workers**, NOT orchestrator-managed infrastructure.

The orchestrator:
- does NOT create/destroy cloud machines
- DOES manage jobs across already-running workers

Core focus:

distributed experiment orchestration + agent-guided optimization

NOT cloud infrastructure provisioning

---

## Key Design Philosophy

Agents reason. Orchestrator controls. Workers execute. MLflow tracks. Memory informs.

---

## High-Level System Flow

1. User submits dataset + goal
2. Dataset profiler computes stats
3. Profiler Agent analyzes risks
4. Planner Agent generates experiment configs
5. Orchestrator validates + queues jobs
6. Remote workers pull jobs
7. Workers train models + stream metrics
8. Orchestrator prunes weak runs
9. Workers get reassigned new jobs
10. Diagnostics Agent analyzes results
11. Planner proposes next iteration
12. Loop continues until stopping condition
13. Best models exported for Gaming AI

---

## Remote Worker Model

### Worker Lifecycle

1. GPU worker is manually started (RunPod, etc.)
2. Worker registers with orchestrator
3. Worker polls for jobs
4. Worker trains + streams metrics
5. Worker uploads artifacts
6. Worker becomes idle

---

## Orchestrator Responsibilities

- job creation
- scheduling
- worker assignment
- metric monitoring
- pruning decisions
- job cancellation
- reassignment
- experiment iteration
- memory updates
- model selection

NOT responsible for:
- provisioning GPUs
- cloud infra
- CUDA setup

---

## Worker Allocation Strategy

User selects:

- Cheap
- Balanced
- Fast

System decides:

- number of workers to use (from available pool)
- number of experiments
- pruning aggressiveness

Example:

Cheap → 2 workers  
Balanced → 4 workers  
Fast → 6–8 workers  

---

## Parallel Execution Model

- experiments queued
- workers pull jobs
- models train in parallel
- metrics streamed
- bad runs pruned early
- workers reassigned new tasks

---

## Pruning Loop

1. worker trains
2. metrics sent to orchestrator
3. orchestrator evaluates:
   - plateau
   - overfitting
4. decision:
   - continue
   - prune
5. if pruned → worker reassigned

---

## Agent Control Model

Agents:
- recommend

Orchestrator:
- validates + enforces

Workers:
- execute

Agents never directly control execution.

---

## Memory Layer

Stores structured insights:

- dataset fingerprints
- run summaries
- pruning decisions
- performance trends

Used to improve future experiments.

---

## Distributed System Clarification

This is:

distributed experiment search

NOT:

multi-GPU training of one model

---

## Compute Strategy

- Use RunPod / similar
- Manually start 2–8 GPU workers
- Each worker runs 1 job

Recommended GPUs:
- RTX 4090
- RTX 3090
- L4

---

## System Constraints

- max_workers
- max_experiments
- max_runtime
- allowed templates

Agents cannot override these.

---

## Final Principle

Agents reason.
Orchestrator controls.
Workers execute.
MLflow tracks.
Memory improves the system.
