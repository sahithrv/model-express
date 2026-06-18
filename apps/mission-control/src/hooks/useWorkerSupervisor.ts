import { useCallback, useEffect, useRef } from "react";

import type { RequestOptions } from "../api/missionControlClient";

type SupervisorJob = {
  status: string;
  config?: Record<string, unknown>;
};

type SupervisorWorkerRequirement = {
  id: string;
  project_id: string;
  plan_id?: string;
  status: string;
  provider?: string;
  gpu_type?: string;
  target_count: number;
};

type RefreshProjectDetail = (
  projectId: string,
  options?: { includeSlowData?: boolean; forceSlowData?: boolean },
) => Promise<void>;

type CloudPreflightStage = "dataset_upload" | "plan_execution" | "worker_start" | "manual";

type WorkerSupervisorOptions = {
  baseUrl: string;
  selectedProjectId: string;
  workerRequirements: SupervisorWorkerRequirement[];
  jobs: SupervisorJob[];
  request: <T>(path: string, options?: RequestOptions) => Promise<T>;
  preflightCloud?: (stage: CloudPreflightStage) => Promise<unknown>;
  refreshProjectDetail: RefreshProjectDetail;
};

export function useWorkerSupervisor({
  baseUrl,
  selectedProjectId,
  workerRequirements,
  jobs,
  request,
  preflightCloud,
  refreshProjectDetail,
}: WorkerSupervisorOptions) {
  const workerRequirementsRef = useRef<SupervisorWorkerRequirement[]>([]);
  const jobsRef = useRef<SupervisorJob[]>([]);
  const supervisingRequirements = useRef<Set<string>>(new Set());
  const activeRequirementEnsureAt = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    workerRequirementsRef.current = workerRequirements;
  }, [workerRequirements]);

  useEffect(() => {
    jobsRef.current = jobs;
  }, [jobs]);

  const resetWorkerSupervisor = useCallback(() => {
    activeRequirementEnsureAt.current.clear();
    supervisingRequirements.current.clear();
  }, []);

  const superviseWorkerRequirements = useCallback(async () => {
    if (!selectedProjectId) return;

    const actionable = workerRequirementsRef.current.filter((requirement) =>
      (requirement.status === "PENDING" || requirement.status === "STARTING" || requirement.status === "ACTIVE") &&
      workerRequirementHasOpenWork(requirement, jobsRef.current),
    );
    const now = Date.now();

    for (const requirement of actionable) {
      if (supervisingRequirements.current.has(requirement.id)) {
        continue;
      }
      const alreadyActive = requirement.status === "ACTIVE";
      const lastEnsureAt = activeRequirementEnsureAt.current.get(requirement.id) ?? 0;
      if (alreadyActive && now - lastEnsureAt < 30_000) {
        continue;
      }
      activeRequirementEnsureAt.current.set(requirement.id, now);
      supervisingRequirements.current.add(requirement.id);
      try {
        if (requirement.provider === "modal") {
          await preflightCloud?.("worker_start");
        }
        if (!alreadyActive) {
          await request<SupervisorWorkerRequirement>(`/worker-requirements/${requirement.id}`, {
            method: "PATCH",
            body: { status: "STARTING", last_error: "" },
          });
        }

        await window.missionControl.ensureProjectWorker({
          projectId: requirement.project_id,
          baseUrl,
          name: `auto-worker-${requirement.project_id}`,
          gpuType: requirement.provider === "modal" ? "modal" : requirement.gpu_type || requirement.provider || "local",
          count: requirement.target_count,
        });

        if (!alreadyActive) {
          await request<SupervisorWorkerRequirement>(`/worker-requirements/${requirement.id}`, {
            method: "PATCH",
            body: { status: "ACTIVE", last_error: "" },
          });
        }
      } catch (error) {
        await request<SupervisorWorkerRequirement>(`/worker-requirements/${requirement.id}`, {
          method: "PATCH",
          body: {
            status: "FAILED",
            last_error: error instanceof Error ? error.message : String(error),
          },
        }).catch(() => undefined);
      } finally {
        supervisingRequirements.current.delete(requirement.id);
        refreshProjectDetail(selectedProjectId, { includeSlowData: false }).catch(() => undefined);
      }
    }
  }, [baseUrl, preflightCloud, refreshProjectDetail, request, selectedProjectId]);

  return { resetWorkerSupervisor, superviseWorkerRequirements };
}

function workerRequirementHasOpenWork(requirement: SupervisorWorkerRequirement, jobs: SupervisorJob[]) {
  return jobs.some((job) => {
    const status = normalizedStatus(job.status);
    if (!["QUEUED", "ASSIGNED", "RUNNING"].includes(status)) return false;
    const planId = recordString(job.config, "plan_id");
    if (requirement.plan_id && planId !== requirement.plan_id) return false;
    const provider = recordString(job.config, "provider");
    if (requirement.provider && provider && provider !== requirement.provider) return false;
    return true;
  });
}

function normalizedStatus(status: string) {
  return String(status || "").trim().toUpperCase();
}

function recordString(record: Record<string, unknown> | undefined, key: string) {
  const value = record?.[key];
  return typeof value === "string" ? value : "";
}
