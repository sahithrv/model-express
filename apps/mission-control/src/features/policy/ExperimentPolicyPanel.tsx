import { useEffect, useMemo, useState } from "react";
import type { Dataset, Job } from "../../types";
import { MODEL_EXPRESS_CATALOG } from "../../modelExpressCatalog.generated";
import { COMPATIBILITY_PROFILES } from "../../compatibilityProfiles.generated";
import { OrchestratorHttpError, type RequestOptions } from "../../api/missionControlClient";

export type ExperimentPolicyScope = "account" | "project" | "dataset" | "run";

type Request = <T>(path: string, options?: RequestOptions) => Promise<T>;

type PolicyRule = {
  id: string;
  effect: "deny";
  selector: { kind: "catalog_ids"; catalog: string; ids: string[] };
};

type PolicyDocument = {
  schema_version: "model_express_experiment_policy.v1";
  profile_refs: { id: string; version: string }[];
  rules: PolicyRule[];
  metadata: { display_name: string; description: string };
};

type PolicyBinding = { revision: number; created_at: string; created_by: string; scope: string };
type PolicyVersion = { document: PolicyDocument; document_hash: string; revision: number; created_at: string };
type PolicySource = { scope: string; subject_id: string; policy_hash: string; binding_revision: number };
type EffectivePolicy = {
  decision: string;
  effective_policy_hash: string;
  blocked_dimensions?: string[];
  permitted_counts?: Record<string, number>;
  snapshot?: { policy_sources?: PolicySource[]; compatibility_profiles?: { id: string; version: string }[] };
};
type QueueImpact = { queued: number; allowed: number; pending: number; blocked: number };
type AdminResponse = {
  scope: ExperimentPolicyScope;
  subject_id: string;
  active_binding?: PolicyBinding;
  policy_version?: PolicyVersion;
  effective_policy: EffectivePolicy;
  queue_impact: QueueImpact;
};
type AuditEvaluation = {
  id: string;
  operation: string;
  decision: string;
  effective_policy_hash: string;
  reason_codes?: string[];
  created_at: string;
};

type CatalogEntry = { id: string; available: boolean };
type CatalogGroup = { id: string; label: string; entries: CatalogEntry[] };

export type ExperimentPolicyPanelProps = {
  request: Request;
  projectId: string;
  datasets: Dataset[];
  jobs: Job[];
};

const roasty = COMPATIBILITY_PROFILES.find((profile) => profile.profile_key === "roasty_v1");
const editableCatalogs = [
  ["models", "Models"],
  ["model_families", "Model families"],
  ["fine_tuning_modes", "Fine-tuning modes"],
  ["resize_strategies", "Resize strategies"],
  ["crop_strategies", "Crop strategies"],
  ["bounding_box_modes", "Bounding-box modes"],
  ["normalization_strategies", "Normalization"],
  ["augmentation_policies", "Augmentation policies"],
  ["augmentation_operations", "Augmentation operations"],
  ["optimizers", "Optimizers"],
  ["schedulers", "Schedulers"],
  ["class_balancing_strategies", "Class balancing"],
  ["sampling_strategies", "Sampling"],
  ["resolution_strategies", "Resolution strategies"],
  ["export_formats", "Artifact formats"],
  ["precisions", "Precision"],
  ["runtimes", "Deployment runtimes"],
  ["execution_providers", "Execution providers"],
  ["execution_requirements", "Execution requirements"],
  ["tasks", "Tasks"],
  ["runners", "Runners"],
] as const;

export function experimentPolicyEndpoint(scope: ExperimentPolicyScope, projectId: string, subjectId: string): string {
  switch (scope) {
    case "account": return "/settings/experiment-policy";
    case "project": return `/projects/${encodeURIComponent(projectId)}/experiment-policy`;
    case "dataset": return `/datasets/${encodeURIComponent(subjectId)}/experiment-policy`;
    case "run": return `/jobs/${encodeURIComponent(subjectId)}/experiment-policy`;
  }
}

export function buildExperimentPolicyDocument(
  useRoasty: boolean,
  deniedFormats: readonly string[],
  retainedRules: readonly PolicyRule[] = [],
): PolicyDocument {
  return buildCatalogExperimentPolicyDocument(
    useRoasty,
    deniedFormats.map((id) => ({ catalog: "export_formats", id })),
    retainedRules,
  );
}

export function buildCatalogExperimentPolicyDocument(
  useRoasty: boolean,
  deniedCapabilities: readonly { catalog: string; id: string }[],
  retainedRules: readonly PolicyRule[] = [],
): PolicyDocument {
  return {
    schema_version: "model_express_experiment_policy.v1",
    profile_refs: useRoasty && roasty ? [{ id: roasty.profile_key, version: roasty.semantic_version }] : [],
    rules: [
      ...retainedRules.filter((rule) => !rule.id.startsWith("ui_deny_")),
      ...deniedCapabilities.slice().sort((left, right) => `${left.catalog}/${left.id}`.localeCompare(`${right.catalog}/${right.id}`)).map(({ catalog, id }): PolicyRule => ({
        id: `ui_deny_${catalog}_${id}`,
        effect: "deny",
        selector: { kind: "catalog_ids", catalog, ids: [id] },
      })),
    ],
    metadata: {
      display_name: "Mission Control experiment policy",
      description: "User-configured experiment and artifact exclusions.",
    },
  };
}

function catalogGroups(): CatalogGroup[] {
  const categories = MODEL_EXPRESS_CATALOG.categories as Record<string, readonly unknown[]>;
  return editableCatalogs.flatMap(([id, label]) => {
    const entries = (categories[id] ?? []).flatMap((value): CatalogEntry[] => {
      if (!value || typeof value !== "object") return [];
      const entry = value as { id?: unknown; available?: unknown };
      return typeof entry.id === "string" && typeof entry.available === "boolean"
        ? [{ id: entry.id, available: entry.available }]
        : [];
    });
    return entries.length > 0 ? [{ id, label, entries }] : [];
  });
}

function deniedCapabilityKey(catalog: string, id: string): string {
  return `${catalog}/${id}`;
}

function roastyAllows(catalog: string, id: string): boolean {
  if (!roasty) return false;
  const rules = roasty.document.rules as readonly { selector?: { catalog?: string; ids?: readonly string[] } }[];
  return rules.some((rule) => rule.selector?.catalog === catalog && rule.selector.ids?.includes(id));
}

function shortHash(value = ""): string {
  return value ? `${value.slice(0, 18)}…${value.slice(-8)}` : "none";
}

function policyErrorMessage(error: unknown): { title: string; details: string[] } {
  if (error instanceof OrchestratorHttpError && error.policy) {
    const details = error.policy.findings.map((finding) =>
      [finding.catalog && finding.id ? `${finding.catalog}/${finding.id}` : finding.fieldPath, finding.remediation]
        .filter(Boolean).join(" — "),
    );
    if (details.length === 0 && error.policy.code === "POLICY_REVISION_CONFLICT") {
      details.push("Reload this scope to review the newer policy revision, then apply your changes again.");
    }
    return {
      title: `${error.policy.code}: policy update could not be applied`,
      details,
    };
  }
  return { title: error instanceof Error ? error.message : String(error), details: [] };
}

export function ExperimentPolicyPanel({ request, projectId, datasets, jobs }: ExperimentPolicyPanelProps) {
  const [scope, setScope] = useState<ExperimentPolicyScope>("project");
  const [subjectId, setSubjectId] = useState(projectId);
  const [response, setResponse] = useState<AdminResponse | null>(null);
  const [audit, setAudit] = useState<AuditEvaluation[]>([]);
  const [useRoasty, setUseRoasty] = useState(false);
  const [deniedCapabilities, setDeniedCapabilities] = useState<string[]>([]);
  const [retainedRules, setRetainedRules] = useState<PolicyRule[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ title: string; details: string[] } | null>(null);

  const subjects = useMemo(() => {
    if (scope === "dataset") return datasets.map((dataset) => ({ id: dataset.id, label: dataset.name || dataset.id }));
    if (scope === "run") return jobs.map((job) => ({ id: job.id, label: `${job.id} · ${job.template} · ${job.status}` }));
    return [{ id: scope === "account" ? "account_local_default" : projectId, label: scope === "account" ? "Local account" : projectId }];
  }, [datasets, jobs, projectId, scope]);

  const endpoint = experimentPolicyEndpoint(scope, projectId, subjectId);

  useEffect(() => {
    const next = scope === "account" ? "account_local_default" : scope === "project" ? projectId : subjects[0]?.id ?? "";
    setSubjectId(next);
  }, [projectId, scope, subjects]);

  useEffect(() => {
    if (!subjectId || (scope !== "account" && !projectId)) return;
    let current = true;
    setBusy(true);
    setError(null);
    Promise.all([
      request<AdminResponse>(endpoint, { bypassCache: true }),
      projectId
        ? request<{ evaluations: AuditEvaluation[] }>(`/projects/${encodeURIComponent(projectId)}/experiment-policy/audit`, { bypassCache: true })
        : Promise.resolve({ evaluations: [] }),
    ]).then(([loaded, auditResponse]) => {
      if (!current) return;
      setResponse(loaded);
      setAudit(auditResponse.evaluations.slice(-10).reverse());
      const document = loaded.policy_version?.document;
      setUseRoasty(Boolean(document?.profile_refs.some((ref) => ref.id === "roasty_v1" && ref.version === "1.0.0")));
      const rules = document?.rules ?? [];
      setRetainedRules(rules);
      setDeniedCapabilities(rules.flatMap((rule) =>
        rule.id.startsWith("ui_deny_") && rule.selector.kind === "catalog_ids"
          ? rule.selector.ids.map((id) => deniedCapabilityKey(rule.selector.catalog, id))
          : [],
      ));
    }).catch((reason) => current && setError(policyErrorMessage(reason))).finally(() => current && setBusy(false));
    return () => { current = false; };
  }, [endpoint, projectId, request, scope, subjectId]);

  async function savePolicy() {
    setBusy(true);
    setError(null);
    try {
      const loaded = await request<AdminResponse>(endpoint, {
        method: "PUT",
        body: {
          document: buildCatalogExperimentPolicyDocument(
            useRoasty,
            deniedCapabilities.flatMap((value) => {
              const separator = value.indexOf("/");
              return separator > 0 ? [{ catalog: value.slice(0, separator), id: value.slice(separator + 1) }] : [];
            }),
            retainedRules,
          ),
          expected_revision: response?.active_binding?.revision ?? 0,
        },
      });
      setResponse(loaded);
    } catch (reason) {
      setError(policyErrorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  async function clearPolicy() {
    const revision = response?.active_binding?.revision;
    if (!revision) return;
    setBusy(true);
    setError(null);
    try {
      const loaded = await request<AdminResponse>(`${endpoint}?expected_revision=${revision}`, { method: "DELETE" });
      setResponse(loaded);
      setUseRoasty(false);
      setDeniedCapabilities([]);
      setRetainedRules([]);
    } catch (reason) {
      setError(policyErrorMessage(reason));
    } finally {
      setBusy(false);
    }
  }

  const effective = response?.effective_policy;
  const queue = response?.queue_impact ?? { queued: 0, allowed: 0, pending: 0, blocked: 0 };

  return (
    <section className="policy-panel" aria-label="Experiment policy">
      <div className="policy-panel-heading">
        <div>
          <div className="eyebrow">Artifact and execution governance</div>
          <h3>Experiment Policy</h3>
          <p>Policies inherit account → project → dataset → run. More specific scopes can only remove capabilities.</p>
        </div>
        <span className={`policy-decision ${effective?.decision === "allowed" ? "allowed" : "blocked"}`}>
          {effective?.decision ?? "loading"}
        </span>
      </div>

      <div className="policy-controls">
        <label><span>Scope</span><select value={scope} onChange={(event) => setScope(event.currentTarget.value as ExperimentPolicyScope)}>
          <option value="account">Account</option><option value="project">Project</option>
          <option value="dataset">Dataset</option><option value="run">Run</option>
        </select></label>
        <label><span>Subject</span><select value={subjectId} disabled={subjects.length < 2} onChange={(event) => setSubjectId(event.currentTarget.value)}>
          {subjects.map((subject) => <option key={subject.id} value={subject.id}>{subject.label}</option>)}
        </select></label>
      </div>

      <div className="policy-editor-grid">
        <div className="policy-card">
          <strong>Compatibility profile</strong>
          <label className="policy-check"><input type="checkbox" checked={useRoasty} onChange={(event) => setUseRoasty(event.currentTarget.checked)} />
            <span>Roasty v1 <small>FP32 · image classification · ONNX Runtime CPU</small></span>
          </label>
          <strong>Catalog exclusions</strong>
          <p>Clear a capability to add a deny rule at this scope. Inherited and profile restrictions cannot be re-enabled here.</p>
          <div className="policy-catalog-groups">
            {catalogGroups().map((group) => <details key={group.id} open={group.id === "models" || group.id === "export_formats"}>
              <summary>{group.label}<small>{group.entries.length} options</small></summary>
              {group.entries.map((entry) => {
                const key = deniedCapabilityKey(group.id, entry.id);
                const excludedByRoasty = useRoasty && !roastyAllows(group.id, entry.id);
                return <label className="policy-check" key={key}>
                  <input
                    type="checkbox"
                    checked={entry.available && !excludedByRoasty && !deniedCapabilities.includes(key)}
                    disabled={!entry.available || excludedByRoasty}
                    onChange={(event) => setDeniedCapabilities((current) => event.currentTarget.checked
                      ? current.filter((value) => value !== key)
                      : [...new Set([...current, key])])}
                  />
                  <span>{entry.id}<small>{!entry.available ? "Unavailable; requests fail closed" : excludedByRoasty ? "Excluded by Roasty v1" : deniedCapabilities.includes(key) ? "Denied at this scope" : "Locally available"}</small></span>
                </label>;
              })}
            </details>)}
          </div>
        </div>

        <div className="policy-card effective-policy-preview">
          <strong>Effective-policy preview</strong>
          <dl><div><dt>Hash</dt><dd title={effective?.effective_policy_hash}>{shortHash(effective?.effective_policy_hash)}</dd></div>
            <div><dt>Models</dt><dd>{effective?.permitted_counts?.models ?? 0}</dd></div>
            <div><dt>Formats</dt><dd>{effective?.permitted_counts?.export_formats ?? 0}</dd></div>
            <div><dt>Runtimes</dt><dd>{effective?.permitted_counts?.runtimes ?? 0}</dd></div></dl>
          <strong>Inheritance</strong>
          <ol className="policy-source-list">
            {(effective?.snapshot?.compatibility_profiles ?? []).map((profile) => <li key={`profile-${profile.id}-${profile.version}`}>
              <span>profile</span><code>{profile.id}@{profile.version}</code>
            </li>)}
            {(effective?.snapshot?.policy_sources ?? []).map((source) => <li key={`${source.scope}-${source.subject_id}`}>
              <span>{source.scope}</span><code>{source.subject_id}</code><small>rev {source.binding_revision}</small>
            </li>)}
            {(effective?.snapshot?.policy_sources ?? []).length === 0 && <li><span>implicit</span><code>allow_all_v0</code></li>}
          </ol>
          {Boolean(effective?.blocked_dimensions?.length) && <div className="policy-error"><strong>No valid configuration</strong><span>{effective?.blocked_dimensions?.join(", ")}</span></div>}
        </div>

        <div className="policy-card queue-impact">
          <strong>Queue impact</strong>
          <div><span><small>Queued</small>{queue.queued}</span><span><small>Allowed</small>{queue.allowed}</span>
            <span><small>Recheck</small>{queue.pending}</span><span><small>Blocked</small>{queue.blocked}</span></div>
          <p>Policy edits mark affected queued work for atomic re-evaluation before claim.</p>
        </div>

        <div className="policy-card policy-audit">
          <strong>Recent enforcement audit</strong>
          <ul>{audit.map((entry) => <li key={entry.id}><span>{entry.operation}<small>{entry.reason_codes?.join(", ")}</small></span><b>{entry.decision}</b><small>{new Date(entry.created_at).toLocaleString()}</small></li>)}</ul>
          {audit.length === 0 && <p>No persisted evaluations for this project yet.</p>}
        </div>
      </div>

      {error && <div className="policy-error" role="alert"><strong>{error.title}</strong>{error.details.map((detail) => <span key={detail}>{detail}</span>)}</div>}
      <div className="settings-actions"><small>{response?.active_binding ? `Local revision ${response.active_binding.revision}` : "Inherited policy only"}</small>
        <button className="command" type="button" disabled={busy || !response?.active_binding} onClick={clearPolicy}>Clear local policy</button>
        <button className="command primary" type="button" disabled={busy || !subjectId} onClick={savePolicy}>{busy ? "Checking…" : "Apply policy"}</button>
      </div>
    </section>
  );
}
