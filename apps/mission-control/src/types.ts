export type Project = {
  id: string;
  name: string;
  goal: string;
  status: string;
  created_at: string;
  updated_at: string;
};

export type Dataset = {
  id: string;
  project_id: string;
  name: string;
  storage_uri: string;
  checksum_sha256?: string;
  size_bytes: number;
  profile: Record<string, unknown>;
  status: string;
  created_at: string;
  profiled_at?: string;
};

export type Worker = {
  id: string;
  project_id: string;
  name: string;
  status: string;
  gpu_type: string;
  last_heartbeat: string;
  current_job_id?: string;
};

export type Job = {
  id: string;
  project_id: string;
  worker_id?: string;
  template: string;
  status: string;
  config: Record<string, unknown>;
  mlflow_run_id?: string;
  error?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
};

export type PlannedExperiment = {
  template: string;
  model: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  reason: string;
};

export type ExperimentPlan = {
  id: string;
  project_id: string;
  dataset_id: string;
  status: string;
  source_decision_id?: string;
  target_metric: string;
  recommended_workers: number;
  estimated_minutes: number;
  experiments: PlannedExperiment[];
  warnings: string[];
  created_at: string;
};

export type TrainingRunSummary = {
  job_id: string;
  project_id: string;
  plan_id?: string;
  dataset_id?: string;
  model: string;
  provider: string;
  gpu_type: string;
  status: string;
  runtime_seconds: number;
  estimated_cost_usd: number;
  best_macro_f1: number;
  best_accuracy: number;
  final_train_loss: number;
  final_val_loss: number;
  epochs_completed: number;
  modal_function_call_id?: string;
  modal_input_id?: string;
  created_at: string;
  updated_at: string;
};

export type AgentDecision = {
  id: string;
  project_id: string;
  plan_id?: string;
  decision_type: string;
  rationale: string;
  payload: Record<string, unknown>;
  created_at: string;
};

export type WorkerRequirement = {
  id: string;
  project_id: string;
  plan_id: string;
  provider: string;
  gpu_type: string;
  target_count: number;
  status: string;
  source: string;
  last_error?: string;
  created_at: string;
  updated_at: string;
};

export type ExecutionEvent = {
  id: string;
  project_id: string;
  plan_id?: string;
  event_type: string;
  message: string;
  payload: Record<string, unknown>;
  created_at: string;
};

export type AgentMemoryRecord = {
  id: string;
  invocation_id?: string;
  project_id: string;
  dataset_id?: string;
  plan_id?: string;
  job_id?: string;
  agent_name: string;
  kind: string;
  summary: string;
  payload: Record<string, unknown>;
  tags: string[];
  created_at: string;
};

export type AgentInvocation = {
  id: string;
  project_id: string;
  dataset_id?: string;
  plan_id?: string;
  job_id?: string;
  agent_name: string;
  agent_version?: string;
  prompt_version?: string;
  provider?: string;
  model?: string;
  input_messages: Array<Record<string, string>>;
  input_context: Record<string, unknown>;
  raw_output: string;
  parsed_output: Record<string, unknown>;
  validation_status: string;
  validation_error?: string;
  accepted_for_memory: boolean;
  human_feedback: Record<string, unknown>;
  downstream_outcome: Record<string, unknown>;
  created_at: string;
};

export type EpochMetric = {
  job_id: string;
  epoch: number;
  metrics: Record<string, number>;
  created_at: string;
};

export type Health = {
  status: string;
  service: string;
  timestamp: string;
};

export type AutomationSettings = {
  auto_review_experiments: boolean;
  auto_schedule_followups: boolean;
  auto_execute_plans: boolean;
  max_followup_rounds: number;
  default_training_provider: string;
  default_gpu_type: string;
  llm_enabled: boolean;
  agent_mode: string;
  llm_provider: string;
  llm_model: string;
  updated_at: string;
};
