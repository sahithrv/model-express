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
  image_size?: number;
  resolution_strategy?: string;
  preprocessing?: {
    resize_strategy?: string;
    normalization?: string;
    crop_strategy?: string;
    bbox_mode?: string;
    use_dataset_normalization?: boolean;
  };
  optimizer?: string;
  scheduler?: string;
  weight_decay?: number;
  augmentation?: Record<string, unknown>;
  augmentation_policy?: string;
  class_balancing?: string;
  sampling_strategy?: string;
  early_stopping_patience?: number;
  strategy?: string;
  pretrained?: boolean;
  freeze_backbone?: boolean;
  fine_tune_strategy?: string;
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

export type TrainingRunEvaluation = {
  job_id: string;
  project_id: string;
  plan_id?: string;
  dataset_id?: string;
  objective_profile: Record<string, unknown>;
  per_class_metrics: Record<string, unknown>;
  confusion_matrix: number[][];
  model_profile: Record<string, unknown>;
  holistic_scores: Record<string, unknown>;
  recommendation_summary: string;
  created_at: string;
  updated_at: string;
};

export type ProjectChampion = {
  id: string;
  project_id: string;
  dataset_id: string;
  plan_id: string;
  job_id: string;
  source_decision_id: string;
  selection_reason: string;
  metrics: Record<string, unknown>;
  evaluation: Record<string, unknown>;
  deployment_profile: Record<string, unknown>;
  champion_exports?: ChampionExport[];
  demo_images?: ChampionDemoImage[];
  demo_predictions?: ChampionDemoPrediction[];
  created_at: string;
  updated_at: string;
};

export type ChampionExport = {
  id?: string;
  project_id?: string;
  champion_id?: string;
  job_id?: string;
  status?: string;
  format?: string;
  artifact_uri?: string;
  model_uri?: string;
  download_url?: string;
  size_bytes?: number;
  validation_errors?: string[];
  created_at?: string;
  updated_at?: string;
  error?: string;
  metadata?: Record<string, unknown>;
};

export type ChampionDemoImage = {
  id?: string;
  image_id?: string;
  uri?: string;
  image_uri?: string;
  thumbnail_uri?: string;
  class_name?: string;
  label?: string;
  true_label?: string;
  split?: string;
  width?: number;
  height?: number;
  size_bytes?: number;
  metadata?: Record<string, unknown>;
};

export type ChampionDemoPrediction = {
  id?: string;
  image_id?: string;
  image_uri?: string;
  status?: string;
  predicted_label?: string;
  true_label?: string;
  confidence?: number;
  latency_ms?: number;
  correct?: boolean;
  top_k?: Array<{ label?: string; class_name?: string; confidence?: number; score?: number }>;
  error?: string;
  error_message?: string;
  runtime_unavailable?: boolean;
  created_at?: string;
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

export type StrategyScorecard = {
  id: string;
  project_id: string;
  dataset_id: string;
  source_decision_id: string;
  source_plan_id: string;
  followup_plan_id: string;
  strategy_type: string;
  planning_mode: string;
  dataset_traits: Record<string, unknown>;
  objective_profile: Record<string, unknown>;
  proposed_changes: Record<string, unknown>;
  expected_delta: number;
  actual_delta: number;
  confidence_before: number;
  confidence_after: number;
  cost_usd: number;
  runtime_seconds: number;
  outcome: string;
  lesson: string;
  tags: string[];
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
