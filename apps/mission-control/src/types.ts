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

export type DatasetMetadataIssue =
  | string
  | {
      code?: string;
      severity?: string;
      message?: string;
      reason?: string;
      source_kind?: string;
      source_format?: string;
      declared_format?: string;
      detected_format?: string;
      count?: number;
      [key: string]: unknown;
    };

export type DatasetMetadataSummary = {
  status?: string;
  import_id?: string;
  source_kinds?: string[];
  source_formats?: string[];
  formats?: string[];
  class_count?: number;
  sample_count?: number;
  split_counts?: Record<string, number>;
  annotation_counts?: Record<string, number>;
  bbox_coverage?: number | string | Record<string, unknown>;
  bbox_count?: number;
  bbox_counts?: Record<string, number> | Record<string, unknown>;
  warnings?: DatasetMetadataIssue[];
  errors?: DatasetMetadataIssue[];
  official_split_available?: boolean;
  unsupported_source_count?: number;
  created_at?: string;
  completed_at?: string;
  [key: string]: unknown;
};

export type DatasetMetadataImport = {
  id?: string;
  import_id?: string;
  dataset_id?: string;
  project_id?: string;
  status?: string;
  import_version?: number;
  source_kind?: string;
  active?: boolean;
  strict_mode?: boolean;
  summary?: DatasetMetadataSummary;
  agent_safe_summary?: DatasetMetadataSummary;
  warnings?: DatasetMetadataIssue[];
  errors?: DatasetMetadataIssue[];
  created_at?: string;
  completed_at?: string;
  [key: string]: unknown;
};

export type VisualCoverageReport = {
  selection_strategy?: string;
  selection_basis?: string[];
  images_available?: number;
  images_analyzed?: number;
  classes_total?: number;
  classes_covered?: number;
  class_coverage_ratio?: number;
  per_class_counts?: Record<string, number>;
  hard_example_count?: number;
  edge_case_count?: number;
  high_detail_image_count?: number;
  limitations?: string[];
};

export type VisualTrait = {
  trait?: string;
  level?: string;
  confidence?: string;
  evidence?: string[];
  example_image_ids?: string[];
  affected_classes?: string[];
  notes?: string;
};

export type ClassWatchItem = {
  class_name?: string;
  reason?: string;
  related_classes?: string[];
  evidence?: string[];
  example_image_ids?: string[];
  confidence?: string;
};

export type PreprocessingHypothesis = {
  id?: string;
  mechanism?: string;
  summary?: string;
  evidence?: string[];
  suggested_preprocessing?: Record<string, unknown>;
  suggested_image_sizes?: number[];
  suggested_augmentation_policy?: string;
  suggested_augmentation_policy_config?: Record<string, unknown>;
  expected_effect?: string;
  risk?: string;
  confidence?: string;
  support_status?: string;
  unsupported_reason?: string;
};

export type VisualCaution = {
  operation?: string;
  reason?: string;
  severity?: string;
  confidence?: string;
  affected_classes?: string[];
  example_image_ids?: string[];
};

export type DatasetVisualAnalysis = {
  id?: string;
  project_id?: string;
  dataset_id?: string;
  dataset_name?: string;
  schema_version?: string;
  analysis_version?: number;
  prompt_version?: string;
  agent_name?: string;
  agent_version?: string;
  provider?: string;
  model?: string;
  trigger_reason?: string;
  trigger_details?: Record<string, unknown>;
  source_job_id?: string;
  source_invocation_id?: string;
  profile_schema_version?: string;
  profile_fingerprint?: string;
  total_images?: number;
  images_analyzed?: number;
  coverage_report?: VisualCoverageReport;
  classes_to_watch?: ClassWatchItem[];
  confidence?: string;
  visual_traits?: VisualTrait[];
  preprocessing_hypotheses?: PreprocessingHypothesis[];
  cautions?: VisualCaution[];
  limitations?: string[];
  validation_status?: string;
  validation_errors?: string[];
  created_at?: string;
  updated_at?: string;
};

export type VisualAnalysisRerunPolicy = {
  enabled?: boolean;
  automation_enabled?: boolean;
  manual_run_allowed?: boolean;
  initial_run_allowed?: boolean;
  deficiency_run_allowed?: boolean;
  run_allowed?: boolean;
  disabled_reason?: string;
  reason?: string;
  next_allowed_at?: string;
  cooldown_seconds?: number;
  max_runs_per_profile?: number;
  runs_for_profile?: number;
  accepted_runs_for_profile?: number;
  active_job_id?: string;
  active_job_status?: string;
  profile_fingerprint?: string;
  deficiency_triggers?: string[];
  deficiency_severity?: number;
  latest_analysis_id?: string;
  latest_analysis_created_at?: string;
  latest_analysis_validation_status?: string;
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
  mechanism?: string;
  intervention?: string;
  expected_effect?: string;
  evidence_used?: string[];
  validation_status?: string;
  validation_error?: string;
  backend_validation_status?: string;
  backend_validation_error?: string;
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
  dropout?: number;
  optimizer_momentum?: number;
  scheduler_step_size?: number;
  scheduler_gamma?: number;
  label_smoothing?: number;
  gradient_clip_norm?: number;
  augmentation?: Record<string, unknown>;
  augmentation_policy?: string;
  augmentation_policy_config?: {
    policy_type?: string;
    magnitude?: number;
    num_ops?: number;
    num_magnitude_bins?: number;
    probability?: number;
    alpha?: number;
  };
  class_balancing?: string;
  class_balancing_config?: Record<string, unknown>;
  sampling_strategy?: string;
  early_stopping_patience?: number;
  strategy?: string;
  pretrained?: boolean;
  freeze_backbone?: boolean;
  fine_tune_strategy?: string;
  automl?: ExperimentAutoML;
};

export type AutoMLParameterSpec = {
  name: string;
  type: "float" | "integer" | "categorical" | string;
  min?: number;
  max?: number;
  step?: number;
  scale?: "linear" | "log" | string;
  choices?: string[];
  int_choices?: number[];
  default?: unknown;
  source?: string;
  condition?: { field?: string; equals?: string };
  notes?: string;
};

export type ExperimentAutoML = {
  enabled?: boolean;
  intent?: {
    summary?: string;
    planning_mode?: string;
    exploration_intent?: string;
    goals?: string[];
    allowed_parameters?: string[];
    strategy_description?: string;
  };
  sampler?: string;
  seed?: number;
  search_space?: { parameters?: AutoMLParameterSpec[] };
  suggestion?: {
    id?: string;
    study_id?: string;
    sampler?: string;
    seed?: number;
    values?: Record<string, unknown>;
    final_values?: Record<string, unknown>;
    provenance?: Record<string, string>;
    validation_status?: string;
    validation_errors?: string[];
  };
  final_values?: Record<string, unknown>;
  value_provenance?: Record<string, string>;
  strategy_snapshot?: Record<string, unknown>;
  validation_status?: string;
  validation_errors?: string[];
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
  best_map50_95?: number;
  best_map50?: number;
  best_precision?: number;
  best_recall?: number;
  target_metric?: string;
  final_train_loss: number;
  final_val_loss: number;
  epochs_completed: number;
  modal_function_call_id?: string;
  modal_input_id?: string;
  dataset_materialization?: Record<string, unknown>;
  stage_telemetry?: Record<string, unknown>;
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
  requested_at?: string;
  started_at?: string;
  completed_at?: string;
  failed_at?: string;
  validation_errors?: string[];
  created_at?: string;
  updated_at?: string;
  error?: string;
  error_message?: string;
  metadata?: Record<string, unknown>;
  portable_bundle_uri?: string;
  portable_inference_bundle?: PortableInferenceBundle;
};

export type PortableInferenceBundle = {
  schema_version?: string;
  status?: string;
  artifact_uri?: string;
  artifact_path?: string;
  uri?: string;
  path?: string;
  bytes?: number;
  contents?: string[];
  error_code?: string;
  error?: string;
};

export type ChampionDemoImage = {
  id?: string;
  image_id?: string;
  uri?: string;
  image_uri?: string;
  preview_uri?: string;
  thumbnail_uri?: string;
  original_image_uri?: string;
  source_artifact_uri?: string;
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
  project_id?: string;
  champion_id?: string;
  image_id?: string;
  image_uri?: string;
  status?: string;
  predicted_label?: string;
  true_label?: string;
  confidence?: number;
  latency_ms?: number;
  correct?: boolean;
  top_k?: Array<{ label?: string; class_name?: string; confidence?: number; probability?: number; score?: number }>;
  detections?: ChampionDetection[];
  detection_count?: number;
  postprocess_latency_ms?: number;
  error?: string;
  error_message?: string;
  runtime_unavailable?: boolean;
  requested_at?: string;
  started_at?: string;
  completed_at?: string;
  created_at?: string;
  updated_at?: string;
  image_metadata?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
};

export type ChampionDetection = {
  label?: string;
  class_name?: string;
  class_id?: number;
  confidence?: number;
  score?: number;
  box?: {
    x?: number;
    y?: number;
    width?: number;
    height?: number;
    x1?: number;
    y1?: number;
    x2?: number;
    y2?: number;
  };
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
};

export type ChampionFeedback = {
  id: string;
  project_id: string;
  champion_id: string;
  prediction_id?: string;
  job_id?: string;
  dataset_id?: string;
  image_uri?: string;
  image_id?: string;
  rating: "good" | "mediocre" | "bad";
  message?: string;
  prediction_snapshot?: Record<string, unknown>;
  metrics_snapshot?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type RetrievedMemoryCard = {
  source_table?: string;
  source_id?: string;
  project_id?: string;
  dataset_id?: string;
  plan_id?: string;
  job_id?: string;
  invocation_id?: string;
  decision_id?: string;
  source_decision_id?: string;
  source_plan_id?: string;
  followup_plan_id?: string;
  kind?: string;
  scope?: string;
  outcome?: string;
  outcome_status?: string;
  mechanism?: string;
  intervention?: string;
  lesson?: string;
  summary?: string;
  compact_lesson?: string;
  compact_summary?: string;
  retrieval_reason?: string;
  reason_for_retrieval?: string;
  match_reason?: string;
  score?: number;
  semantic_score?: number;
  structured_score?: number;
  summary_card?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
};

export type LLMUsage = {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  cached_input_tokens?: number;
  reasoning_tokens?: number;
  request_model?: string;
  api_style?: string;
  tool_rounds?: number;
  [key: string]: unknown;
};

export type AgentInvocationRuntime = {
  api_style?: string;
  provider?: string;
  model?: string;
  reasoning_effort?: string;
  plateau_reasoning_effort?: string;
  stored_responses?: boolean;
  max_tool_rounds?: number;
  tool_rounds?: number;
  tool_names?: string[];
  rejected_tool_calls?: unknown[];
  dry_run_validation_results?: unknown[];
  tool_calls?: unknown[];
  tool_results?: unknown[];
  tool_calls_are_questions?: boolean;
  mutation_authority?: boolean;
  llm_usage?: LLMUsage;
  [key: string]: unknown;
};

export type RetrievedMemoryPayload = {
  retrieval_enabled?: boolean;
  successful_strategy_cards?: RetrievedMemoryCard[];
  failed_strategy_cards?: RetrievedMemoryCard[];
  blocked_or_rejected_cards?: RetrievedMemoryCard[];
  dataset_preprocessing_cards?: RetrievedMemoryCard[];
  run_dynamics_cards?: RetrievedMemoryCard[];
  cards?: RetrievedMemoryCard[];
  items?: RetrievedMemoryCard[];
  results?: RetrievedMemoryCard[];
  hits?: RetrievedMemoryCard[];
  caps?: Record<string, unknown>;
  [key: string]: unknown;
};

export type CandidateRanking = {
  candidate_index?: number;
  hypothesis?: string;
  planning_mode?: string;
  mechanism?: string;
  intervention?: string;
  expected_effect?: string;
  score?: number;
  total_score?: number;
  score_components?: Record<string, unknown>;
  selected?: boolean;
  rejected?: boolean;
  reasons?: string[];
  memory_reasons?: string[];
  retrieved_memory_reasons?: string[];
  retrieval_reasons?: string[];
  retrieved_memory?: RetrievedMemoryCard[] | RetrievedMemoryPayload;
  retrieved_memory_hits?: RetrievedMemoryCard[];
  memory_retrieval_hits?: RetrievedMemoryCard[];
  memory_hits?: RetrievedMemoryCard[];
  memory_cards?: RetrievedMemoryCard[];
  retrieved_cards?: RetrievedMemoryCard[];
  experiment_signature?: string;
  [key: string]: unknown;
};

export type AgentDecision = {
  id: string;
  project_id: string;
  plan_id?: string;
  decision_type: string;
  rationale: string;
  payload: Record<string, unknown> & {
    retrieved_memory?: RetrievedMemoryCard[] | RetrievedMemoryPayload;
    retrieved_run_memory?: RetrievedMemoryCard[];
    candidate_rankings?: CandidateRanking[];
    planner_context_snapshot?: {
      retrieved_memory?: RetrievedMemoryCard[] | RetrievedMemoryPayload;
      [key: string]: unknown;
    };
    training_monitor_context_snapshot?: {
      retrieved_run_memory?: RetrievedMemoryCard[];
      [key: string]: unknown;
    };
  };
  created_at: string;
};

export type StrategyScorecard = {
  id: string;
  project_id: string;
  dataset_id: string;
  source_decision_id: string;
  source_plan_id: string;
  followup_plan_id: string;
  mechanism?: string;
  intervention?: string;
  expected_effect?: string;
  evidence_used?: string[];
  diagnosis_triggers?: string[];
  validation_status?: string;
  validation_error?: string;
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
  dataset_id?: string;
  dataset_checksum?: string;
  dataset_cache_key?: string;
  dataset_materialization_status?: string;
  cold_cache_policy?: string;
  max_concurrent_jobs?: number;
  max_cold_dataset_materializations?: number;
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

export type AgentActivityEvent = {
  id: string;
  project_id: string;
  plan_id?: string;
  job_id?: string;
  type: string;
  severity: "info" | "warning" | "error" | "success" | string;
  title: string;
  message: string;
  status: "active" | "waiting" | "succeeded" | "failed" | "blocked" | string;
  created_at: string;
  metadata?: Record<string, unknown>;
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
  input_messages?: Array<Record<string, string>>;
  input_context?: Record<string, unknown> & {
    invocation_runtime?: AgentInvocationRuntime;
    retrieved_memory?: RetrievedMemoryCard[] | RetrievedMemoryPayload;
    retrieved_run_memory?: RetrievedMemoryCard[];
    planner_context_snapshot?: {
      retrieved_memory?: RetrievedMemoryCard[] | RetrievedMemoryPayload;
      [key: string]: unknown;
    };
    training_monitor_context_snapshot?: {
      retrieved_run_memory?: RetrievedMemoryCard[];
      [key: string]: unknown;
    };
  };
  raw_output?: string;
  parsed_output?: Record<string, unknown>;
  validation_status?: string;
  validation_error?: string;
  accepted_for_memory?: boolean;
  human_feedback?: Record<string, unknown>;
  downstream_outcome?: Record<string, unknown>;
  created_at: string;
};

export type MemoryEmbeddingUsageEvent = {
  id: string;
  project_id: string;
  dataset_id?: string;
  plan_id?: string;
  job_id?: string;
  invocation_id?: string;
  purpose: string;
  retrieval_purpose?: string;
  source_table?: string;
  source_id?: string;
  embedding_model?: string;
  embedding_dimensions?: number;
  input_bytes?: number;
  provider_call_count?: number;
  retrieved_count?: number;
  injected?: boolean;
  log_only?: boolean;
  cached?: boolean;
  skipped?: boolean;
  skip_reason?: string;
  source_text_hash?: string;
  query_hash?: string;
  provider_usage?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  created_at: string;
};

export type MissionControlTelemetryResponse = {
  project_id?: string;
  generated_at?: string;
  limit?: number;
  agent_invocations?: AgentInvocation[];
  invocations?: AgentInvocation[];
  memory_embedding_usage_events?: MemoryEmbeddingUsageEvent[];
  embedding_usage_events?: MemoryEmbeddingUsageEvent[];
  usage_events?: MemoryEmbeddingUsageEvent[];
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
  cost_mode: string;
  budget_cap_usd: number;
  llm_enabled: boolean;
  agent_mode: string;
  llm_provider: string;
  llm_model: string;
  automl_enabled: boolean;
  automl_sampler: string;
  updated_at: string;
};
