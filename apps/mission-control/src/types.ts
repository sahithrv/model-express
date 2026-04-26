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
