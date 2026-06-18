/// <reference types="vite/client" />

interface OrchestratorRequest {
  baseUrl: string;
  method?: string;
  path: string;
  body?: unknown;
}

interface Window {
  missionControl: {
    request<T>(request: OrchestratorRequest): Promise<T>;
    selectDatasetFolder(): Promise<{
      token: string;
      path: string;
      name: string;
      preflight?: DatasetUploadPreflight;
    } | null>;
    preflightDatasetFolder(options: {
      datasetToken: string;
      datasetPath?: string;
      warnFileCount?: number;
      warnBytes?: number;
      maxFileCount?: number;
      maxBytes?: number;
    }): Promise<DatasetUploadPreflight>;
    preflightCloud(options: {
      stage: "dataset_upload" | "plan_execution" | "worker_start" | "manual";
      baseUrl: string;
      live?: boolean;
    }): Promise<CloudPreflightResult>;
    uploadDatasetFolder(options: {
      projectId: string;
      datasetToken: string;
      datasetPath?: string;
      endpoint?: string;
      bucket?: string;
      accessKeyId?: string;
      secretAccessKey?: string;
      region?: string;
    }): Promise<{
      name: string;
      storage_uri: string;
      checksum_sha256: string;
      size_bytes: number;
      file_count?: number;
      uncompressed_size_bytes?: number;
      upload_warnings?: DatasetUploadWarning[];
    }>;
    selectAndUploadDataset(options: {
      projectId: string;
      endpoint?: string;
      bucket?: string;
      accessKeyId?: string;
      secretAccessKey?: string;
      region?: string;
    }): Promise<{
      name: string;
      storage_uri: string;
      checksum_sha256: string;
      size_bytes: number;
    } | null>;
    selectDemoImage(): Promise<{
      path: string;
      name: string;
      uri: string;
      image_uri: string;
      image_id: string;
      thumbnail_uri?: string;
      split?: string;
      size_bytes?: number;
      metadata?: Record<string, unknown>;
    } | null>;
    loadModelArtifact(options: {
      artifactUri: string;
      externalData?: Array<{
        path?: string;
        relative_path?: string;
        uri?: string;
        artifact_uri?: string;
        artifact_path?: string;
        local_path?: string;
        file_name?: string;
      }>;
      endpoint?: string;
      accessKeyId?: string;
      secretAccessKey?: string;
      region?: string;
    }): Promise<{
      artifact_uri: string;
      size_bytes: number;
      bytes: ArrayBuffer;
      external_data?: Array<{
        path: string;
        uri?: string;
        size_bytes?: number;
        bytes: ArrayBuffer;
      }>;
    }>;
    saveArtifact(options: {
      artifactUri: string;
      defaultPath?: string;
      endpoint?: string;
      accessKeyId?: string;
      secretAccessKey?: string;
      region?: string;
    }): Promise<{
      artifact_uri: string;
      path: string;
      size_bytes: number;
    } | null>;
    saveExportArtifact(options: {
      artifactUri?: string;
      artifactPath?: string;
      suggestedName?: string;
      kind?: string;
      endpoint?: string;
      accessKeyId?: string;
      secretAccessKey?: string;
      region?: string;
    }): Promise<{
      canceled: boolean;
      file_path: string;
      bytes: number;
      artifact_uri?: string;
    }>;
    ensureProjectWorker(options: {
      projectId: string;
      baseUrl: string;
      name?: string;
      gpuType?: string;
      count?: number;
    }): Promise<{
      project_id: string;
      pid: number;
      pids: number[];
      started: boolean;
      started_count: number;
      running_count: number;
      process_count?: number;
      dispatcher?: boolean;
      status: string;
    }>;
    stopProjectWorker(options: {
      projectId: string;
    }): Promise<{
      project_id: string;
      stopped_count: number;
      stopped_pids: number[];
      already_stopped_count: number;
      status: string;
    }>;
  };
}

interface DatasetUploadWarning {
  code: string;
  message: string;
  threshold?: number;
  value?: number;
}

interface DatasetUploadPreflight {
  file_count: number;
  uncompressed_size_bytes: number;
  archive_size_bytes: number;
  largest_file?: {
    path: string;
    size_bytes: number;
  } | null;
  warnings: DatasetUploadWarning[];
  errors: DatasetUploadWarning[];
}

interface CloudPreflightCheck {
  id: string;
  status: "ok" | "failed" | "warning" | string;
  message: string;
  remediation?: string;
  metadata?: Record<string, unknown>;
}

interface CloudPreflightResult {
  status: "ok" | "failed" | string;
  checks: CloudPreflightCheck[];
}
