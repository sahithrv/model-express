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
      path: string;
      name: string;
    } | null>;
    uploadDatasetFolder(options: {
      projectId: string;
      datasetPath: string;
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
