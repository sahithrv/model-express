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
  };
}
