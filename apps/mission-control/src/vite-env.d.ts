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
  };
}
