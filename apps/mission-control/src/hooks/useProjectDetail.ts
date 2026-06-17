import type {
  AgentDecision,
  AgentInvocation,
  AgentMemoryRecord,
  ChampionDemoImage,
  ChampionDemoPrediction,
  ChampionExport,
  ChampionFeedback,
  Dataset,
  DatasetMetadataImport,
  DatasetMetadataSummary,
  DatasetVisualAnalysis,
  ExecutionEvent,
  ExperimentPlan,
  Job,
  MemoryEmbeddingUsageEvent,
  MissionControlTelemetryResponse,
  ProjectChampion,
  StrategyScorecard,
  TrainingRunEvaluation,
  TrainingRunSummary,
  VisualAnalysisRerunPolicy,
  Worker,
  WorkerRequirement,
} from "../types";

export type VisualAnalysisDetail = {
  analysis: DatasetVisualAnalysis | null;
  status: "available" | "empty" | "unsupported" | "error";
  message: string;
  manualRunSupported: boolean;
  rerunPolicy?: VisualAnalysisRerunPolicy | null;
};

export type DatasetMetadataDetail = {
  summary: DatasetMetadataSummary | null;
  imports: DatasetMetadataImport[];
  status: "available" | "empty" | "unsupported" | "error";
  message: string;
};

export type ProjectDetailLoadStatus = {
  status: "available" | "empty" | "stale" | "error";
  message: string;
  last_success_at?: string;
};

export type ProjectDetailLoadStatusKey =
  | "runEvaluations"
  | "decisions"
  | "workerRequirements"
  | "championExports"
  | "championDemoPredictions"
  | "championFeedback"
  | "liveRefresh";

export type ProjectDetailLoadStatusMap = Record<ProjectDetailLoadStatusKey, ProjectDetailLoadStatus>;

export type ChampionExportsStatus = ProjectDetailLoadStatus;

export type ProjectDetail = {
  decisions: AgentDecision[];
  datasets: Dataset[];
  telemetry: MissionControlTelemetryResponse | null;
  visualAnalysis: VisualAnalysisDetail;
  datasetMetadata: DatasetMetadataDetail;
  jobs: Job[];
  plans: ExperimentPlan[];
  runSummaries: TrainingRunSummary[];
  runEvaluations: TrainingRunEvaluation[];
  champion: ProjectChampion | null;
  championExports: ChampionExport[];
  championExportsStatus: ChampionExportsStatus;
  championDemoImages: ChampionDemoImage[];
  championDemoPredictions: ChampionDemoPrediction[];
  championFeedback: ChampionFeedback[];
  workers: Worker[];
  workerRequirements: WorkerRequirement[];
  executionEvents: ExecutionEvent[];
  agentInvocations: AgentInvocation[];
  agentMemory: AgentMemoryRecord[];
  strategyScorecards: StrategyScorecard[];
  loadStatus: ProjectDetailLoadStatusMap;
};

export function emptyProjectDetailLoadStatus(): ProjectDetailLoadStatusMap {
  return {
    runEvaluations: {
      status: "empty",
      message: "Training evaluations have not been loaded.",
    },
    decisions: {
      status: "empty",
      message: "Agent decisions have not been loaded.",
    },
    workerRequirements: {
      status: "empty",
      message: "Worker requirements have not been loaded.",
    },
    championExports: {
      status: "empty",
      message: "No champion export records have been loaded.",
    },
    championDemoPredictions: {
      status: "empty",
      message: "Champion demo prediction history has not been loaded.",
    },
    championFeedback: {
      status: "empty",
      message: "Champion feedback has not been loaded.",
    },
    liveRefresh: {
      status: "empty",
      message: "Live refresh has not run yet.",
    },
  };
}

export function emptyProjectDetail(message = "Select a dataset to load visual analysis evidence."): ProjectDetail {
  const loadStatus = emptyProjectDetailLoadStatus();
  return {
    decisions: [],
    datasets: [],
    telemetry: null,
    visualAnalysis: {
      analysis: null,
      status: "empty",
      message,
      manualRunSupported: false,
    },
    datasetMetadata: {
      summary: null,
      imports: [],
      status: "empty",
      message: "Dataset metadata imports have not been reported.",
    },
    jobs: [],
    plans: [],
    runSummaries: [],
    runEvaluations: [],
    champion: null,
    championExports: [],
    championExportsStatus: {
      status: "empty",
      message: "No champion export records have been loaded.",
    },
    championDemoImages: [],
    championDemoPredictions: [],
    championFeedback: [],
    workers: [],
    workerRequirements: [],
    executionEvents: [],
    agentInvocations: [],
    agentMemory: [],
    strategyScorecards: [],
    loadStatus,
  };
}
