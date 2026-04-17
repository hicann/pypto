export type TransitionAction = "start_stage" | "complete_stage" | "fail_stage" | "init";

export type TransitionInput = {
  action: TransitionAction;
  stage: number;
};

export type OrchestratorState = {
  operator_name?: string;
  current_stage: number;
  stage_status: Record<string, string>;
  stage_retry_count?: Record<string, number>;
  last_updated?: string;
  [key: string]: unknown;
};

function cloneState(prev: OrchestratorState): OrchestratorState {
  return JSON.parse(JSON.stringify(prev));
}

function ensureNumber(val: unknown): number {
  const n = Number(val);
  if (!Number.isFinite(n) || n < 1) {
    throw new Error(`invalid stage: ${val}`);
  }
  return Math.floor(n);
}

export function applyTransition(
  prev: OrchestratorState,
  input: TransitionInput,
): OrchestratorState {
  const stage = ensureNumber(input.stage);
  const next = cloneState(prev);
  const statusMap = next.stage_status ?? {};
  next.stage_status = statusMap;
  const retryMap = next.stage_retry_count ?? {};
  next.stage_retry_count = retryMap;

  switch (input.action) {
    case "init": {
      if (stage !== 1) {
        throw new Error(`init action must target stage 1, got stage ${stage}`);
      }
      const hasInProgress = Object.values(statusMap).some((s) => s === "in_progress");
      if (hasInProgress) {
        throw new Error(`cannot init: a stage is already in_progress`);
      }
      next.current_stage = stage;
      statusMap[String(stage)] = "in_progress";
      break;
    }

    case "start_stage": {
      const otherInProgress = Object.entries(statusMap).some(
        ([k, s]) => s === "in_progress" && k !== String(stage),
      );
      if (otherInProgress) {
        throw new Error(`cannot start stage ${stage}: another stage is already in_progress`);
      }
      const key = String(stage);
      if (statusMap[key] === "completed") {
        throw new Error(`cannot start stage ${stage}: already completed`);
      }
      if (stage > 1) {
        const prevKey = String(stage - 1);
        const prevStatus = statusMap[prevKey];
        if (prevStatus !== "completed") {
          throw new Error(
            `cannot start stage ${stage}: previous stage ${stage - 1} is "${prevStatus ?? "unknown"}", not "completed"`,
          );
        }
      }
      next.current_stage = stage;
      statusMap[String(stage)] = "in_progress";
      break;
    }

    case "complete_stage": {
      if (stage !== prev.current_stage) {
        throw new Error(
          `cannot complete stage ${stage}: current_stage is ${prev.current_stage}`,
        );
      }
      const compKey = String(stage);
      if (statusMap[compKey] !== "in_progress") {
        throw new Error(
          `cannot complete stage ${stage}: status is "${statusMap[compKey]}", not "in_progress"`,
        );
      }
      statusMap[compKey] = "completed";
      // Auto-advance to next stage
      const nextStage = stage + 1;
      const nextKey = String(nextStage);
      if (nextKey in statusMap) {
        next.current_stage = nextStage;
        statusMap[nextKey] = "in_progress";
      }
      break;
    }

    case "fail_stage": {
      const failKey = String(stage);
      retryMap[failKey] = (Number(retryMap[failKey]) || 0) + 1;
      statusMap[failKey] = "failed";
      break;
    }

    default:
      throw new Error(`unsupported action: ${(input as { action?: string }).action ?? "unknown"}`);
  }

  next.last_updated = new Date().toISOString();
  return next;
}
