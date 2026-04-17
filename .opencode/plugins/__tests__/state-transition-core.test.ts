import { describe, test, expect } from "bun:test";
import { applyTransition, type OrchestratorState } from "../lib/state-transition-core";

function makeState(overrides: Partial<OrchestratorState> = {}): OrchestratorState {
  return {
    operator_name: "test_op",
    current_stage: 1,
    stage_status: {
      "1": "completed",
      "2": "pending",
      "3": "pending",
      "4": "pending",
      "5": "pending",
      "6": "pending",
      "7": "pending",
    },
    stage_retry_count: {},
    ...overrides,
  };
}

describe("complete_stage", () => {
  test("advances to next stage", () => {
    const state = makeState({
      current_stage: 2,
      stage_status: { "1": "completed", "2": "in_progress", "3": "pending" },
    });
    const next = applyTransition(state, { action: "complete_stage", stage: 2 });
    expect(next.stage_status["2"]).toBe("completed");
    expect(next.stage_status["3"]).toBe("in_progress");
    expect(next.current_stage).toBe(3);
  });

  test("rejects wrong stage", () => {
    expect(() =>
      applyTransition(
        { current_stage: 2, stage_status: { "2": "in_progress" } } as OrchestratorState,
        { action: "complete_stage", stage: 1 },
      ),
    ).toThrow("current_stage");
  });

  test("rejects non-in_progress stage", () => {
    expect(() =>
      applyTransition(
        { current_stage: 2, stage_status: { "2": "pending" } } as OrchestratorState,
        { action: "complete_stage", stage: 2 },
      ),
    ).toThrow("in_progress");
  });

  test("does not advance past last stage", () => {
    const state: OrchestratorState = {
      current_stage: 3,
      stage_status: { "1": "completed", "2": "completed", "3": "in_progress" },
    };
    const next = applyTransition(state, { action: "complete_stage", stage: 3 });
    expect(next.stage_status["3"]).toBe("completed");
    expect(next.current_stage).toBe(3);
  });
});

describe("start_stage", () => {
  test("starts a failed stage for retry", () => {
    const state: OrchestratorState = {
      current_stage: 2,
      stage_status: { "1": "completed", "2": "failed" },
      stage_retry_count: { "2": 1 },
    };
    const next = applyTransition(state, { action: "start_stage", stage: 2 });
    expect(next.stage_status["2"]).toBe("in_progress");
    expect(next.current_stage).toBe(2);
  });

  test("rejects if previous stage not completed", () => {
    const state: OrchestratorState = {
      operator_name: "test",
      current_stage: 2,
      stage_status: { "1": "pending", "2": "pending" },
    };
    expect(() =>
      applyTransition(state, { action: "start_stage", stage: 2 }),
    ).toThrow();
  });

  test("rejects starting a completed stage", () => {
    const state = makeState({
      current_stage: 2,
      stage_status: { "1": "completed", "2": "completed", "3": "pending" },
    });
    expect(() =>
      applyTransition(state, { action: "start_stage", stage: 2 }),
    ).toThrow("already completed");
  });
});

describe("fail_stage", () => {
  test("marks stage as failed and increments retry", () => {
    const result = applyTransition(makeState(), { action: "fail_stage", stage: 2 });
    expect(result.stage_status["2"]).toBe("failed");
  });
});

describe("Stage 5 precision pass → complete 5 then complete 6", () => {
  test("orchestrator can complete_stage(5) then complete_stage(6) to reach Stage 7", () => {
    const initial: OrchestratorState = {
      operator_name: "test_op",
      current_stage: 5,
      stage_status: {
        "1": "completed",
        "2": "completed",
        "3": "completed",
        "4": "completed",
        "5": "in_progress",
        "6": "pending",
        "7": "pending",
      },
    };

    // Step 1: complete stage 5 (precision passed)
    const after5 = applyTransition(initial, { action: "complete_stage", stage: 5 });
    expect(after5.stage_status["5"]).toBe("completed");
    expect(after5.stage_status["6"]).toBe("in_progress");
    expect(after5.current_stage).toBe(6);

    // Step 2: immediately complete stage 6 (no precision fix needed)
    const after6 = applyTransition(after5, { action: "complete_stage", stage: 6 });
    expect(after6.stage_status["6"]).toBe("completed");
    expect(after6.stage_status["7"]).toBe("in_progress");
    expect(after6.current_stage).toBe(7);
  });
});
