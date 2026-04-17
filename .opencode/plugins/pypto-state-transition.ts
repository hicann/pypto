import { type Plugin, tool } from "@opencode-ai/plugin";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { applyTransition, type OrchestratorState, type TransitionAction } from "./lib/state-transition-core";

type GateFinding = {
  rule_id: string;
  severity: string;
  status: string;
  message: string;
  file: string;
};

type GateSummary = {
  warnCount: number;
  infoCount: number;
  /** FAIL findings included when gate blocks — gives the agent actionable detail. */
  failFindings: GateFinding[];
};

const ALLOWED_ACTIONS = new Set<TransitionAction>([
  "init",
  "start_stage",
  "complete_stage",
  "fail_stage",
]);

/** 仅允许以下 agent 调用 state_transition 工具 */
const ALLOWED_AGENTS = new Set<string>([
  "pypto-op-orchestrator",
]);

function parseState(content: string): OrchestratorState {
  const parsed = JSON.parse(content);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("invalid orchestrator state file content");
  }
  return parsed as OrchestratorState;
}

function parseGateSummary(raw: string): GateSummary {
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const summary = parsed.summary;
    if (!summary || typeof summary !== "object" || Array.isArray(summary)) {
      return { warnCount: 0, infoCount: 0, failFindings: [] };
    }
    const warnCount = Number((summary as Record<string, unknown>).warn);
    const infoCount = Number((summary as Record<string, unknown>).info);

    // Extract FAIL findings for detailed error reporting
    const failFindings: GateFinding[] = [];
    const findings = parsed.findings;
    if (Array.isArray(findings)) {
      for (const f of findings) {
        if (f && typeof f === "object" && (f as Record<string, unknown>).status === "FAIL") {
          failFindings.push({
            rule_id: String((f as Record<string, unknown>).rule_id ?? ""),
            severity: String((f as Record<string, unknown>).severity ?? ""),
            status: "FAIL",
            message: String((f as Record<string, unknown>).message ?? ""),
            file: String((f as Record<string, unknown>).file ?? ""),
          });
        }
      }
    }

    return {
      warnCount: Number.isFinite(warnCount) ? warnCount : 0,
      infoCount: Number.isFinite(infoCount) ? infoCount : 0,
      failFindings,
    };
  } catch {
    return { warnCount: 0, infoCount: 0, failFindings: [] };
  }
}

function buildInitialState(opDir: string): OrchestratorState {
  return {
    operator_name: path.basename(opDir),
    current_stage: 1,
    stage_status: {
      "1": "pending",
      "2": "pending",
      "3": "pending",
      "4": "pending",
      "5": "pending",
      "6": "pending",
      "7": "pending",
    },
    stage_retry_count: {
      "1": 0,
      "2": 0,
      "3": 0,
      "4": 0,
      "5": 0,
      "6": 0,
      "7": 0,
    },
    perf_iteration: {
      count: 0,
      last_improvement: 0,
      consecutive_no_improvement: 0,
    },
    last_updated: new Date().toISOString(),
  };
}

function readStateOrInit(statePath: string): OrchestratorState {
  if (!fs.existsSync(statePath)) {
    return buildInitialState(path.dirname(statePath));
  }
  const raw = fs.readFileSync(statePath, "utf8");
  return parseState(raw);
}

function writeStateAtomically(statePath: string, state: OrchestratorState): void {
  const tmpPath = `${statePath}.tmp`;
  fs.writeFileSync(tmpPath, `${JSON.stringify(state, null, 2)}\n`, "utf8");
  fs.renameSync(tmpPath, statePath);
}

/** Stage 1 完成时记录 SPEC.md 内容 hash，后续阶段用于冻结校验。 */
const SPEC_HASH_KEY = "spec_md_hash";

function computeFileHash(filePath: string): string | null {
  if (!fs.existsSync(filePath)) return null;
  const content = fs.readFileSync(filePath, "utf8");
  return crypto.createHash("sha256").update(content).digest("hex");
}

export const PyptoStateTransitionPlugin: Plugin = async (input) => {
  const $ = input.$;
  const client = input.client;
  const baseDir = input.worktree || input.directory || process.cwd();
  const lintScript = new URL(
    "../../.agents/hooks/pypto-op-lint/pypto_op_lint.py",
    import.meta.url,
  ).pathname;

  async function runGateIfNeeded(opDir: string, action: TransitionAction, stage: number): Promise<GateSummary> {
    if (action !== "complete_stage") return { warnCount: 0, infoCount: 0, failFindings: [] };

    // Use nothrow() so stdout is available even when exit code is non-zero.
    // The lint script writes findings JSON to stdout and exits 2 on S1 FAIL.
    const result = await $`python3 ${lintScript} --check-gate --op-dir ${opDir} --stage ${stage}`
      .cwd(baseDir).quiet().nothrow();
    const raw = result.stdout.toString();
    const summary = parseGateSummary(raw);

    if (result.exitCode !== 0) {
      // Build detailed error lines from FAIL findings
      const details = summary.failFindings
        .map((f) => `  [${f.rule_id}][${f.severity}] ${f.message}${f.file ? ` (${f.file})` : ""}`)
        .join("\n");
      throw new Error(
        details
          ? `以下规则违规导致门禁阻断：\n${details}`
          : `lint 脚本返回 exit code ${result.exitCode}，但未解析到具体违规项`,
      );
    }

    return summary;
  }

  return {
    tool: {
      state_transition: tool({
        description: "Safely transition .orchestrator_state.json with stage gate enforcement. Actions: init (stage=1 only, first call), start_stage (set stage to in_progress — for init or retry after failure), complete_stage (gate check + mark done + auto-advance to next stage), fail_stage (mark failed + increment retry).",
        args: {
          opDir: tool.schema.string(),
          action: tool.schema.string(),
          stage: tool.schema.number(),
          reason: tool.schema.string().optional(),
        },
        execute: async (args, context) => {
          // ── Agent 权限校验：仅 pypto-op-orchestrator 可调用 ──
          const callerAgent = context?.agent ?? "";
          if (!ALLOWED_AGENTS.has(callerAgent)) {
            throw new Error(
              `permission denied: state_transition is restricted to pypto-op-orchestrator, ` +
              `but was called by "${callerAgent || "(unknown)"}". ` +
              `Subagents must not modify .orchestrator_state.json; return stage results to the orchestrator instead.`,
            );
          }

          const action = String(args.action) as TransitionAction;
          if (!ALLOWED_ACTIONS.has(action)) {
            throw new Error(`unsupported action: ${args.action}`);
          }

          const opDir = path.isAbsolute(args.opDir)
            ? args.opDir
            : path.resolve(baseDir, args.opDir);
          const statePath = path.join(opDir, ".orchestrator_state.json");
          const prevState = readStateOrInit(statePath);

          let gateSummary: GateSummary = { warnCount: 0, infoCount: 0, failFindings: [] };
          try {
            gateSummary = await runGateIfNeeded(opDir, action, args.stage);
          } catch (error) {
            const detail = error instanceof Error ? error.message : String(error);
            throw new Error(
              `[pypto-op-lint] 交付门禁阻断（ERROR 级违规）：action=${action}, stage=${args.stage}, op_dir=${opDir}\n${detail}`,
            );
          }

          // ── SPEC.md 冻结校验 ──
          // complete_stage(1) 时记录 SPEC.md hash；
          // complete_stage(N>=3) 时校验 hash 不变，防止 subagent 绕过 hook 修改 SPEC。
          const specPath = path.join(opDir, "SPEC.md");
          if (action === "complete_stage" && args.stage === 1) {
            const hash = computeFileHash(specPath);
            if (hash) {
              prevState[SPEC_HASH_KEY] = hash;
            }
          }
          if (action === "complete_stage" && args.stage >= 3) {
            const savedHash = prevState[SPEC_HASH_KEY];
            if (typeof savedHash === "string") {
              const currentHash = computeFileHash(specPath);
              if (currentHash && currentHash !== savedHash) {
                throw new Error(
                  `[pypto-op-lint] SPEC.md 冻结违规：SPEC.md 在 Stage 2 完成后被修改。` +
                  `当前 hash=${currentHash.slice(0, 12)}… 与记录 hash=${savedHash.slice(0, 12)}… 不一致。` +
                  `如需变更需求规格，应通过 fail_stage 回退到 Stage 1 重新审核。`,
                );
              }
            }
          }

          const nextState = applyTransition(prevState, {
            action,
            stage: args.stage,
          });

          // 将 SPEC hash 持久化到 state（init/complete_stage(1) 时写入）
          if (prevState[SPEC_HASH_KEY]) {
            nextState[SPEC_HASH_KEY] = prevState[SPEC_HASH_KEY];
          }

          writeStateAtomically(statePath, nextState);

          if (gateSummary.warnCount > 0 || gateSummary.infoCount > 0) {
            // 记录日志失败不应影响状态迁移结果（避免“落盘成功但工具返回异常”）。
            try {
              await client.app.log({
                body: {
                  service: "pypto-state-transition",
                  level: "info",
                  message: `[state_transition] ${action} stage=${args.stage} completed`,
                  extra: {
                    opDir,
                    reason: args.reason ?? "",
                    warnCount: gateSummary.warnCount,
                    infoCount: gateSummary.infoCount,
                  },
                },
              });
            } catch {
              // no-op
            }
          }

          return JSON.stringify({
            ok: true,
            action,
            stage: args.stage,
            next_stage: nextState.current_stage,
            statePath,
            warnCount: gateSummary.warnCount,
            infoCount: gateSummary.infoCount,
          });
        },
      }),
    },
  };
};
