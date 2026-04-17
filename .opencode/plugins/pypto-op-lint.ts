import type { Plugin } from "@opencode-ai/plugin";
import path from "node:path";

type HookOutput = {
  hookSpecificOutput?: {
    additionalContext?: string;
    decision?: "allow" | "block";
    reason?: string;
  };
};

function appendMessage(output: { output?: string }, message: string): void {
  output.output = `${output.output ?? ""}\n\n${message}`.trim();
}

function formatPluginError(scope: string, error: unknown): string {
  const detail = error instanceof Error ? error.message : String(error);
  return `[pypto-op-lint plugin-error] ${scope} 自动检查失败：${detail}`;
}

function parseHookOutput(raw: string): { additionalContext: string; decision: "allow" | "block"; reason: string } {
  if (!raw.trim()) return { additionalContext: "", decision: "allow", reason: "" };

  try {
    const parsed = JSON.parse(raw) as HookOutput;
    return {
      additionalContext: parsed.hookSpecificOutput?.additionalContext ?? "",
      decision: parsed.hookSpecificOutput?.decision ?? "allow",
      reason: parsed.hookSpecificOutput?.reason ?? "",
    };
  } catch (error) {
    return {
      additionalContext: formatPluginError("post-edit 输出解析", error),
      decision: "allow",
      reason: "",
    };
  }
}

function resolvePath(baseDir: string, filePath: string): string {
  if (!filePath) return "";
  return path.isAbsolute(filePath) ? filePath : path.resolve(baseDir, filePath);
}

function isStateWriteCommand(command: string): boolean {
  if (!command.includes(".orchestrator_state.json")) return false;
  // Split compound commands (&&, ||, ;) and only inspect segments that mention the state file.
  const segments = command.split(/&&|\|\||;/).map(s => s.trim());
  const stateSegs = segments.filter(s => s.includes(".orchestrator_state.json"));
  // If all state-file segments start with a read-only verb, allow the command.
  const readOnly = /^(cat|ls|stat|test|head|tail|grep|wc|file|find|read)\b/;
  if (stateSegs.length > 0 && stateSegs.every(s => readOnly.test(s.replace(/^sudo\s+/, "")))) {
    return false;
  }
  // Otherwise treat as a write attempt.
  return true;
}

export const PyptoOpLintPlugin: Plugin = async (input) => {
  const $ = input.$;
  const client = input.client;
  const baseDir = input.worktree || input.directory || process.cwd();
  const lintScript = new URL(
    "../../.agents/hooks/pypto-op-lint/pypto_op_lint.py",
    import.meta.url,
  ).pathname;

  async function execHookJson(hook: string, payload: unknown): Promise<string> {
    return await $`python3 ${lintScript} --hook ${hook}`
      .env({
        PYPTO_OP_LINT_HOOK_INPUT: JSON.stringify(payload),
      })
      .quiet()
      .text();
  }

  return {
    "tool.execute.after": async ({ tool, args }, output) => {
      const filePath: string = args?.file_path ?? args?.path ?? "";
      if (
        (tool === "write" || tool === "edit" || tool === "multiedit") &&
        (filePath.endsWith("_impl.py") ||
         filePath.endsWith("_golden.py") ||
         /test_\w+\.py$/.test(filePath))
      ) {
        try {
          const raw = await execHookJson("post-edit", {
            tool_input: { file_path: filePath },
          });
          const parsed = parseHookOutput(raw);
          if (parsed.additionalContext) appendMessage(output, parsed.additionalContext);
          if (parsed.decision === "block") {
            throw new Error(parsed.reason || "[pypto-op-lint] 产物写入后门禁未通过");
          }
        } catch (error) {
          // post-edit 为强约束：违规时应阻断本次工具调用
          if (error instanceof Error) {
            throw error;
          }
          throw new Error(formatPluginError("post-edit", error));
        }
        return;
      }

      if (
        (tool === "bash" || tool === "shell") &&
        /python3?\s+.*test_\w+\.py/.test(args?.command ?? "")
      ) {
        try {
          // OpenCode bash tool result 的输出在 output 字段（stdout+stderr 合并），
          // 没有单独的 stderr/exit_code。逐字段安全提取，避免映射错误导致 marker 丢失。
          const result = output as Record<string, unknown>;
          const combinedOutput = String(result.output ?? result.stdout ?? result.text ?? "");
          const stderrText = String(result.stderr ?? "");
          const exitCode = typeof result.exit_code === "number" ? result.exit_code
            : typeof result.code === "number" ? result.code
            : typeof result.exitCode === "number" ? result.exitCode
            : 0;
          const raw = await execHookJson("post-bash", {
            tool_input: { command: args?.command ?? "" },
            tool_result: {
              stdout: combinedOutput,
              stderr: stderrText,
              exit_code: exitCode,
            },
          });
          const context = parseHookOutput(raw).additionalContext;
          if (context) appendMessage(output, context);
        } catch (error) {
          appendMessage(output, formatPluginError("post-bash", error));
        }
      }

      return;
    },

    "tool.execute.before": async ({ tool }, output) => {
      const command = String(output.args?.command ?? "");
      if ((tool === "bash" || tool === "shell") && isStateWriteCommand(command)) {
        throw new Error(
          "[pypto-op-lint] 禁止通过 bash/shell 直接写入 .orchestrator_state.json。"
          + "请使用 state_transition 工具更新状态文件。",
        );
      }

      if (tool !== "write" && tool !== "edit" && tool !== "multiedit") return;

      const filePath = output.args?.file_path ?? output.args?.path ?? "";
      const absPath = resolvePath(baseDir, filePath);
      if (!absPath) return;

      if (absPath.endsWith(`${path.sep}.orchestrator_state.json`)) {
        throw new Error(
          "[pypto-op-lint] 禁止直接修改 .orchestrator_state.json。"
          + "请使用 state_transition 工具进行阶段状态迁移。",
        );
      }

      if (!absPath.endsWith("_impl.py")) return;

      // pre-edit-backup: Stage 6 编辑 impl 前 git auto-commit
      try {
        await execHookJson("pre-edit-backup", {
          tool_input: { file_path: absPath },
        });
      } catch (error) {
        await client.app.log({
          body: {
            service: "pypto-op-lint",
            level: "warn",
            message: formatPluginError("pre-edit-backup", error),
            extra: { tool, filePath },
          },
        });
      }
    },

  };
};
