import { describe, expect, test } from "bun:test";
import { $ } from "bun";
import { PyptoOpLintPlugin } from "../pypto-op-lint";

// ─── tool.execute.before tests ───

test("blocks direct state write via write tool", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);

  const before = plugin["tool.execute.before"];
  await expect(
    before?.(
      { tool: "write" } as never,
      { args: { file_path: "/tmp/qat/.orchestrator_state.json", content: "{}" } } as never,
    ),
  ).rejects.toThrow("禁止直接修改 .orchestrator_state.json");
});

test("blocks direct state write via bash tool", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);

  const before = plugin["tool.execute.before"];
  await expect(
    before?.(
      { tool: "bash" } as never,
      { args: { command: "echo '{}' > /tmp/qat/.orchestrator_state.json" } } as never,
    ),
  ).rejects.toThrow("禁止通过 bash/shell 直接写入");
});

test("no watcher handler registered", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);

  expect(plugin["file.watcher.updated"]).toBeUndefined();
});

test("allows readonly bash commands mentioning state file", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);

  const before = plugin["tool.execute.before"];

  // cat、ls、stat、grep 等只读命令不应被拦截
  const readonlyCommands = [
    "cat /tmp/qat/.orchestrator_state.json",
    "ls -la /tmp/qat/.orchestrator_state.json",
    "stat /tmp/qat/.orchestrator_state.json",
    "grep current_stage /tmp/qat/.orchestrator_state.json",
    "test -f /tmp/qat/.orchestrator_state.json",
  ];

  for (const cmd of readonlyCommands) {
    // 只读命令不应抛出异常
    await expect(
      before?.(
        { tool: "bash" } as never,
        { args: { command: cmd } } as never,
      ),
    ).resolves.toBeUndefined();
  }
});

// ── tool.execute.after (post-edit / post-bash) ──

function createPlugin(shellMock: unknown) {
  return PyptoOpLintPlugin({
    $: shellMock,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
  } as never);
}

/** Helper: build a mock $ that returns the given stdout text. */
function mockShell(stdout: string) {
  const chain = {
    text: async () => stdout,
    quiet: () => chain,
    env: () => chain,
  };
  // Tagged-template invocation: $`...`
  const fn = () => chain;
  return fn as unknown as typeof $;
}

test("post-edit skips non-operator files silently", async () => {
  const plugin = await createPlugin(mockShell(""));
  const after = plugin["tool.execute.after"]!;
  const output: Record<string, unknown> = {};
  // A random file that is not *_impl.py / *_golden.py / test_*.py
  await after(
    { tool: "write", args: { file_path: "/tmp/README.md" } } as never,
    output as never,
  );
  // No output should be appended
  expect(output.output).toBeUndefined();
});

test("post-bash skips non-test commands", async () => {
  let called = false;
  const shell = (() => { called = true; return mockShell("")(""); }) as any;
  const plugin = await (await import("../pypto-op-lint")).PyptoOpLintPlugin({
    $: mockShell(""),
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
  } as never);

  const after = plugin["tool.execute.after"]!;
  const output: Record<string, unknown> = {};
  // A non-test bash command should not trigger post-bash hook
  await after(
    { tool: "bash", args: { command: "ls -la" } } as never,
    output as any,
  );
  expect(output.output).toBeUndefined();
});

test("post-edit appends lint feedback for impl file", async () => {
  const hookResponse = JSON.stringify({
    hookSpecificOutput: {
      hookEventName: "post-edit",
      decision: "allow",
      additionalContext: "[pypto-op-lint] OL01: kernel 缺少 jit 装饰器",
    },
  });
  const mockShellFn = mockShell(hookResponse);
  const plugin = await createPlugin(mockShellFn);

  const after = plugin["tool.execute.after"]!;
  const output: Record<string, unknown> = {};

  await after(
    { tool: "write", args: { file_path: "/workspace/custom/sinh/sinh_impl.py" } } as never,
    output as any,
  );
  expect(typeof output.output).toBe("string");
  expect((output.output as string)).toContain("OL01");
});

test("post-edit blocks on S0/S1 violation", async () => {
  const hookResponse = JSON.stringify({
    hookSpecificOutput: {
      hookEventName: "PostToolUse",
      decision: "block",
      reason: "[S0] 缺少 import pypto",
      additionalContext: "OL07 致命",
    },
  });

  const plugin = await createPlugin(mockShell(hookResponse));
  const after = plugin["tool.execute.after"];
  await expect(
    after?.(
      { tool: "write", args: { file_path: "/tmp/custom/sinh/sinh_impl.py" } } as never,
      {} as never,
    ),
  ).rejects.toThrow();
});

