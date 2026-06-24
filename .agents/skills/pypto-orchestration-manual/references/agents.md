# Agent Team — Dispatch Contract

This file is the orchestrator's roster of sub-agents: who owns which stage,
what each produces, the gate criterion the verifier checks before
`complete_stage`, and the handoff rule that decides the next dispatch.

Sub-agent-internal skill loading is **not** described here. Each sub-agent
declares its own active skills in its `.opencode/agents/<name>.md` →
**Mandatory reads** section. The orchestrator dispatches by name and does
not inspect what skills each agent loads.

> **Reading order for the orchestrator:** `principles.md` → this file →
> `agent-plan.md` → `rules.md`.

---

## Sub-agent roster

| # | Agent | Stage owned |
|---|-------|-------------|
| 1 | `pypto-op-planner` | Stage 1 |
| 2 | `pypto-op-mathematician` | Stage 2 |
| 3 | `pypto-op-architect` | Stage 3 |
| 4 | `pypto-op-designer` | Stage 4 |
| 5 | `pypto-op-coder` | Stage 5 (per-module + cleanup) |
| 6 | `pypto-op-verifier` | Stage 4 scaffolding (Step B) + Stage 5 phase scaffolding + Stage 5 composition + Stage 6 E2E + Stage 7 regression |
| 7 | `pypto-op-debugger` | Stage 5 failure investigation |
| 8 | `pypto-op-optimizer` | Stage 7 |

**Separation of concerns in Stage 5 phase failure:** verifier = judge
(pass / fail with `failure_category`); debugger = investigator (proposes a
patch in `MEMORY.md`); coder = the only agent that writes production
kernel code. The orchestrator never edits kernel code itself.

---

## Per-agent dispatch contract

### 1. pypto-op-planner — Stage 1

- **Deliverables:** `custom/<op>/SPEC.md`, `custom/<op>/API_REPORT.md`,
  `custom/<op>/MEMORY.md` seeded with the API map.
- **Gate:** API map has zero `unsupported` rows (or each has a documented
  workaround in the row).
- **Handoff:** Returns to orchestrator → dispatch `pypto-op-mathematician`.

### 2. pypto-op-mathematician — Stage 2

- **Deliverables:** `custom/<op>/golden.py` (PyPTO-friendly form),
  `MEMORY.md` → **Golden function inventory**,
  `custom/<op>/GOLDEN_PERF_REPORT.md` (NPU profiling via
  `pypto-golden-generate/scripts/profile_golden.py` §15).
- **Gate:** shape comments on all intermediates; inventory recorded;
  `allclose(original, normalized)` passes; **`GOLDEN_PERF_REPORT.md` exists with Op Performance section** (mandatory — generated via `pypto-golden-generate/scripts/profile_golden.py` §15; do NOT defer to orchestrator).
- **Handoff:** Returns to orchestrator → dispatch `pypto-op-architect`.

### 3. pypto-op-architect — Stage 3

- **Deliverables:** `custom/<op>/DESIGN.md` (§0 decomposition decision,
  §1 API mapping + precision routing, §3 tiling, §4 loop / data flow, §5
  cross-validation).
- **Gate:** §0.3 `module_count` set; if `module_count ≥ 2`, §0.5 lists
  `module_count - 1` data-flow breakpoints; Layers A–L populated; vec
  tile axes ∈ [16, 64] (or rationale for going below 16); cube tile by
  the M-based recommendation table. **Performance target sheet is NOT
  produced here** — it is the optimizer's Stage 7 entry deliverable.
- **Handoff:** Returns to orchestrator → dispatch `pypto-op-designer`.

### 4. pypto-op-designer — Stage 4

- **Deliverables:** `custom/<op>/module_interfaces.yaml`, `MEMORY.md` →
  **Module decomposition**, **Module contracts**, **Staged module files**.
- **Gate (Step 1, designer handoff):** `state_transition(submit_design)`
  runs the DESIGN-side lint gate (OL12 structural sections; OL55
  `pypto.<attr>` existence in fenced code blocks). FAIL throws →
  re-dispatch designer with the failure details.
- **Handoff:** On `submit_design` PASS → dispatch `pypto-op-verifier` in
  **Stage 4 scaffolding mode (Step B)** to produce the adversarial harness.

### 5. pypto-op-coder — Stage 5

Stage 5 is a per-module loop. The coder is dispatched once per phase
M_k, and once more for the cleanup step after the last phase.

- **Per-phase deliverable (M_k):**
  `custom/<op>/modules/<op>_module<suffix_k>_impl.py` — one new file,
  exactly the cumulative scope through M_k.
- **Per-phase gate:** `state_transition(submit_for_verify, phase=M_k)`
  runs the phase-scoped lint gate. FAIL throws → patch the impl and
  re-submit; the verifier is NOT dispatched until lint passes.
- **Cleanup deliverable (after the last verified phase):**
  `custom/<op>/<op>_impl.py` (integrated kernel) + `custom/<op>/README.md`.
- **Handoff:** Returns per-phase. On verifier FAIL → debugger investigates
  → coder applies the proposed patch → re-`submit_for_verify`.

**Forbidden during a coder dispatch:** producing later staged files,
loading any debug sub-skill, calling `pypto_op_lint.py` manually.

### 6. pypto-op-verifier — Stages 4, 5, 6, 7 (judge-only)

Verifier is dispatched in distinct **modes**, never as a general "review
everything" agent. The orchestrator chooses the mode per dispatch.

- **Stage 4 scaffolding mode (Step B):**
  Produce `custom/<op>/eval/test_inputs.py`,
  `custom/<op>/eval/adversarial_suite.json` (≥ 2 cases per level L1–L5),
  `custom/<op>/eval/adversarial_runner.py` (with `--up-to-module`). Run
  `--self-test`. These are operator-level; per-module artifacts are NOT
  produced here.
- **Stage 5 Phase scaffolding mode (per M_k):**
  Produce `modules/<op>_module<suffix_k>_golden.py` (derived from
  `module_interfaces.yaml` + `<op>_golden.py`, never from the coder's
  impl), `modules/test_<op>_module<suffix_k>.py`, and run prefix-eval at
  `--up-to-module k`.
- **Stage 5 composition verification mode:** After the last phase
  completes, confirm the cumulative module<suffix_N> golden reproduces
  `<op>_golden.py` under the operator's composition_verification seeds /
  shapes / tolerances.
- **Stage 6 final E2E mode:** Run `detailed_tensor_compare` on every
  output of `<op>_impl.py` plus the final layout / structure lint gate.
- **Stage 7 regression mode:** Re-run E2E + layout + perf-analyzer delta
  for each optimizer change. Reports adopt / regression / no-gain.

**Verdict format — always one of:**

```
Stage N completion passed for <scope>. Evidence: <memory row pointer>.
```

```
Stage N completion FAILED for <scope>. failure_category: <cat>.
Failing file: <path>. Evidence: <memory row + log excerpt>.
Dispatch @pypto-op-debugger.
```

**`failure_category` enum** (orchestrator routes the debugger on this):
`precision`, `aicore`, `host_crash`, `workspace_overlap`, `oom`,
`structure`, `layout`, `tile_shape`, `other`.

**Forbidden:** loading debug sub-skills, editing kernel code, bisecting,
retrying checks without a prior debugger → coder cycle.

### 7. pypto-op-debugger — Stage 5 failure investigation

- **Dispatch input:** one failing file path + the `failure_category`
  reported by the verifier.
- **Deliverable:** a patch proposal logged to `MEMORY.md` →
  **Development & debug log** with (a) file + line range, (b) current
  snippet, (c) proposed snippet, (d) expected effect on the failing
  verifier check. Scratch / diagnostic files may be written under
  `custom/<op>/_debug/`.
- **Forbidden:** writing production kernel code directly; modifying any
  file under `custom/<op>/` outside `_debug/`.
- **Iteration cap:** 10 fix / re-verify cycles per module. On the 11th
  failure of the same module, stop and surface the blocker to the
  orchestrator with all collected evidence.

### 8. pypto-op-optimizer — Stage 7

- **Activation precondition:** Stage 6 completion has passed (E2E
  `all_close: true` on every output; layout check exit 0). The optimizer
  is dormant until then.
- **Entry deliverable:** the **Performance target sheet** in
  `MEMORY.md` — concrete baseline (measured on a representative P0
  shape), target (from `SPEC.md` performance budget or torch-eager
  comparable), required speedup. The tile-shape upper bound (≤ 64 from
  Stage 3) is lifted here based on profiling.
- **Per-change deliverable:** a kernel modification + the verifier's
  regression report logged to `MEMORY.md`.

**Regression loop (per optimizer change):**

```
optimizer: apply change N
  ↓
verifier (Stage 7 mode):
  (1) detailed_tensor_compare → all_close?
  (2) layout / structure lint  → exit 0?
  (3) perf-analyzer            → delta vs baseline
  ↓
optimizer:
  - regression       → rollback, log, try next idea
  - no gain          → log, try next idea
  - gain, no regress → adopt, move on
  - target reached   → stop, hand back to orchestrator
```
