# Agent Execution Plan — PyPTO Kernel Development

**Reading order:** `principles.md` → this file → `rules.md` → then follow this checklist phase by phase. Skills live flat under `.agents/skills/<skill-id>/`; read each skill's `SKILL.md` directly when this checklist directs you to.

**Do not pre-read all skills.** Read each skill when this checklist directs you to — not before.

Replace `<op>` with the operator name everywhere.

---

## Checklist

You may not proceed past any **⛔ Stage completion criterion** until it passes. Record evidence in `custom/<op>/MEMORY.md`.

---

### Stage 1: Preparation

- [ ] **0.1** Create `custom/<op>/MEMORY.md` from the operator memory template
- [ ] **0.2** List every operation in the target formula (add, matmul, softmax, transpose, etc.)
- [ ] **0.3** Confirm each operation exists in PyPTO → record in plan **API map**

  ```bash
  # Read per-op signature docs
  cat docs/zh/api/operation/pypto-matmul.md
  cat docs/zh/api/operation/pypto-softmax.md
  cat docs/zh/api/operation/pypto-exp.md
  ```

- [ ] **0.4** Search for similar kernel examples → record in plan

  ```bash
  grep -rn "<kernel type>" examples/ custom/ models/
  ```

⛔ **Stage 1 completion:** API map exists in plan; zero `unsupported` rows (or each has a documented workaround).

---

### Stage 2: Build a PyPTO-friendly golden

- [ ] **1.1** Identify the primary reference implementation. (`examples/` is an API-usage reference only, not the production standard; on conflict with lint/gates, lint takes priority — never cite an example to call lint a false positive.)
- [ ] **1.2** Write the **PyPTO-friendly golden** and apply **all** rules below:

| Rule | Rationale |
|------|-----------|
| `.sum(dim)` may stay; be aware of 32-byte alignment issues in PyPTO | reduction alignment caveat |
| No implicit broadcast → explicit `reshape` then op | broadcast rule |
| Name every intermediate (`scores`, `weights`, …) | Debugging |
| Shape comment on every intermediate `# [B, H, T, K]` | `rules.md` rule 6 |
| Mark semantic boundaries with `# --- Module M1: ... ---` | Prepares Stage 4 split |

- [ ] **1.3** Build the **golden operation inventory**:

```
List every operation in the golden (one op per line):
  1. torch.matmul(q, k^T)     # scores  [B,H,T,T]
  2. torch.softmax(scores, -1) # weights [B,H,T,T]
  3. torch.matmul(weights, v)  # out     [B,H,T,V]
  ...
```

→ Record this list in `custom/<op>/MEMORY.md` under **Golden function inventory**.

- [ ] **1.4** Compare normalized golden vs original golden

  ```python
  assert torch.allclose(original_out, normalized_out, rtol=1e-5, atol=1e-5)
  ```

- [ ] **1.5** Freeze golden → set `correctness: golden_ok` in plan

⛔ **Stage 2 completion:** (a) shape comments on all intermediates (b) Golden function inventory exists in plan (c) original-vs-normalized test passes. **Do not enter Stage 3 until all hold.**

---

### Stage 3: Architecture and Design

**Purpose:** Produce `DESIGN.md` with decomposition decision, API mapping, precision routing, tiling strategy, and loop structure. The architect runs a 5-round iterative design process (Round 0 + R1-R4). **Performance target sheet is NOT produced here** — it is produced by `pypto-op-optimizer` at Stage 7 entry (baseline must be measured on actual hardware).

- [ ] **R0** Decomposition decision: Run `python .agents/skills/pypto-op-design/scripts/count_golden_lines.py custom/<op>/<op>_golden.py` for `effective_lines`. Compute `L`, `S`, `O`, `total_complexity = max(L,S,O)`, `module_count = 1 if total<1.3 else min(round(total), ceil(lines/12))`. Classify heavy/light ops (cross-tile communication criterion). If `module_count ≥ 2`, mark `module_count - 1` data-flow breakpoints on the golden. → `DESIGN.md §0`
- [ ] **R1** API mapping and precision routing: Map every golden step to PyPTO API. Document dtype flow and cast points. → `DESIGN.md §1`
- [ ] **R2** Tile shape minimum-viable derivation: **Vec tile** axes in `[16, 64]` (lower below 16 only if UB overflows with rationale; never above 64). **Cube tile** by `quick_ref.md` M-based recommendation table (not constrained by [16, 64]). Verify UB capacity per op (`tile × dtype × tensor_count ≤ UB`, where tensor_count = 2 unary / 3 binary / 4 reduce-expand 保守) and 18000 expansion. → `DESIGN.md §3`
- [ ] **R3** Loop structure and data flow: Dynamic axis analysis, SymbolicScalar checks, full pseudo-code with Layer A–L structure. → `DESIGN.md §4`
- [ ] **R4** Constraint cross-validation: Verify API/tiling/loop/SymbolicScalar compliance. → `DESIGN.md §5`

⛔ **Stage 3 completion:** `DESIGN.md` exists with all mandatory sections (§0, §1, §3, §4, §5, §8, §9); `module_count` is set in §0.3; if §0.3 says `module_count ≥ 2`, §0.5 has `module_count - 1` breakpoints listed; Layers A–L populated; vec tile values in [16, 64] (or rationale for <16); cube tile by M-based table. **Do not enter Stage 4 until DESIGN.md is complete.**

---

### Stage 4: Module decomposition (gated by DESIGN.md §0.3)

- [ ] **2.0 Decomposition gate (mandatory first step)** Read `module_count` from `DESIGN.md §0.3`:
  - If `module_count == 1` (**L0 path**): write `decomposition_level: L0` to MEMORY.md, fill the single-row module table covering the entire kernel, **skip 2.1-2.3**, and proceed directly to Stage 5 (single `<op>_impl.py` deliverable).
  - If `module_count ≥ 2` (**L1 path**): write `decomposition_level: L1` to MEMORY.md and continue with 2.1-2.3.

- [ ] **2.1** *(L1 only)* Read the `module_count - 1` data-flow breakpoints from `DESIGN.md §0.5` and instantiate them as module rows
  - **Do not** invent new breakpoints. They are fixed by architect in §0.5.
  - Apply module boundary rules R1-R6: each module must contain ≥1 heavy op; light ops + view/assemble merge to neighbors; aim for ≈1 complexity unit per module.
  - Define each module's input/output tensor names, shapes, dtypes
  - → Record in plan **Module decomposition** (with heavy ops, light ops merged, CU estimate columns) + **Module contracts**

- [ ] **2.2** *(L1 only)* Fix staged file sequence → fill **Staged module files** table in plan

  ```
  <op>_module1.py     → M1 only
  <op>_module12.py    → M1 + M2
  ...
  <op>_module1…N.py   → all modules = complete kernel
  ```

- [ ] **2.3** *(both L0 and L1)* Complete the pre-write debug checklist in plan (MEMORY.md `DEBUG §9 pre-write` section)

⛔ **Stage 4 completion (L0)**: `decomposition_level: L0` + single-row module table + 2.3 done in plan. **Skip staged file table** (fill with `N/A — L0 path`).
⛔ **Stage 4 completion (L1)**: `decomposition_level: L1` + module decomposition (with heavy/light/CU columns) + contracts + staged file table + 2.3 done in plan.

---

### Stage 5: Implement (path-conditional)

#### Stage 5 (L0 path — `module_count == 1`)

When `decomposition_level: L0` in MEMORY.md, Stage 5 is **one-shot**: produce `custom/<op>/<op>_impl.py` directly. No staged file chain, no freeze protocol, no per-module loop.

- [ ] **3L0.1** Consult the pre-write DEBUG §9 subsections matching the kernel's operations
- [ ] **3L0.2** Create `custom/<op>/<op>_impl.py` from the impl template
- [ ] **3L0.3** AST lint runs automatically via post-edit hook (OL01-OL54) — fix until zero errors
- [ ] **3L0.4** **Cross-check Golden function inventory vs PyPTO** — every op marked ✅; **do not run tests until all ✅**
- [ ] **3L0.5** Verifier produces `test_<op>.py` and runs E2E:
      `PYTHONPATH=.agents/skills/pypto-op-verify python custom/<op>/test_<op>.py`
- [ ] **3L0.6** Confirm no pypto-op-lint FAIL (layout / structure rules — OL44/OL45/OL48/OL52/OL57 — run automatically on file write and at the gate; no separate script)
- [ ] **3L0.7** Coder writes `README.md`

⛔ **Stage 5 completion (L0)**: (a) `<op>_impl.py`, `test_<op>.py`, `README.md` exist (b) E2E `all_close: true` on all outputs (c) Golden inventory 100% ✅ (d) no pypto-op-lint FAIL. Skip the per-Phase loop and 5-cleanup below.

#### Stage 5 (L1 path — `module_count ≥ 2`): Implement modules (repeat for each M_k)

**One module at a time. Do not advance until the current module passes.**

- [ ] **3.1** Consult the pre-write DEBUG §9 subsections matching the kernel's operations
- [ ] **3.2** Create `custom/<op>/modules/<op>_module<suffix>_impl.py` from the impl template
  - Golden for this cumulative scope
  - PyPTO: **one** `@pypto.frontend.jit`; stub later modules with golden-fed tensors
  - Runner: `if __name__ == "__main__":` + `detailed_tensor_compare`
- [ ] **3.3** AST lint runs automatically via post-edit hook (OL01-OL54) — fix until zero errors
- [ ] **3.4** **Cross-check Golden function inventory vs PyPTO**

  ```
  Golden function inventory (from Stage 2.3):
    1. matmul(q, k^T)        → ✅ pypto.matmul(q, k, dtype, b_trans=True)  L.42
    2. softmax(scores, -1)    → ✅ pypto.softmax(scores, dim=-1)           L.45
    3. matmul(weights, v)     → ❌ not implemented — skipping causes precision error
  ```

  **Do not run tests until every operation in M_k's scope has ✅.**

- [ ] **3.5** Run tests

  ```bash
  python custom/<op>/modules/<op>_module<suffix>_impl.py
  ```

- [ ] **3.6** Confirm no pypto-op-lint FAIL (layout / structure rules run automatically on file write and at the gate — no separate script)

⛔ **Phase M_k completion (Stage 5):** (a) `detailed_tensor_compare` → `all_close: true` on **all** outputs (b) Golden inventory: all ops in M_k scope ✅ (c) no pypto-op-lint FAIL (d) Per-module verification log updated. **Do not start M_{k+1} until all hold.**

- [ ] **3.7** Freeze M_k → update plan: append to `modules_pypto_verified`, set `active_module: M_{k+1}`

**After Phase M_k completion:** return to 3.1 for the next module, or go to **3.8 Stage 5 cleanup** when all modules are done.

---

### Stage 5 cleanup (after the last verified Phase M_k)

Once every Phase M_k has passed verification, the coder runs a one-shot cleanup pass and the verifier writes the E2E test. This finishes Stage 5; there is no separate Integration stage.

- [ ] **3.8** Coder copies the final cumulative module file `modules/<op>_module1...N_impl.py` to `<op>_impl.py`. Keep the algorithm and function bodies; only rename / clean imports / remove debug scaffolding so the file reads as a standalone production kernel.
- [ ] **3.9** Verifier writes `test_<op>.py` (E2E validation runner). Compare **every** output tensor with `detailed_tensor_compare` (single-output-only compare is forbidden).
- [ ] **3.10** Coder writes `README.md` (Chinese; op overview + usage example + run command + known constraints).
- [ ] **3.11** **Final Golden function inventory cross-check**

  Line-by-line: golden operation list vs final PyPTO kernel.
  **If any ❌ remains, do not run E2E — implement first.**

- [ ] **3.12** Run E2E

  ```bash
  python custom/<op>/test_<op>.py
  ```

⛔ **Stage 5 completion:** (a) all Phase M_k verified (b) `<op>_impl.py`, `test_<op>.py`, `README.md` exist (c) E2E `all_close: true` on all outputs (d) Golden inventory 100% ✅.

---

### Stage 6: Final structural rules

- [ ] Post-edit lint hook (OL01-OL57) → zero errors
- [ ] No Python `for`/`while` in JIT code; iteration via `pypto.loop` (OL45 host wrapper, OL57 JIT graph)
- [ ] `set_vec_tile_shapes` with valid tile dimensions per doc
- [ ] `set_cube_tile_shapes` before matmul; each m/k/n is `[L0, L1]`, 0<L0<=L1, L1%L0==0 (OL48)
- [ ] Write-back via `output[:] = expr` or `pypto.assemble(...)`

⛔ **Stage 6 completion:** (a) Post-edit lint hook (OL01-OL57) zero errors (b) no pypto-op-lint FAIL.

---

### Stage 7: Optimization (only after correctness)

- [ ] **7.0 Performance target sheet (entry, mandatory)**: Before loading any `tune-*` sub-skill, `pypto-op-optimizer` produces the Performance target sheet in `custom/<op>/MEMORY.md` — concrete **baseline** (measured on representative P0 shape via `perf-analyzer/scripts/analyze_perf.py`), **target** (from SPEC.md performance budget or torch-eager comparable), **required speedup**. The tile shape upper bound (≤ 64 from Stage 3) is also lifted here based on profiling.
- [ ] **7.1 Frontend / Swimlane / Incore phases**: Follow the 3-stage tune skill. Roll back immediately if correctness regresses.

---

## When precision errors appear (required order)

1. Open **Golden function inventory**
2. Line-by-line vs PyPTO kernel → find missing or wrong ops
3. Run `extract_pypto_calls.py` on the kernel and reconcile with golden
4. If still unclear → escalate to the debugger via verifier (failure_category: `precision`)

---

## Required MEMORY.md sections

| Section | When to update |
|---------|----------------|
| API map | Stage 1 |
| **Golden function inventory** | Stage 2 (create); Stage 5/4 (append cross-check results) |
| Module decomposition + rationale | Stage 4 |
| Module contracts | Stage 4 |
| Staged module files table | Stage 4 (create) → Stage 5 (check each milestone) |
| Per-module verification log | Each Phase M_k pass in Stage 5 |
| DEBUG §9 pre-write checklist | Before Stage 5 |
| Development & debug log | Every error and fix |


