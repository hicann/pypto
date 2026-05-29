---
name: pypto-op-plan
description: Stage 1 requirement planning and triage. Covers complex-kernel workflow triage, API availability checking, structurally-similar example search, and planning file (MEMORY.md) initial setup. Golden preparation rules live in `pypto-golden-generate` §13 (mathematician's skill).
---

# PyPTO Complex Kernel — Stage 1 Planning

## Contents

| File | Purpose |
|------|---------|
| **This file (SKILL.md)** | Stage 1 triage + planning file setup workflow |

For PyPTO API lookups during planning, read `docs/zh/api/operation/index.md` for the op category index and `docs/zh/api/operation/pypto-<op>.md` for per-op signatures. For semantic exploration, load skill `pypto-api-explore` or grep `docs/zh/api/operation/`.

## Triage and Planning

Before any code generation, decide whether the complex kernel workflow applies.

Use this workflow if any of the following are true:
- backward kernel,
- recurrent or stateful kernel,
- scan-like kernel,
- multiple dependent math stages,
- multiple layout transitions,
- multiple reductions,
- nested loop structure,
- previous failed attempts,
- temptation to "just implement the whole thing now".

### Step 0.1: API availability check

Decompose the formula into atomic operations and verify each one exists in PyPTO:

```bash
# Browse all PyPTO ops by category
cat docs/zh/api/operation/index.md

# Look up specific API signatures and constraints
grep -l '<op1>\|<op2>' docs/zh/api/operation/*.md
cat docs/zh/api/operation/pypto-<op1>.md
```

For any operation not found by name, search by category or semantically:

```bash
grep -l "<relevant_category>" docs/zh/api/operation/*.md
grep -rn "<formula step keyword>" docs/zh/api/operation/
```

Mark each formula step as **supported**, **needs substitute**, or **unsupported** before proceeding. Do not begin implementation for unsupported operations without a confirmed substitute.

### Subskill delegation: requirements and environment (optional)

**If requirements are unstructured:** Read skill `pypto-intent-understand` (SKILL.md auto-loads) and follow its workflow to produce a `SPEC.md`. Use the SPEC.md output to populate the memory file's task summary and API map.

**If environment issues arise:** Read skill `pypto-environment-setup` (SKILL.md auto-loads) and follow its workflow to diagnose and fix CANN, torch_npu, or build-chain problems.

### Subskill delegation: enhanced API exploration

After completing Step 0.1, if deeper constraint verification is needed (3-layer validation, reference implementation search across `models/` and `examples/`), read skill `pypto-api-explore` (SKILL.md auto-loads) and produce an `API_REPORT.md`. Merge the API_REPORT.md findings into the memory's API map section.

### Step 0.2: Find structurally similar examples

Search for existing kernels with similar structure:

```bash
grep -rn "<kernel type>" examples/ custom/ models/
```

Record the most relevant example path in the memory file. Note: `examples/` is an **API-usage reference only**, not the production implementation standard — its simplified forms (e.g. `pypto.Tensor([])`) may violate lint / gates. When an example conflicts with lint, lint takes priority; do NOT cite an example to declare lint a false positive.

### Mandatory planning file

Create `custom/<operator_name>/MEMORY.md` first, then continue implementation immediately.

The memory must contain:
- task summary,
- reference locations (including example paths from Step 0.2),
- API availability map (from Step 0.1),
- normalized golden status,
- module list,
- module contracts,
- frozen items,
- attempt history,
- integration status,
- optimization status,
- blocker list,
- **design format compliance:** which layers from skill `pypto-op-develop`'s `references/pypto-kernel-design-format.md` (A–L) apply, which are omitted and why,
- **`active_module`** and **`modules_pypto_verified`** (see skill `pypto-orchestration-manual`'s `references/rules.md` → Module-at-a-time enforcement).

