---
name: pypto-op-plan
description: Requirement planning — structurally-similar example search and planning file (MEMORY.md) initial setup.
---

# PyPTO Complex Kernel — Planning

## Requirements and environment (optional)

**If requirements are unstructured:** structure them into `SPEC.md`, then use it to populate the memory file's task summary.

**If environment issues arise:** confirm with the caller whether to ignore them and continue planning.

## Find structurally similar examples

Search for existing kernels with similar structure:

```bash
grep -rn "<kernel type>" examples/ custom/ models/
```

Record the most relevant example path in the memory file. Note: `examples/` is an **API-usage reference only**, not the production implementation standard — its simplified forms (e.g. `pypto.Tensor([])`) may violate lint / gates. When an example conflicts with lint, lint takes priority; do NOT cite an example to declare lint a false positive.

## Planning file

Create `custom/<operator_name>/MEMORY.md` from the memory template, then seed the planning fields:
- task summary,
- reference locations (example paths),
- blocker list.

