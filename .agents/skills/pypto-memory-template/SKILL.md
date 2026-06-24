---
name: pypto-memory-template
description: Template for the operator-specific memory file (custom/<op>/MEMORY.md). Defines required sections, machine-readable fields, and update cadence.
---

# PyPTO Complex Kernel — Memory Template

This skill contains the memory template that agents copy to `custom/<operator_name>/MEMORY.md` at the start of each kernel implementation.

## Contents

| File | Purpose |
|------|---------|
| **`MEMORY.template.md`** | The actual template — copy to `custom/<op>/MEMORY.md` and fill in |

---

## When to use

- **On dispatch start:** Copy `MEMORY.template.md` to `custom/<operator_name>/MEMORY.md` as the first action
- **Every turn:** Update `active_module`, `modules_pypto_verified`, `current_staged_file`, `next_mandatory_step`
- **Golden authoring:** Fill in Golden function inventory
- **Module design:** Fill in Module decomposition, Module contracts, Staged module files table
- **Per-Phase:** Append to Per-module verification log after each boundary check
- **Cleanup:** Final Golden function inventory cross-check
- **Debugging:** Paste `extract_pypto_calls.py` output, append to Development & debug log

## Key sections in the template

| Section | When to fill | Mandatory? |
|---------|-------------|------------|
| Agent status (YAML) | Every turn | Yes |
| Task summary | On dispatch start | Yes |
| Validation | On dispatch start | Yes |
| Module decomposition + rationale | Module design | Yes |
| Staged module files table | Module design (create), Per-Phase (update) | Yes |
| Per-module verification log | Each Phase M_k pass | Yes |
| API map | On dispatch start | Yes |
| Experience Preflight | Before DESIGN.md (architect preflight creates) | Yes |
| Golden function inventory | Golden authoring (create), Per-Phase (cross-check) | Yes |
| Module contracts | Module design | Yes |
| Design format compliance | Module design | Yes |
| DEBUG_GUIDEBOOK.md §9 pre-write checklist | Before per-Phase coding | Yes |
| Development & debug log | Every error/fix | Yes |
| Human review milestones | Optional | No |
