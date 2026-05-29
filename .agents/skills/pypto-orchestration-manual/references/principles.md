# Behavioral Principles

Foundational guidelines for all skill execution in this library.
Derived from [Karpathy-inspired coding guidelines](https://github.com/forrestchang/andrej-karpathy-skills).

**These principles apply to every skill. rules.md adds PyPTO-specific enforcement on top.**

---

## 1. Think Before pypto-op-coder

**Do not assume. Do not hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain about an API, tiling strategy, or op semantic, stop and ask.
- If multiple interpretations exist (e.g., transpose convention, reduction axis, broadcast shape), present them. Do not pick silently.
- If a simpler approach exists, say so. Push back on unnecessary complexity.
- If something is unclear, stop. Name what is confusing. Ask for clarification.

**PyPTO application:** Before writing any kernel code, verify API signatures. Docs follow a strict 1:1 file convention — `pypto.amax` lives at `docs/zh/api/operation/pypto-amax.md`:

- **Known op name** → `Read docs/zh/api/operation/pypto-<op>.md` (e.g., `pypto-amax.md`, `pypto-view.md`, `pypto-matmul.md`)
- **Keyword / constraint search** → `Grep -rn "<keyword>" docs/zh/api/operation/` (e.g., `Grep -rn "32-byte alignment" docs/zh/api/operation/`)
- **All ops overview (117 files)** → `Read docs/zh/api/operation/index.md` or `Glob docs/zh/api/operation/pypto-*.md`

If the op index does not cover a constraint, state the gap. Do not guess dtype or alignment rules.

---

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that was not requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

**The test:** Would a senior engineer say this is overcomplicated? If yes, simplify.

**PyPTO application:** Every module should be the minimum viable implementation that passes `detailed_tensor_compare`. Do not add speculative optimization, extra loop unrolling, or unused tile configurations.

---

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Do not "improve" adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it — do not delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Do not remove pre-existing dead code unless asked.

**The test:** Every changed line should trace directly to the current task.

**PyPTO application:** Do not "improve" frozen modules. Do not refactor working golden code. Do not edit upstream staged files when a downstream module fails — inspect the failing boundary first.

---

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

| Instead of... | Transform to... |
|---------------|-----------------|
| "Implement Phase M1" | "M1 passes `detailed_tensor_compare` with `all_close: true` on all outputs" |
| "Fix precision" | "Identify diverging checkpoint via bisection, apply fix, re-run compare" |
| "Optimize performance" | "Reduce kernel time by N% while layout check exits 0 and all outputs pass compare" |

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These principles are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, clarifying questions come before implementation rather than after mistakes, and every module passes on fewer attempts.
