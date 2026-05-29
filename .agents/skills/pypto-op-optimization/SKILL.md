---
name: pypto-op-optimization
description: Config-level and algorithm-level optimization. Starts only after the final production design is numerically correct.
---

# PyPTO Complex Kernel — Optimization

Optimization starts only after the final production design is numerically correct.

## Config-level optimization

Before searching blindly, retrieve tiling patterns from existing production kernels:
```bash
grep -rn "set_vec_tile_shapes\|set_cube_tile_shapes" custom/ examples/ models/
grep -rn "loop_unroll\|stitch" custom/ examples/ models/
```

Search space may include: vector tile shape, cube tile shape, runtime options, stitch settings, loop unroll options, device scheduling options, reuse settings.

Use constrained search:
1. evaluate 10 initial candidates,
2. keep top candidates,
3. mutate locally,
4. stop when improvement stalls.

Reject any candidate that breaks correctness, times out, exceeds memory limits, or fails to compile.

## Algorithm-level optimization

Only after config-level tuning stabilizes.

Check systematically:
- can intermediate tensors be reduced?
- can data movement be reduced?
- can cast count be reduced?
- can reuse be increased?
- can loop order be improved?
- can memory-bound stages be simplified?
- can view/reshape/assemble count be reduced?

Change one algorithmic idea at a time.

## Subskill delegation: systematic performance tuning

For a more systematic, multi-stage performance analysis and tuning workflow, read skill `pypto-op-perf-tune` (SKILL.md auto-loads). It provides a 3-stage approach:

1. **Frontend tuning** (skill `pypto-op-perf-tune`'s `tune-frontend/SKILL.md`): loop write patterns, TileShape settings, data operation optimization.
2. **Swimlane tuning** (skill `pypto-op-perf-tune`'s `tune-swimlane/SKILL.md`): stitch tuning, deep TileShape tuning, graph fusion, scheduling strategy optimization via swimlane diagram analysis.
3. **Incore tuning** (skill `pypto-op-perf-tune`'s `tune-incore/SKILL.md`): single-task instruction-level optimization, incore pipeline, operation implementation optimization.

Additional automation scripts (swimlane data extraction, AIV dependency chain analysis, leafhash → code mapping) live under skill `pypto-op-perf-tune`'s `tune-swimlane/scripts/` and `tune-swimlane/references/leafhash-to-code-mapping.md`. Load them via the `tune-swimlane` sub-skill above when running stage 2.

**Kernel-complex override:** The correctness guard below still applies. Every tuning change must be validated with `detailed_tensor_compare` on all outputs before proceeding.

## Correctness guard

After every optimization change:
1. Re-run `test_<operator_name>.py` — all outputs must still pass `detailed_tensor_compare`.
2. If correctness regresses, **roll back immediately** and log the failed attempt in the memory.
3. Do not accumulate multiple optimization changes before re-testing.
